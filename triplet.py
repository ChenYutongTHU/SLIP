from utils import freeze_params, get_rank, unfreeze_params, get_world_size
from losses import ClipInfoCELoss, ClipInfoCELoss2
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import defaultdict
import wukong, clip
from copy import deepcopy
import numpy as np
import torch.distributed as dist

class AllGather(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, tensor):
        ctx.rank = get_rank()
        ctx.world_size = get_world_size()

#         y = tensor.new(ctx.world_size, *tensor.size())
        
        y = [tensor.new(*tensor.size()) for _ in range(ctx.world_size)]
        
        dist.all_gather(y, tensor)

        y = torch.cat(y, 0).view(-1, *tensor.size())
        
        return y

    @staticmethod
    def backward(ctx, grad_output):
        in_grad = torch.zeros_like(grad_output)
        in_grad.copy_(grad_output)
        # sum grad for gathered tensor
        dist.all_reduce(in_grad)
        # split
        return in_grad[ctx.rank]


class Triplet(torch.nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        device = 'cpu'
        self.use_allgather = cfg['use_allgather']
        self.model_en, self.preprocess = clip.load(cfg['Model_En'], device=device)
        self.tokenize_en = clip.tokenize
        self.model_zh, self.tokenize_zh = wukong.load(pkl_path=cfg['Model_Zh'],device=device,lang='zh',context_length=32)
        
        self.model_weights_check()
        self.visual = deepcopy(self.model_en.visual)
        del self.visual.proj
        self.visual.proj = None
        
        if cfg.get('from_scratch', False):
            print('Reinitialize model')
            self.model_zh.initialize_parameters()
            self.model_en.initialize_parameters()
                        
        if cfg['visual_proj'] == 'scratch':
            width = self.visual.conv1.out_channels
            scale = width ** -0.5
            output_dim = self.visual.output_dim
            self.visual_proj = nn.Parameter(scale * torch.randn(width, output_dim))
        

        elif cfg['visual_proj'] == 'load_en':
            self.visual_proj = deepcopy(self.model_en.visual.proj)
        elif cfg['visual_proj'] == 'loca_zh':
            self.visual_proj = deepcopy(self.model_zh.visual.proj)
        if self.model_en.dtype==torch.float16:
            self.visual_proj.data = self.visual_proj.data.half()            
        
        del self.model_en.visual
        del self.model_zh.visual

        if cfg['logit_scale'] == 'scratch':
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        elif cfg['logit_scale'] == 'load_en':
            self.logit_scale = deepcopy(self.model_en.logit_scale) #4.61
        elif cfg['logit_scale'] == 'load_zh':
            self.logit_scale = deepcopy(self.model_zh.logit_scale) #4.30    
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * float(cfg['logit_scale']))     
        del self.model_en.logit_scale
        del self.model_zh.logit_scale
        
        print('Logit_scale={:.2f}({}) Visual_proj={}'.format(self.logit_scale, cfg['logit_scale'], cfg['visual_proj']))
        self.contrastive_type = cfg['contrastive_type']
        if self.contrastive_type in ['one_one','one_mean']:
            self.criterion = ClipInfoCELoss()
        elif self.contrastive_type=='one_two':
            self.criterion = ClipInfoCELoss2(mode=cfg.get('one_two_loss_mode','log_plus'))
        
        if 'loss_weight' in cfg:
            self.loss_weight = cfg['loss_weight']
            for k, v in self.loss_weight.items():
                print('{}:{:.2f}'.format(k,v))
        else:
            self.loss_weight = defaultdict(lambda :1)
        self.label_requires_grad = cfg['label_requires_grad']
        
        #convert_weights(self)
        if cfg.get('trainable_modules', 'all')!='all':
            assert cfg.get('from_scratch', False)==False
            freeze_params(self.model_zh) #text encoder (zh)
            freeze_params(self.model_en) #text encoder (en)
            freeze_params(self.visual) #visual encoder
            if 'text_encoder' in cfg['trainable_modules']:
                unfreeze_params(self.model_zh.transformer)
                unfreeze_params(self.model_zh.token_embedding)
                self.model_zh.positional_embedding.requires_grad = True
                unfreeze_params(self.model_zh.ln_final)
                unfreeze_params(self.model_en.transformer)
                self.model_en.positional_embedding.requires_grad = True
                unfreeze_params(self.model_en.token_embedding)
                unfreeze_params(self.model_en.ln_final)
                self.model_zh.text_projection.requires_grad = True
                self.model_en.text_projection.requires_grad = True
            if 'text_projection' in cfg['trainable_modules']:
                self.model_zh.text_projection.requires_grad = True
                self.model_en.text_projection.requires_grad = True
            if 'visual_proj' not in cfg['trainable_modules']:
                self.visual_proj.requires_grad = False
            if 'logit_scale' not in cfg['trainable_modules']:
                self.logit_scale.requires_grad = False
            print('Freeze some parameters; Trainable params:')
            for n, p in self.named_parameters():
                if p.requires_grad==True:
                    print(n)
                    
    def model_weights_check(self):
        assert self.model_zh.visual.input_resolution == self.model_en.visual.input_resolution, (self.model_zh.visual.input_resolution, self.model_en.visual.input_resolution)
        zh_visual_dict = self.model_zh.visual.state_dict()
        en_visual_dict = self.model_en.visual.state_dict()
        for k,v_zh in zh_visual_dict.items():
            if 'proj' == k[:4]:
                print(f'Skip checking weights of {k}')
                continue
            assert k in en_visual_dict, k
            v_en = en_visual_dict[k]
            assert v_zh.shape==v_en.shape, (k, v_zh.shape, v_en.shape)
            assert torch.min(v_zh-v_en)<0.00001, (k)
        print('Check pass. model_zh.visual==model_en.visual (exclude .proj)')
        return
    
    def all_gather(self, input):
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output
    
    def encode_image(self, img):
        img_x = self.visual(img.type(self.model_en.dtype))[:,0,:] #B,D
        image_features = img_x @ self.visual_proj
        return image_features
            
    def forward(self, img, en, zh):
        image_features = self.encode_image(img)
        # print('image_features', torch.max(torch.isnan(image_features)))
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True))
        
        zh_text_features = self.model_zh.encode_text(zh)
        zh_text_features = zh_text_features / (zh_text_features.norm(dim=-1, keepdim=True)+1e-10)
        en_text_features = self.model_en.encode_text(en)
        en_text_features = en_text_features / (en_text_features.norm(dim=-1, keepdim=True)+1e-10)

        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)
        
        features_dict = {'image': image_features, 
                         'zh_text': zh_text_features,
                         'en_text': en_text_features}
        loss_dict = {'total':0}
        acc_dict = {}
        if self.training and self.use_allgather:
            gathered_features_dict = {}
            for k in features_dict:
                gathered_features_dict[k] = self.all_gather(features_dict[k])    
        else:
            gathered_features_dict = features_dict
        if self.contrastive_type=='one_one':
            for k1,k2 in [['image','zh_text'],['image','en_text'],['zh_text', 'en_text']]:
                f1, f2 = features_dict[k1], features_dict[k2]
                gathered_f1, gathered_f2 = gathered_features_dict[k1], gathered_features_dict[k2]
                if self.label_requires_grad==False:
                    gathered_f1 = gathered_f1.detach()
                    gathered_f2 = gathered_f2.detach()
                logits1 = logit_scale*f1@gathered_f2.t()
                logits2 = logit_scale*f2@gathered_f1.t()
                loss1_2, loss2_1, acc1_2, acc2_1 = self.criterion(logits1, logits2)
                acc_dict[f'{k1}->{k2}'], acc_dict[f'{k2}->{k1}'] = acc1_2, acc2_1
                loss_dict[f'{k1}->{k2}'], loss_dict[f'{k2}->{k1}'] = loss1_2, loss2_1
                loss_dict['total'] += (
                    loss_dict[f'{k1}->{k2}']*self.loss_weight[f'{k1}->{k2}']+ \
                        loss_dict[f'{k2}->{k1}']*self.loss_weight[f'{k2}->{k1}'])
        elif self.contrastive_type=='one_mean':
            for k0 in features_dict:
                k1,k2 = sorted([k for k in features_dict if not k==k0])
                f0 = features_dict[k0]  
                gathered_f0 = gathered_features_dict[k0]

                gathered_f1, gathered_f2 = gathered_features_dict[k1], gathered_features_dict[k2]
                gathered_f12 = (gathered_f1+gathered_f2)/2
                gathered_f12 = gathered_f12 / (gathered_f12.norm(dim=-1, keepdim=True)+1e-10)
                
                f1, f2 = features_dict[k1], features_dict[k2]
                f12 = (f1+f2)/2
                f12 = f12 / (f12.norm(dim=-1, keepdim=True)+1e-10)

                if self.label_requires_grad==False:
                    gathered_f0 = gathered_f0.detach()
                    gathered_f12 = gathered_f12.detach()
                logits1 = logit_scale*f0@gathered_f12.t()
                logits2 = logit_scale*f12@gathered_f0.t()
                loss1_2, loss2_1, acc1_2, acc2_1 = self.criterion(logits1, logits2)
                acc_dict[f'{k0}->mean({k1},{k2})'], acc_dict[f'mean({k1},{k2})->{k0}'] = acc1_2, acc2_1
                loss_dict[f'{k0}->mean({k1},{k2})'], loss_dict[f'mean({k1},{k2})->{k0}'] = loss1_2, loss2_1
                loss_dict['total'] += (
                    loss_dict[f'{k0}->mean({k1},{k2})']*self.loss_weight[f'{k0}->mean({k1},{k2})']+ \
                        loss_dict[f'mean({k1},{k2})->{k0}']*self.loss_weight[f'mean({k1},{k2})->{k0}'])
        elif self.contrastive_type=='one_two':
            for k0 in features_dict:
                f0 = features_dict[k0]    
                k1,k2 = sorted([k for k in features_dict if not k==k0])
                f12 = torch.cat([gathered_features_dict[k1], gathered_features_dict[k2]], dim=0)  
                if self.label_requires_grad==False:
                    f12 = f12.detach()
                #print(f0.dtype, f12.dtype)
                logits = logit_scale*f0@f12.t()
                #print('logit_scale ',logit_scale)
                loss, acc1, acc2 = self.criterion(logits)
                loss_dict[k0] = loss
                loss_dict['total'] += loss_dict[k0]*self.loss_weight[k0]
                acc_dict[f'{k0}->{k1}'], acc_dict[f'{k0}->{k2}'], acc_dict[f'{k0}->{k1}|{k2}'] = acc1, acc2, acc1+acc2
        return loss_dict, features_dict, acc_dict
            
        
        
        
        
        
    # def encode_text(self, text):
    #     x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

    #     x = x + self.positional_embedding.type(self.dtype)
    #     x = x.permute(1, 0, 2)  # NLD -> LND
    #     x, _ = self.transformer(x)
    #     x = x.permute(1, 0, 2)  # LND -> NLD
    #     x = self.ln_final(x).type(self.dtype)

    #     # x.shape = [batch_size, n_ctx, transformer.width]
    #     # take features from the eot embedding (eot_token is the highest number in each sequence)
    #     # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
    #     if self.return_full_embed==False:
    #         eot_loc = torch.sum(text!=0, dim=-1) #B,L -> B
    #         x = x[torch.arange(x.shape[0]), eot_loc-1] @ self.text_projection
    #     else:
    #         x = x@self.text_projection
    #     return x