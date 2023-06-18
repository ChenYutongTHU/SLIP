from utils import freeze_params, get_rank, unfreeze_params, get_world_size
from losses import ClipInfoCELoss, ClipInfoCELoss2, ClipInfoCELoss_unidirectional
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import defaultdict
import wukong, clip, mengzi
from clip.model import Transformer, X_Attn_Encoder
from copy import deepcopy
import numpy as np
import torch.distributed as dist
from transformers import BertTokenizer, BertModel, BertConfig


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

        self.both_en = cfg.get('both_en',None) #'different_rand')
        if self.both_en==None:
            self.lang = ['en','zh']
            if 'wukong' in cfg['Model_Zh'].lower():
                self.model_zh_type = 'wukong'
                self.model_zh, self.tokenize_zh = wukong.load(pkl_path=cfg['Model_Zh'],device=device,lang='zh',
                                                              context_length=cfg.get('zh_context_length', 32))
            elif 'mengzi' in cfg['Model_Zh'].lower():
                self.model_zh_type = 'mengzi'
                output_dim = self.model_en.embed_dim
                self.model_zh, self.tokenize_zh = mengzi.load(model_path=cfg['Model_Zh'], device=device, output_dim=output_dim, context_length=32)
                #import ipdb; ipdb.set_trace()
                # print(self.model_zh.dtype)
        else:
            self.lang = ['en']
            if self.both_en=='different_rand':
                self.model_zh_type = 'wukong'
                self.model_zh = deepcopy(self.model_en)
                self.tokenize_zh = self.tokenize_en
            elif self.both_en=='different_arc':
                self.model_zh_type = 'bert'
                self.model_zh = BertModel(
                    config=BertConfig.from_pretrained("bert-base-uncased"),
                    add_pooling_layer=False)
                self.model_zh_text_projection = torch.nn.Linear(
                        self.model_zh.config.hidden_size, self.model_en.embed_dim)
                self.tokenize_zh_ = BertTokenizer.from_pretrained('bert-base-uncased')
                self.tokenize_zh = lambda txt: self.tokenize_zh_(
                        txt, padding='max_length',
                        truncation=True, max_length=77,
                        return_tensors='pt')['input_ids']
            else:
                raise ValueError

        if cfg.get('visual','load_en')=='load_en':
            self.visual = deepcopy(self.model_en.visual)
        else:
            self.visual = deepcopy(self.model_zh.visual)
        del self.visual.proj
        self.visual.proj = None

        
        if cfg.get('from_scratch', False):
            print('Reinitialize model')
            if self.model_zh_type != 'bert':
                self.model_zh.initialize_parameters()
            self.model_en.initialize_parameters() 
            #import ipdb; ipdb.set_trace() 

        if cfg['visual_proj'] == 'scratch':
            width = self.visual.conv1.out_channels
            scale = width ** -0.5
            output_dim = self.visual.output_dim
            self.visual_proj = nn.Parameter(scale * torch.randn(width, output_dim))
        

        elif cfg['visual_proj'] == 'load_en':
            self.visual_proj = deepcopy(self.model_en.visual.proj)
        elif cfg['visual_proj'] == 'load_zh':
            self.visual_proj = deepcopy(self.model_zh.visual.proj)
        if self.model_en.dtype==torch.float16:
            self.visual_proj.data = self.visual_proj.data.half()            
        
        del self.model_en.visual
        if self.model_zh_type=='wukong':
            del self.model_zh.visual

        if cfg['logit_scale'] == 'scratch':
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        elif cfg['logit_scale'] == 'load_en':
            self.logit_scale = deepcopy(self.model_en.logit_scale) #4.61
        elif cfg['logit_scale'] == 'load_zh':
            assert self.model_zh_type=='wukong'
            self.logit_scale = deepcopy(self.model_zh.logit_scale) #4.30    
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * float(cfg['logit_scale']))     
        del self.model_en.logit_scale
        if self.model_zh_type=='wukong':
            del self.model_zh.logit_scale
        
        # print('Logit_scale={:.2f}({}) Visual_proj={}'.format(self.logit_scale, cfg['logit_scale'], cfg['visual_proj']))
        # self.contrastive_type = cfg['contrastive_type']
        # if self.contrastive_type in ['one_one','one_mean']:
        #     self.criterion = ClipInfoCELoss()
        # elif self.contrastive_type=='one_two':
        #     self.criterion = ClipInfoCELoss2(mode=cfg.get('one_two_loss_mode','log_plus'))
        
        if cfg.get('share_text_encoder',False)==True:
            print('Share text encoder')
            # del self.model_zh
            # self.model_zh = self.model_en    
            assert self.model_zh.context_length<=self.model_en.context_length and self.model_zh_type=='wukong' #32,77
            assert cfg.get('from_scratch', False)==True
            del self.model_zh.transformer 
            del self.model_zh.positional_embedding
            del self.model_zh.ln_final
            self.model_zh.transformer = self.model_en.transformer
            self.model_zh.positional_embedding = self.model_en.positional_embedding
            self.model_zh.ln_final = self.model_en.ln_final
            self.model_zh.context_length = self.model_en.context_length #77
            from wukong.simple_tokenizer_wukong import set_tokenizer_lang
            self.tokenize_zh._tokenizer = set_tokenizer_lang(lang='zh', context_length=self.model_en.context_length)
            #language-specific: token_embedding/text_projection

        self.only_distillation = False
        if 'loss_weight' in cfg:
            if list(cfg['loss_weight'].keys()) == ['distillation']:
                self.only_distillation = True
            #adapt to new version
            loss_weight_load  = {}
            for k, v in cfg['loss_weight'].items():
                if k in ['image','en_text','zh_text']:
                    if k=='image':
                        k = 'image->en_text|zh_text'
                    elif k=='en_text':
                        k = 'en_text->image|zh_text'
                    else:
                        k = 'zh_text->en_text|image'
                    cfg['one_two_loss_mode'] = cfg.get('one_two_loss_mode','log_plus') 
                    if 'one_two_loss_mode' in cfg:
                        assert cfg['one_two_loss_mode'] in ['plus_log', 'log_plus', 'single_positive'], cfg['one_two_loss_mode']
                        k = k+'#'+cfg['one_two_loss_mode']
                elif k=='distillation':
                    assert cfg['visual_proj'] == 'load_en'
                    assert cfg['from_scratch'] == False 
                    assert cfg['trainable_modules'] in [['zh_text_encoder'], ['zh_text_projection']] 
                    assert cfg['reinitialize_modules'] == ['zh_text_projection']
                    pass
                else:
                    pass
                loss_weight_load[k] = v
            self.loss_weight = defaultdict(lambda :0, loss_weight_load) 
            for k, v in self.loss_weight.items():
                print('{}:{:.2f}'.format(k,v))
        else:
            self.loss_weight = defaultdict(lambda :0)
            raise ValueError 
        self.label_requires_grad = cfg['label_requires_grad']
        
        #convert_weights(self)
        if cfg.get('freeze_visual', False) == True:
            print('Freeze visual encoder')
            freeze_params(self.visual)
        if cfg.get('trainable_modules', 'all')!='all':
            assert cfg.get('from_scratch', False)==False
            freeze_params(self.model_zh) #text encoder (zh)
            freeze_params(self.model_en) #text encoder (en)
            freeze_params(self.visual) #visual encoder
            if 'visual' in cfg['trainable_modules']:            
                unfreeze_params(self.visual)
            if 'zh_text_encoder' in cfg['trainable_modules']:
                if self.model_zh_type=='mengzi':
                    unfreeze_params(self.model_zh)
                else:
                    unfreeze_params(self.model_zh.transformer)
                    unfreeze_params(self.model_zh.token_embedding)
                    self.model_zh.positional_embedding.requires_grad = True
                    unfreeze_params(self.model_zh.ln_final)
                    self.model_zh.text_projection.requires_grad = True
            if 'zh_text_projection' in cfg['trainable_modules']:
                self.model_zh.text_projection.requires_grad = True
            if 'en_text_projection' in cfg['trainable_modules']:
                self.model_en.text_projection.requires_grad = True
            if 'en_text_encoder' in cfg['trainable_modules']:
                unfreeze_params(self.model_en.transformer)
                self.model_en.positional_embedding.requires_grad = True
                unfreeze_params(self.model_en.token_embedding)
                unfreeze_params(self.model_en.ln_final)
                self.model_en.text_projection.requires_grad = True
            if 'visual_proj' not in cfg['trainable_modules']:
                self.visual_proj.requires_grad = False
            if 'logit_scale' not in cfg['trainable_modules']:
                self.logit_scale.requires_grad = False
            # print('Freeze some parameters; Trainable params:')
            # for n, p in self.named_parameters():
            #     if p.requires_grad==True:
            #         print(n)
        if cfg.get('reinitialize_modules', [])!=[]:
            if 'zh_text_projection' in cfg['reinitialize_modules']:
                if self.model_zh_type == 'wukong':
                    nn.init.normal_(self.model_zh.text_projection, std=self.model_zh.transformer.width ** -0.5) 

        if 'x_attn_encoder' in cfg:
            self.self_attn_layers = self.model_zh.transformer.layers-cfg['x_attn_encoder']['layers']
            self.x_attn_encoder = X_Attn_Encoder(
                **cfg['x_attn_encoder'], 
                width=self.model_en.transformer.width, 
                heads=self.model_en.transformer.heads,
                embed_dim=self.model_en.embed_dim, 
                context_length=self.model_en.context_length+self.model_zh.context_length)
            self.x_attn_encoder.initialize_parameters() 
        else:
            self.self_attn_layers = -1
            self.x_attn_encoder = None

        self.use_en_text_encoder = cfg.get('use_en_text_encoder', True)
        if self.use_en_text_encoder == False:
            del self.model_en
            
    #     self.model_weights_check()
        self.mask_clip = cfg.get('mask_clip',False) #New feature
        if self.mask_clip:
            self.visual.transformer.enable_mask_clip()
            assert self.visual.proj is None



    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.bi_context_length, self.bi_context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
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
        img_x, attention_weights = self.visual(img.type(self.model_zh.dtype)) #B,L,D
        image_features = img_x @ self.visual_proj #B,L,D
        if self.mask_clip:
            return image_features, attention_weights
        else:
            return image_features[:,0,:]

    
    def normalize(self, x, eps=0):
        return x/(x.norm(dim=-1, keepdim=True)+eps)
    
    def encode_text(self, zh, en, output_attention=False):
        if self.both_en==None:
            if zh is not None:
                zh_text_features, zh_text_features_imme, zh_eot, zh_attention = self.model_zh.encode_text(zh, self.self_attn_layers)
                zh_text_features = self.normalize(zh_text_features, eps=1e-10)
            else:
                zh_text_features, zh_attention = None, None
        else:
            assert en is not None, en #encode en!!
            if self.model_zh_type == 'wukong':
                zh_text_features, zh_text_features_imme, zh_eot, zh_attention = self.model_zh.encode_text(en, self.self_attn_layers)
            elif self.model_zh_type == 'bert': #encode zh (different tokenizer)
                #import ipdb; ipdb.set_trace()
                attention_mask = (zh!=0).long() #B,L
                zh_text_features = self.model_zh(zh, attention_mask)[0] #B,L,D
                zh_text_features = zh_text_features[:,0,:]
                #import ipdb; ipdb.set_trace()
                zh_text_features = self.model_zh_text_projection(zh_text_features)  #B,D
            zh_text_features = self.normalize(zh_text_features, eps=1e-10)


        if en is not None and self.use_en_text_encoder:
            en_text_features, en_text_features_imme, en_eot, en_attention = self.model_en.encode_text(en, self.self_attn_layers)
            en_text_features = self.normalize(en_text_features, eps=1e-10)
        else:
            en_text_features, en_attention = None, None

        if self.x_attn_encoder is not None:
            bi_text_features = self.x_attn_encoder(zh=zh_text_features_imme, en=en_text_features_imme,
                    zh_eot=zh_eot, en_eot=en_eot)
            bi_text_features = self.normalize(bi_text_features)
        else:
            bi_text_features = None
        if output_attention:
            return zh_text_features, en_text_features, bi_text_features, zh_attention, en_attention
        else:
            return zh_text_features, en_text_features, bi_text_features

    def forward_only_distillation(self, en, zh):
        with torch.no_grad():
            en_text_features, _, _, _ = self.model_en.encode_text(en) #B,D
        zh_text_features, _, _, _ = self.model_zh.encode_text(zh) #B,D
        loss_dict = {}
        loss_dict['distillation_loss'] = torch.nn.functional.mse_loss(zh_text_features, en_text_features, reduction='mean') 
        loss_dict['distillation_loss'] *= self.loss_weight['distillation']
        loss_dict['distill_total_loss'] = loss_dict['distillation_loss']
        return loss_dict
    
    def forward_distill_am(self, zh1, zh2):
        zh1_text_features, _, _, _ = self.model_zh.encode_text(zh1)
        zh2_text_features, _, _, _ = self.model_zh.encode_text(zh2)
        loss_dict = {}
        #loss_dict['distillation_am_loss'] = torch.nn.functional.mse_loss(zh1_text_features, zh2_text_features, reduction='mean')
        loss_dict['distill_am_total_loss'] = torch.nn.functional.mse_loss(zh1_text_features, zh2_text_features, reduction='mean')
        return loss_dict

    def forward(self, mode, en=None, zh=None, img=None, zh1=None, zh2=None):
        if mode == 'distill_am':
            return self.forward_distill_am(zh1=zh1, zh2=zh2), {}
        elif (self.only_distillation and img==None) or mode=='distill':
            return self.forward_only_distillation(en=en, zh=zh), {}, {}
        
        image_features = self.encode_image(img)
        image_features = self.normalize(image_features)

        zh_text_features, en_text_features, bi_text_features = self.encode_text(zh, en)

        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)

        features_dict = {'image': image_features}
        for k,v in [['zh_text',zh_text_features], ['en_text',en_text_features], ['bi_text',bi_text_features]]:
            if v is not None:   
                features_dict[k] = v

        loss_dict = {'contrastive_total_loss':0}
        acc_dict = {}
        if self.training and self.use_allgather:
            gathered_features_dict = {}
            for k in features_dict:
                gathered_features_dict[k] = self.all_gather(features_dict[k])    
        else:
            gathered_features_dict = features_dict
        
        for loss_key, loss_weight in self.loss_weight.items():
            if loss_key=='distillation':
                continue  #compute separately
                loss_dict[loss_key] = torch.nn.functional.mse_loss(zh_text_features, en_text_features, reduction='mean')                 
                loss_dict['total'] += loss_weight*loss_dict[loss_key]
            elif '|' in loss_key:
                #one-two loss
                assert 'bi_text' not in features_dict, 'do not support one-two loss for x_attn'
                k0k1k2, one_two_loss_mode = loss_key.split('#')
                k0, k1k2 = k0k1k2.split('->')
                k1,k2 = sorted(k1k2.split('|'))
                criterion = ClipInfoCELoss2(mode=one_two_loss_mode)

                f0 = features_dict[k0]    
                f12 = torch.cat([gathered_features_dict[k1], gathered_features_dict[k2]], dim=0)  
                if self.label_requires_grad==False:
                    f12 = f12.detach()
                logits = logit_scale*f0@f12.t()
                loss, acc1, acc2 = criterion(logits)
                loss_dict[loss_key+'_loss'] = loss
                loss_dict['contrastive_total_loss'] += loss_weight*loss
                acc_dict[f'{k0}->{k1}({k1}|{k2})_acc'], acc_dict[f'{k0}->{k2}({k1}|{k2})_acc'], acc_dict[f'{k0}->{k1}|{k2}_acc'] = acc1, acc2, acc1+acc2
            else:
                #one-one loss
                criterion = ClipInfoCELoss_unidirectional()
                if 'mean' in loss_key:
                    if loss_key.find('->')<loss_key.find('mean'): #image->mean(en_text,zh_text)
                        k0 = loss_key.split('->')[0]
                        k1, k2 = sorted(loss_key[loss_key.find('mean(')+5:loss_key.find(')')].split(',')) 
                        gathered_f1, gathered_f2 = gathered_features_dict[k1], gathered_features_dict[k2]
                        gathered_f12 = (gathered_f1+gathered_f2)/2
                        gathered_f12 = gathered_f12 / (gathered_f12.norm(dim=-1, keepdim=True)+1e-10)
                        fs = features_dict[k0]
                        gathered_ft = gathered_f12
                    else: #mean(en_text,zh_text)->image
                        k0 = loss_key.split('->')[1]
                        k1, k2 = sorted(loss_key[loss_key.find('mean(')+5:loss_key.find(')')].split(','))                   
                        f1, f2 = features_dict[k1], features_dict[k2]
                        f12 = (f1+f2)/2
                        f12 = f12 / (f12.norm(dim=-1, keepdim=True)+1e-10)
                        fs = f12
                        gathered_ft = gathered_features_dict[k0] 
                else:
                    k0, k1 = loss_key.split('->')[0], loss_key.split('->')[1]
                    fs, ft = features_dict[k0], features_dict[k1]
                    gathered_ft = gathered_features_dict[k1]
                if self.label_requires_grad==False:
                    gathered_ft = gathered_ft.detach()
                logits = logit_scale*fs@gathered_ft.t()
                loss, acc = criterion(logits)
                loss_dict[loss_key+'_loss'] = loss
                loss_dict['contrastive_total_loss'] += loss_weight*loss
                acc_dict[loss_key+'_acc'] = acc       
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