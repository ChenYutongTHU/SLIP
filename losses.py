# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import utils
from utils import get_rank, get_world_size


class ClipInfoCELoss_unidirectional(_Loss):
    def __init__(self):
        super(ClipInfoCELoss_unidirectional, self).__init__()
    def forward(self, logits):
        bs, l_bs = logits.shape
        if l_bs == bs:
            labels = torch.arange(len(logits)).cuda()
        else:
            labels = get_rank() * bs + torch.arange(0, bs, dtype=torch.long).cuda()

        loss = F.cross_entropy(logits, labels)        
        pred = torch.argmax(logits, dim=-1) #bs,
        acc = (torch.sum(pred==labels)/bs).item()
        return loss, acc*100

class ClipInfoCELoss(_Loss):
    def __init__(self):
        super(ClipInfoCELoss, self).__init__()
    def forward(self, logits_per_image, logits_per_text):
        bs, l_bs = logits_per_image.shape
        if l_bs == bs:
            labels = torch.arange(len(logits_per_image)).cuda()
        else:
            labels = get_rank() * bs + torch.arange(0, bs, dtype=torch.long).cuda()

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        #loss = (loss_i+loss_t)/2
        
        pred_i = torch.argmax(logits_per_image, dim=-1) #bs,
        pred_t = torch.argmax(logits_per_text, dim=-1) #bs,
        acc_i = (torch.sum(pred_i==labels)/bs).item()
        acc_t = (torch.sum(pred_t==labels)/bs).item()
        return loss_i, loss_t, acc_i*100, acc_t*100

class ClipInfoCELoss2(_Loss):
    def __init__(self, mode='log_plus'):
        super(ClipInfoCELoss2, self).__init__()
        self.mode=mode
    def forward(self, logits):
        bs, l_bs = logits.shape
        assert l_bs%2==0, logits.shape
        if l_bs == bs*2: #gpu=1
            labels1 = torch.arange(bs).cuda()
            labels2 = bs+labels1
        else:
            labels1 = get_rank()*bs + torch.arange(bs).cuda()
            labels2 = labels1+ (bs*get_world_size())

        if self.mode=='single_positive':
            N, r = get_world_size(), get_rank()
            index1, index2 = [], []
            for i in range(bs):
                row1 = [torch.arange(labels2[i]), torch.arange(labels2[i]+1, l_bs)]
                row2 = [torch.arange(labels1[i]), torch.arange(labels1[i]+1, l_bs)]
                index1.append(torch.cat(row1,dim=0))
                index2.append(torch.cat(row2,dim=0))
            index1 = torch.stack(index1, dim=0).to(logits.device) #BS,l_bs-1
            index2 = torch.stack(index2, dim=0).to(logits.device) #BS,l_bs-1
            labels2 = labels2 - 1 #only one single positive pairs in the denominator

            logits1 = torch.gather(logits, dim=1, index=index1)
            logits2 = torch.gather(logits, dim=1, index=index2)
            loss1 = F.cross_entropy(logits1, labels1)
            loss2 = F.cross_entropy(logits2, labels1)
            preds1 = torch.argmax(logits1, dim=-1)#bs,
            preds2 = torch.argmax(logits2, dim=-1)#bs,
            acc1 = (torch.sum(preds1==labels1))/bs
            acc2 = (torch.sum(preds2==labels2))/bs
            loss = loss1+loss2
            #print(r, 'index1:', index1, 'index2:',index2, 'logits1.shape', logits1.shape, 'labels12', labels1, labels2)

        elif self.mode=='log_plus':
            loss1 = F.cross_entropy(logits, labels1)
            loss2 = F.cross_entropy(logits, labels2)
            loss = loss1+loss2
            preds = torch.argmax(logits, dim=-1)#bs,
            acc1 = (torch.sum(preds==labels1))/bs
            acc2 = (torch.sum(preds==labels2))/bs
        elif self.mode=='plus_log':
            target_id = torch.arange(l_bs, device=labels1.device).unsqueeze(0).tile(bs,1) #bs,l_bs
            labels1_, labels2_ = labels1[:,None], labels2[:,None] #bs, 1
            labels = torch.cat([labels1_, labels2_],dim=-1)
            #target = ((target_id==labels1_)+(target_id==labels2_)).float() #bs, l_bs 128,2048
            selected_logits = torch.gather(logits,dim=1,index=labels)
            nominator = torch.logsumexp(selected_logits, dim=-1) #bs
            denumerator = torch.logsumexp(logits, dim=-1) #bs
            loss = torch.mean(denumerator- nominator) #AVERAGE on batch
            preds = torch.argmax(logits, dim=-1)#bs,
            acc1 = (torch.sum(preds==labels1))/bs
            acc2 = (torch.sum(preds==labels2))/bs
        else:
            raise ValueError
        return loss, (acc1*100).item(), (acc2*100).item()
        
class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        image_embed = outputs['image_embed']
        text_embed = outputs['text_embed']
        logit_scale = outputs['logit_scale']
        local_batch_size = image_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather features from all GPUs
        image_embed_all, text_embed_all = \
            utils.all_gather_batch([image_embed, text_embed])

        # cosine similarity as logits
        logits_per_image = logit_scale * image_embed @ text_embed_all.t()
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()

        loss = (F.cross_entropy(logits_per_image, self.labels) + \
            F.cross_entropy(logits_per_text, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'clip_loss': loss, 'clip_acc': acc}


class SIMCLRLoss(nn.Module):
    """
    This is the SimCLR loss in https://arxiv.org/abs/2002.05709
    The embedding vectors are assumed to have size (2 x batch_size, embedding_dim) and
    the memory layout that can be reshaped into shape (2, batch_size, embedding_dim).
    This memory layout is consistent with the SimCLR collator in
    https://github.com/facebookresearch/vissl/blob/master/vissl/data/collators/simclr_collator.py
    Config params:
        temperature (float): the temperature to be applied on the logits
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature
        self.labels = None
        self.masks = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        q_a = outputs['aug1_embed']
        q_b = outputs['aug2_embed']

        q_a = F.normalize(q_a, dim=-1, p=2)
        q_b = F.normalize(q_b, dim=-1, p=2)

        local_batch_size = q_a.size(0)

        k_a, k_b = utils.all_gather_batch_with_grad([q_a, q_b])

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=q_a.device
            )
            total_batch_size = local_batch_size * utils.get_world_size()
            self.masks = F.one_hot(self.labels, total_batch_size) * 1e9
            self.last_local_batch_size = local_batch_size

        logits_aa = torch.matmul(q_a, k_a.transpose(0, 1)) / self.tau
        logits_aa = logits_aa - self.masks
        logits_bb = torch.matmul(q_b, k_b.transpose(0, 1)) / self.tau
        logits_bb = logits_bb - self.masks
        logits_ab = torch.matmul(q_a, k_b.transpose(0, 1)) / self.tau
        logits_ba = torch.matmul(q_b, k_a.transpose(0, 1)) / self.tau

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), self.labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), self.labels)
        loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(torch.cat([logits_ab, logits_aa], dim=1), dim=-1)
            correct = pred.eq(self.labels).sum()
            acc = 100 * correct / local_batch_size

        return {'loss': loss, 'ssl_loss': loss, 'ssl_acc': acc}


class SLIPLoss(nn.Module):
    def __init__(self, ssl_loss, ssl_scale):
        super().__init__()
        self.clip_loss = CLIPLoss()
        self.ssl_loss = ssl_loss
        self.ssl_scale = ssl_scale

    def forward(self, outputs):
        clip_loss_dict = self.clip_loss(outputs)
        clip_loss = clip_loss_dict['clip_loss']
        clip_acc = clip_loss_dict['clip_acc']

        ssl_loss_dict = self.ssl_loss(outputs)
        ssl_loss = ssl_loss_dict['ssl_loss']
        ssl_acc = ssl_loss_dict['ssl_acc']

        return {'loss': clip_loss + self.ssl_scale * ssl_loss,
                'clip_loss': clip_loss,
                'clip_acc': clip_acc,
                'ssl_loss': ssl_loss,
                'ssl_acc': ssl_acc}
