import os
PWD=os.path.abspath(os.getcwd())
PATH2ALTCLIP=os.path.abspath(os.path.join(PWD,'../Multilingual-CLIP'))
import sys
sys.path.append(PATH2ALTCLIP)
from multilingual_clip import pt_multilingual_clip
import transformers
import open_clip

import torch.nn as nn
import torch.nn.functional as F
import torch
model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus'
class MClip(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        # Load Model & Tokenizer
        self.text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name).to(device)
        self.tokenizer_ = transformers.AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = lambda txt: self.tokenizer_(
            txt, padding='max_length',
            truncation=True, max_length=77,
            return_tensors='pt')
        self.vision_model, _, self.transform = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
        self.vision_model.to(device)
        self.logit_scale = torch.tensor(1.0) #DIRTY SOLUTION

    def normalize(self, x, eps=0):
        return x/(x.norm(dim=-1, keepdim=True)+eps)
    
    def encode_text(self, zh_input_ids, en_input_ids, zh_attention_mask, en_attention_mask):
        lang2text_features = {'zh':None, 'en':None}

        embs = self.text_model.transformer(input_ids=zh_input_ids, attention_mask=zh_attention_mask)[0]
        att = zh_attention_mask
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None] #B,D
        embs = self.text_model.LinearTransformation(embs)
        lang2text_features['zh'] = self.normalize(embs)

        #DIRTY SOLUTION
        lang2text_features['en'] = lang2text_features['zh']

        return lang2text_features['zh'], lang2text_features['en']

    def encode_image(self, images):
        #import ipdb; ipdb.set_trace()
        image_features = self.vision_model.encode_image(images)
        x = self.normalize(image_features)
        return x