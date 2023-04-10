from PIL import Image
import requests
import clip
import torch
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import CLIPProcessor, CLIPModel
import numpy as np

class Taiyi(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.text_tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese")
        self.tokenizer = lambda txt: self.text_tokenizer(
            txt, padding='max_length',
            truncation=True, max_length=77,
            return_tensors='pt')
        self.text_encoder = BertForSequenceClassification.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese").eval()
        self.text_encoder.to(device)

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")  
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.transform = lambda img: self.processor.image_processor(img, return_tensors='pt') 
        self.logit_scale = self.clip_model.logit_scale.exp()
    
    def normalize(self, x):
        x = x / x.norm(dim=-1, keepdim=True)
        return x 
    
    def encode_text(self, zh_input_ids, en_input_ids, zh_attention_mask, en_attention_mask):
        lang2text_features = {'zh':None, 'en':None}
        #import ipdb; ipdb.set_trace()

        text_features = self.text_encoder(zh_input_ids, attention_mask=zh_attention_mask).logits
        #import ipdb; ipdb.set_trace()
        text_features = self.normalize(text_features)
        lang2text_features['zh'] = text_features
        return lang2text_features['zh'], lang2text_features['en']

    def encode_image(self, images):
        import ipdb; ipdb.set_trace()
        image_features = self.clip_model.get_image_features(pixel_values=images)
        import ipdb; ipdb.set_trace()
        x = self.normalize(image_features)
        return x        