import os
PWD=os.path.abspath(os.getcwd())
PATH2ALTCLIP=os.path.abspath(os.path.join(PWD,'../FlagAI'))
import sys
sys.path.append(PATH2ALTCLIP)
from flagai.auto_model.auto_loader import AutoLoader

import torch.nn as nn
import torch.nn.functional as F
import torch

class AltCLIP(torch.nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        loader = AutoLoader(
            task_name="txt_img_matching",
            model_name="AltCLIP-XLMR-L",   # Load the checkpoints from Modelhub(model.baai.ac.cn/models)
            model_dir=os.path.join(PATH2ALTCLIP,'checkpoints')
        )
        self.model = loader.get_model()
        self.model.to(device)
        self.model.eval()

        tokenizer = loader.get_tokenizer()
        self.tokenizer = lambda txt: tokenizer(txt, padding='max_length',
                                    truncation=True,
                                    max_length=77,
                                    return_tensors='pt')
        self.transform = loader.get_transform()
        self.logit_scale = self.model.logit_scale

    def normalize(self, x):
        x = x / x.norm(dim=-1, keepdim=True)
        return x 

    def encode_text(self, zh_input_ids, en_input_ids, zh_attention_mask, en_attention_mask):
        lang2text_features = {}
        for lang, input_ids, attention_mask in [('zh',zh_input_ids, zh_attention_mask),
                ('en',en_input_ids, en_attention_mask)]:
            lang2text_features[lang]=None
            if input_ids is None:
                continue  
            # import ipdb; ipdb.set_trace()          
            x = self.model.get_text_features(input_ids, attention_mask=attention_mask)
            lang2text_features[lang] = self.normalize(x)

        return lang2text_features['zh'], lang2text_features['en']
    
    def encode_image(self, images):
        x = self.model.get_image_features(images)
        x = self.normalize(x)
        return x
        