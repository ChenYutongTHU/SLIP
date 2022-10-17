from multiprocessing import context
import torch, pickle, os
import numpy as np
from typing import Any, Union, List
import sys
sys.path.append('CLIP_Wukong')
from clip.model import build_model
from .simple_tokenizer_wukong import set_tokenizer_lang, tokenize

def build_wukong_clip(pkl_path):
    with open(pkl_path,'rb') as f:
        state_dict_np = pickle.load(f)
    state_dict_torch = {}
    for k,v in state_dict_np.items():
        if k in ['transformer.token_embedding.weight', 'transformer.ln_final.weight', 
            'transformer.ln_final.bias', 'transformer.text_projection', 'transformer.positional_embedding']:
            k_ = k.replace('transformer.','')
        elif k=='loss.logit_scale':
            k_ = k.replace('loss.','')
        else:
            k_ = k 
        state_dict_torch[k_] = torch.tensor(v)
    #debug
    model_wukong_clip = build_model(state_dict_torch)
    return model_wukong_clip


class wukong_tokenizer(object):
    def __init__(self, lang='zh', context_length=32, padding='max'):
        self._tokenizer = set_tokenizer_lang(lang=lang, context_length=context_length)
        self.padding = padding
    def __call__(self, texts: Union[str, List[str]]) -> np.ndarray:
        return tokenize(texts, padding=self.padding)
    def ids2tokens(self, ids):
        #ids list of ids
        return [self._tokenizer.decoder[i] for i in ids]



def load(pkl_path: str, 
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        lang='zh', context_length=32):
    #model
    if 'clip' in pkl_path:
        model = build_wukong_clip(pkl_path)
    else:
        raise ValueError
    model = model.to(device)
    if str(device) == "cpu":
        model.float()
    tokenizer= wukong_tokenizer(lang=lang, context_length=context_length)
    return model, tokenizer


