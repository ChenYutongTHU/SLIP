import argparse, os, json
import torch, pickle
import numpy as np
import utils, models, Datasets
from tqdm import tqdm
# from PoemSegmentor.segmentor import Poem_Segmenter
# segmentor = Poem_Segmenter()  


def build_and_load_model(model_type, model_cfg_path, model_path):
    print('Building model')
    if model_type=='TRIPLET':
        model = getattr(models, 'TRIPLET')(
            ssl_mlp_dim=4096, ssl_emb_dim=256,
            model_cfg_path=model_cfg_path, toolkit='torch')
    else:
        model = getattr(models, model_type)()
    if model_path!='':
        print('Load model from {}'.format(model_path))
        state_dict = torch.load(model_path, map_location='cpu')['state_dict']
        state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
        model.load_state_dict(state_dict)
    model.eval()
    return model

class Sentence_dataset(torch.utils.data.Dataset):
    def __init__(self, sent_list, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.sent_list = sent_list 
    def __len__(self):
        return len(self.sent_list)
    def __getitem__(self, i):
        return self.tokenizer(self.sent_list[i])
    

def main(args):
    model = build_and_load_model(args.model, args.model_cfg_path, args.model_path)
    model.cuda(args.gpu)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200)
    if args.model=='TRIPLET':
        tokenizer = {'en':utils.get_model(model).tokenize_en, 'zh':utils.get_model(model).tokenize_zh}
    else:
        tokenizer = {'en':utils.get_model(model).tokenizer, 'zh':utils.get_model(model).tokenizer}

    print('Building sentence loader (only support single GPU)')
    sentences = json.load(open(args.sentence_path, 'r')) #list
    if type(sentences[0])==dict:
        sentences = [s['content'] for s in sentences]

    print(f'#sentences={len(sentences)}')
    dataset = Sentence_dataset(sentences, tokenizer['zh'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    if os.path.isfile(args.output_path):
        print(args.output_path+' already exists.')
    else:
        print('Computing embeddings ...')
        result_embeddings = []
        for zh in tqdm(dataloader, total=len(dataloader)):
            if args.model=='TRIPLET':
                if zh.dim()==3:
                    zh = zh.squeeze(1)
                with torch.no_grad():
                    zh_embeddings, _, _, layer_attn_weights, _ = \
                        utils.get_model(model).encode_text(zh=zh.cuda(), en=None, output_attention=True) #already normalize #B,N 
            else:
                with torch.no_grad():
                    if zh['input_ids'].dim()==3:
                        zh['input_ids'] = zh['input_ids'].squeeze(1)
                    if zh['attention_mask'].dim()==3:
                        zh['attention_mask'] = zh['attention_mask'].squeeze(1)                                                    
                    zh_embeddings, _,  = \
                        utils.get_model(model).encode_text(
                        zh_input_ids=zh['input_ids'].cuda(), 
                        zh_attention_mask=zh['attention_mask'].cuda(),
                        en_input_ids=None, en_attention_mask=None) #already normalize #B,                
            result_embeddings.extend([e.cpu() for e in zh_embeddings])  

        print('Saving embedding as {}'.format(args.output_path))
        torch.save(result_embeddings, args.output_path)
   
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='compute sentence embeddings', add_help=False)
    parser.add_argument('--model', default='TRIPLET')
    parser.add_argument('--model_cfg_path', default='experiments/configs_duet/zh_large_tune_all.yaml')
    parser.add_argument('--model_path', default='') 
    parser.add_argument('--sentence_path', required=True) 
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--batch_size', default=512)
    parser.add_argument('--gpu', default='cuda:0', help='GPU id to use.')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    main(args)