import argparse, os, json
import torch, pickle
import numpy as np
import utils, models, datasets
from tqdm import tqdm
from PoemSegmentor.segmentor import Poem_Segmenter
segmentor = Poem_Segmenter()

def segment_sentence(sentence):
    #xxxxx,xxxxx. -> [xxxxx,xxxxx]
    puncs = [',','.','!','?',';','，','。','！','？']
    sents = []
    puncs_id = [i for i,s in enumerate(sentence) if s in puncs]
    j = 0
    for i in puncs_id:
        sents.append((j,i,sentence[j:i]))
        j = i+1
    if puncs_id!=[] and i!=len(sentence)-1:
        sents.append((i+1, len(sentence), sentence[i+1:]))
    if puncs_id == []:
        sents.append((0, len(sentence), sentence))
    return sents    

def segment(sentence):
    segments = []
    if not len(sentence) in [5,7]: #can only handle 7yan or 5yan
        segments.extend([[k,k+1,w] for k,w in enumerate(sentence)])
        return segments
    input_ = ''.join(sentence)
    input_ = input_.replace('[UNK]','U')
    outputs = segmentor.cut(input_)
    i = 0
    for seg in outputs.split(' | '):
        segments.append([i,i+len(seg),seg])
        i += len(seg)
    return segments  

def extract_keyword(sentence, attention, method='eos_last_attn_max',thrsh=0.035):
    segments = segment(sentence)
    if method=='eos_last_attn_max':
        #attention = np.mean(attention[-1][:,-1,1:],axis=0) #average on head (L-1) exclude CLS 
        #already average
        attention = attention[-1][-1,1:]
        sentence = list(sentence)
        assert len(sentence) == attention.shape[0]-1, (len(sentence), sentence, attention.shape)
        for k, (si, sj, seg) in enumerate(segments):
            if sj>attention.shape[0]:
                return []
                # print(attention.shape)
                # print('sentence', sentence)
                # print(segments)
            attn_score = np.max([attention[i] for i in range(si, sj)]) #now max
            segments[k].append(attn_score)
        sorted_segments = sorted(segments, key=lambda x:x[-1]*(-1)) 
        # print(sorted_segments)
        # input()
        if len(sorted_segments)==0:
            return []
        if sorted_segments[0][-1]<thrsh:
            return [[sorted_segments[0][2], sorted_segments[0][-1]]]
        else:
            return [[ss[2],ss[3]] for ss in sorted_segments if ss[-1]>=thrsh]
        
def build_and_load_model(model_cfg_path, model_path):
    print('Building model')
    model = getattr(models, 'TRIPLET')(
        ssl_mlp_dim=4096, ssl_emb_dim=256,
        model_cfg_path=model_cfg_path, toolkit='torch')
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
    model = build_and_load_model(args.model_cfg_path, args.model_path)
    model.cuda(args.gpu)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200)
    tokenizer = {'en':utils.get_model(model).tokenize_en, 'zh':utils.get_model(model).tokenize_zh}

    print('Building sentence loader (only support single GPU)')
    sentences = json.load(open(args.sentence_path, 'r')) #list
    DEBUG=True
    if DEBUG:
        sentences = sentences[:4]
    #further split
    sentences_sub = []
    sub2id = []
    for si,sent in enumerate(sentences):
        sent_sub = segment_sentence(sent)
        sentences_sub.extend([s for _,_,s in sent_sub])
        sub2id.extend([si]*len(sent_sub))

    print(f'#sentences={len(sentences)}')
    dataset = Sentence_dataset(sentences, tokenizer['zh'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    dataset_sub = Sentence_dataset(sentences_sub, tokenizer['zh'])
    dataloader_sub = torch.utils.data.DataLoader(dataset_sub, batch_size=args.batch_size, shuffle=False)

    print('Computing embeddings ...')
    result_embeddings = []
    extracted_keywords = []
    for zh in tqdm(dataloader, total=len(dataloader)):
        if zh.dim()==3:
            zh = zh.squeeze(1)
        with torch.no_grad():
            zh_embeddings, _, _, layer_attn_weights, _ = \
                utils.get_model(model).encode_text(zh=zh.cuda(), en=None, output_attention=True) #already normalize #B,N 
        result_embeddings.extend([e.cpu() for e in zh_embeddings])  

    print('Compute keywords ... ')
    id2keywords = [[] for _ in result_embeddings]
    cnt = 0
    for zh in tqdm(dataloader_sub, total=len(dataloader_sub)):
        if zh.dim()==3:
            zh = zh.squeeze(1)
        with torch.no_grad():
            zh_embeddings, _, _, layer_attn_weights, _ = \
                utils.get_model(model).encode_text(zh=zh.cuda(), en=None, output_attention=True) #already normalize #B,N 
        for bi, input_ids_ in enumerate(zh): #input_ids 1,L
            eot_loc = torch.sum(input_ids_!=0) #14
            decoded = tokenizer['zh'].ids2tokens(input_ids_[1:eot_loc-1].cpu().numpy())
            attention = [w[bi,:eot_loc,:eot_loc].cpu().numpy()  for w in layer_attn_weights] #already average
            keywords = extract_keyword(
                sentence=decoded, #remove cls and eos
                attention=attention,
                thrsh=0)
            id2keywords[sub2id[cnt]].append(keywords) 
            cnt += 1
            # if decoded==[]:
            #     print(input_ids_, sentences[len(extracted_keywords)])
            #     import ipdb; ipdb.set_trace()
            # print(decoded)
            # print(keywords)
            # input()
    import ipdb; ipdb.set_trace()
    print('Saving embedding as {}'.format(args.output_path))
    torch.save(result_embeddings, args.output_path)
    output_path_keyword = ''.join(args.output_path.split('.')[:-1])+'.keyword.pkl'
    print('Saving keywords as {}'.format(output_path_keyword))
    with open(output_path_keyword,'wb') as f:
        pickle.dump(id2keywords, f)
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='compute sentence embeddings', add_help=False)
    parser.add_argument('--model_cfg_path', default='experiments/configs_duet/zh_large_tune_all.yaml')
    parser.add_argument('--model_path', default='experiments/outputs_duet/cc3m_gpu8_bs16_duet_tune_all_lr1e-6_wd0.001_warmup50_steps250/checkpoint_best.pt') 
    parser.add_argument('--sentence_path', required=True) 
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--batch_size', default=512)
    parser.add_argument('--gpu', default='cuda:0', help='GPU id to use.')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    main(args)