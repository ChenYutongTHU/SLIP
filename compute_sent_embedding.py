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

def segment2word(sentence):
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
    segments = segment2word(sentence)
    word_attn = []
    if method=='eos_last_attn_max':
        #attention = np.mean(attention[-1][:,-1,1:],axis=0) #average on head (L-1) exclude CLS 
        #already average
        attention = attention[-1][-1,1:]
        sentence = list(sentence)
        assert len(sentence) == attention.shape[0]-1, (len(sentence), sentence, attention.shape)
        for k, (si, sj, seg) in enumerate(segments):
            if sj>attention.shape[0]:
                return [], list(zip(sentence,attention[:-1]))
                # print(attention.shape)
                # print('sentence', sentence)
                # print(segments)
            attn_score = np.max([attention[i] for i in range(si, sj)]) #now max
            segments[k].append(attn_score)
        sorted_segments = sorted(segments, key=lambda x:x[-1]*(-1)) 
        # print(sorted_segments)
        # input()
        if len(sorted_segments)==0:
            return [], list(zip(sentence,attention[:-1]))
        if sorted_segments[0][-1]<thrsh:
            return [[sorted_segments[0][2], sorted_segments[0][-1]]], list(zip(sentence,attention[:-1]))
        else:
            return [[ss[2],ss[3]] for ss in sorted_segments if ss[-1]>=thrsh], list(zip(sentence,attention[:-1]))

def filter_keyword(segs, cluster_thresh=0.25, abs_thresh=6.8, rel_thresh=26.6):
    infos_word = []
    range_ = segs[0][-1]-segs[-1][-1]
    thresh = range_*cluster_thresh
    c = []
    i = 0
    while i<len(segs):
        j = i+1
        while j<len(segs) and segs[i][-1]-segs[j][-1]<=thresh:
            j+=1
        c.append([segs[k] for k in range(i,j)])
        i = j
    c_score = np.array([np.mean([w[-1] for w in cc]) for cc in c]) #[cluster_mean1, cluster_mean2, cluster_mean3]
    c_norm_score = c_score/np.sum(c_score) #cluster norm #relative-score
    sum_ = 0
    for cc in c:
        sum_ += sum([w[-1] for w in cc]) #sum over all scores
    norm_c_score = np.array([np.mean([w[-1]/sum_ for w in cc]) for cc in c]) #norm cluster #normalization before computing the center
    for cc, cs, cns, ncs in zip(c, c_score, c_norm_score, norm_c_score):
        for w,_ in cc: #for each cluster
            infos_word.append(
                {'abs':cs*100, 
                 'rel_norm_c':ncs*100, 
                 'word':w,
                 'is_keyword':(cs*100>abs_thresh or ncs*100>rel_thresh)})
    return infos_word

def build_and_load_model(model_cfg_path, model_path):
    print('Building model')
    model = getattr(models, 'TRIPLET')(
        ssl_mlp_dim=4096, ssl_emb_dim=256,
        model_cfg_path=model_cfg_path, toolkit='torch')
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
    model = build_and_load_model(args.model_cfg_path, args.model_path)
    model.cuda(args.gpu)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200)
    tokenizer = {'en':utils.get_model(model).tokenize_en, 'zh':utils.get_model(model).tokenize_zh}

    print('Building sentence loader (only support single GPU)')
    sentences = json.load(open(args.sentence_path, 'r')) #list
    assert type(sentences)==list
    if type(sentences[0])==dict:
        sentences = [s['content'] for s in sentences]
    else:
        pass
    # DEBUG=True
    # if DEBUG:
    #     sentences = sentences[:4]
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

    if os.path.isfile(args.output_path):
        print(args.output_path+' already exists.')
    else:
        print('Computing embeddings ...')
        result_embeddings = []
        for zh in tqdm(dataloader, total=len(dataloader)):
            if zh.dim()==3:
                zh = zh.squeeze(1)
            with torch.no_grad():
                zh_embeddings, _, _, layer_attn_weights, _ = \
                    utils.get_model(model).encode_text(zh=zh.cuda(), en=None, output_attention=True) #already normalize #B,N 
            result_embeddings.extend([e.cpu() for e in zh_embeddings])  

        print('Saving embedding as {}'.format(args.output_path))
        torch.save(result_embeddings, args.output_path)
    
    print('Compute keywords ... ')
    id2keywords, id2attention = [[] for _ in range(len(dataset))], [[] for _ in range(len(dataset))]
    cnt = 0
    for zz, zh in enumerate(tqdm(dataloader_sub, total=len(dataloader_sub))):
        if zh.dim()==3:
            zh = zh.squeeze(1)
        with torch.no_grad():
            zh_embeddings, _, _, layer_attn_weights, _ = \
                utils.get_model(model).encode_text(zh=zh.cuda(), en=None, output_attention=True) #already normalize #B,N 
        for bi, input_ids_ in enumerate(zh): #input_ids 1,L
            eot_loc = torch.sum(input_ids_!=0) #14
            decoded = tokenizer['zh'].ids2tokens(input_ids_[1:eot_loc-1].cpu().numpy())
            attention = [w[bi,:eot_loc,:eot_loc].cpu().numpy()  for w in layer_attn_weights] #already average
            try:
                keywords, word_attn = extract_keyword(
                    sentence=decoded, #remove cls and eos
                    attention=attention,
                    thrsh=0)
            except:
                print('Error', decoded)
            if len(keywords)==0:
                print('empty keywords', decoded)
            else:
                keywords = filter_keyword(segs=keywords)
                id2keywords[sub2id[cnt]].extend(keywords) 
                id2attention[sub2id[cnt]].append(word_attn)
            cnt += 1
            # if decoded==[]:
            #     print(input_ids_, sentences[len(extracted_keywords)])
            #     import ipdb; ipdb.set_trace()
            # print(decoded)
            # print(keywords)
            # input() 
    #import ipdb; ipdb.set_trace()

    output_path_keyword = '.'.join(args.output_path.split('.')[:-1])+'.keyword.pkl'
    print('Saving keywords as {}'.format(output_path_keyword))
    with open(output_path_keyword,'wb') as f:
        pickle.dump(id2keywords, f)

    output_path_attention= '.'.join(args.output_path.split('.')[:-1])+'.attention.pkl'
    print('Saving attention as {}'.format(output_path_attention))
    with open(output_path_attention,'wb') as f:
        pickle.dump(id2attention, f)   
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