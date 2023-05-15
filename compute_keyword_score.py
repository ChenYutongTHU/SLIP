
import argparse, os, pickle, json
from compute_sent_embedding_deprecated import segment_sentence
from transformers import BertModel, BertTokenizer
import torch
from tqdm import tqdm 
import numpy as np
import math

def filter_algorithm(word_score, C=49.5, R=0.70, S0=15.9, S1=12):
    word_score = sorted(word_score, key=lambda ws:ws[1]*-1)
    words = [w for w,s in word_score]
    scores = np.array([s for w,s in word_score])
    scores = scores/np.sum(scores)
    scores = scores*100
    cumsum = np.cumsum(scores)
    #condition 1: cumsum > 49.5%
    i0 = np.sum(cumsum<C)
    #condition 2: include ones >16%
    i1 = i0+1
    while (i1)<len(words) and scores[i1]>S0:
        i1 += 1
    #condition 3:
    while i1<len(words) and scores[i1]>S1 and scores[i1]/scores[i0]>R:
        i1 += 1
    return words, scores, cumsum, i1
    

def main(args):
    sentences = json.load(open(args.sentence_path, 'r')) #list
    print(f'Load sentences from {args.sentence_path} #={len(sentences)}')
    sub_sentences = []
    for ss in sentences:
        seg = segment_sentence(ss['content'])
        assert len(seg)==2, ss
        sub_sentences.extend([s for _,_,s in seg])

    #pos 
    if os.path.isfile(args.pos_path):
        pos = json.load(open(args.pos_path,'r'))
        print('Load pos from', args.pos_path) 
    else:
        from deepthulac import LacModel
        pos = [[] for s in sentences]
        model_path_pos = '/data03/private/chengzhili/segmentation/output/pos/2023-02-10_17-11-24 pos'
        lac_pos = LacModel.load(path=model_path_pos, device='cpu', use_f16=False)
        n_batch = math.ceil(len(sub_sentences)/args.batch_size)
        bert_attention = [[] for ss in sentences]
        for n in tqdm(range(n_batch)):
            s, e = n*args.batch_size, min((n+1)*args.batch_size, len(sub_sentences))
            batch_sents = sub_sentences[s:e]
            results_pos = lac_pos.seg(batch_sents)['pos']['res']
            for i in range(s,e):
                pos[i//2].append(results_pos[i-s])

        with open(args.pos_path,'w') as f:
            json.dump(pos, f)
        print('Save pos as ', args.pos_path)
        
    assert os.path.isfile(args.clip_attention_path)
    clip_attention = pickle.load(open(args.clip_attention_path,'rb')) #[n_poem][0/1]
    print('Load clip_attention from ',args.clip_attention_path)

    #compute bert attention
    if os.path.isfile(args.bert_attention_path):
        bert_attention = pickle.load(open(args.bert_attention_path,'rb'))
        print('Load bert attention from', args.bert_attention_path)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_path) 
        model = BertModel.from_pretrained(args.bert_model_path)
        print('Load bert model from', args.bert_model_path)
        model.to(args.gpu)

        n_batch = math.ceil(len(sub_sentences)/args.batch_size)
        bert_attention = [[] for ss in sentences]
        for n in tqdm(range(n_batch)):
            s, e = n*args.batch_size, min((n+1)*args.batch_size, len(sub_sentences))

            batch_sents = sub_sentences[s:e]

            inputs = tokenizer(batch_sents, padding='longest')
            inputs = {k:torch.tensor(ip, device=args.gpu) for k,ip in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs,output_attentions=True)
            for ni in range(e-s):
                valid_len = torch.sum(inputs['attention_mask'][ni]).item()
                last_layer_attention = torch.mean(outputs.attentions[-1][ni,:,:valid_len,:valid_len],dim=0) #H,L,L -> (L,L)
                sim_show = np.mean(last_layer_attention.cpu().numpy(),axis=0)[1:-1] #exclude cls and eos
                sim_show = sim_show/sim_show.sum()
                sim_show = sim_show[None,:]
                bert_attention[(s+ni)//2].append(sim_show)
        with open(args.bert_attention_path,'wb') as f:
            pickle.dump(bert_attention, f)
        print('Save bert attention as ', args.bert_attention_path)

    #ensemble
    if os.path.isfile(args.ensemble_attention_path):
        ensemble_attention = pickle.load(open(args.ensemble_attention_path,'rb'))
        print('Load ensemble attention from', args.ensemble_attention_path)
    else:
        assert len(clip_attention)==len(bert_attention), (len(clip_attention), len(bert_attention))
        ensemble_attention = [[] for s in sentences]
        for i, (cas, bas) in enumerate(zip(clip_attention, bert_attention)):
            for ca,ba in zip(cas,bas):
                ca = np.array([s for w,s in ca])[None,:]
                ca = ca/np.sum(ca) #!!!
                ensemble_attention[i].append((ca+ba)/2)
        print(ensemble_attention[62155])
        with open(args.ensemble_attention_path, 'wb') as f:
            pickle.dump(ensemble_attention, f)
        print('Save ensemble attention as ', args.ensemble_attention_path)

    #filter
    keywords = [[] for s in sentences]
    for i in tqdm(range(len(sentences))):
        for ii in range(2):
            sent, score, pos_ = sub_sentences[i*2+ii], ensemble_attention[i][ii], pos[i][ii]
            # import ipdb; ipdb.set_trace()
            sorted_words, _, cumsum, K  = \
                filter_algorithm([(c,s) for c, s in zip(sent,score[0])]) #score shape!
            c_ids = np.argsort(score[0]*-1)[:K] #location
            cid2pos = []
            for seg in pos_:
                w,p = seg.split('_')
                for ww in w:
                    cid2pos.append([w,p])
            kw = []
            for cid in c_ids:
                w,p = cid2pos[cid]
                if p in ['n','np','ni','ns','nz']:
                    kw.append(w)
            keywords[i].append(list(set(kw)))
        #print(i, sentences[i], list(set(kw)))
    with open(args.keyword_path, 'w') as f:
        json.dump(keywords, f)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='compute sentence embeddings', add_help=False)
    parser.add_argument('--bert_model_path', required=True)
    parser.add_argument('--sentence_path', required=True) 
    parser.add_argument('--pos_path', required=True) 
    parser.add_argument('--clip_attention_path', required=True)
    parser.add_argument('--bert_attention_path', required=True)
    parser.add_argument('--ensemble_attention_path', required=True)
    parser.add_argument('--keyword_path', required=True)
    parser.add_argument('--batch_size', default=512)
    parser.add_argument('--gpu', default='cuda:0', help='GPU id to use.')
    args = parser.parse_args()
    for p in [args.bert_attention_path, 
                args.ensemble_attention_path, args.keyword_path]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    main(args)