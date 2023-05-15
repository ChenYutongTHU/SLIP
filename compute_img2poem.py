import argparse, os, json
import torch, pickle, math
import utils, models, Datasets
from compute_sent_embedding import build_and_load_model, Sentence_dataset
from tqdm import tqdm
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
#TEMPLATE = '{}'
#TEMPLATES = [li.strip() for li in open('zh_templates.txt','r').readlines()]
TEMPLATES=['这张图片里有{}。']
def main(args):
    model = build_and_load_model(args.model, args.model_cfg_path, args.model_path)
    if args.model=='TRIPLET':
        tokenizer = utils.get_model(model).tokenize_zh
    model.cuda(args.gpu)
    print('Load sentence ...')
    sentences = json.load(open(args.sentence_path,'r'))
    id2poem = json.load(open(args.id2poem_path,'r'))
    print('Load sentence embeddings ...')
    sentence_embeddings = torch.load(args.sentence_embed_path,map_location='cpu')
    assert len(sentences)==len(sentence_embeddings), (len(sentences),len(sentence_embeddings))
    
    if args.rerank and os.path.isfile(args.keyword_path):
        print('Load keywords ...')
        sentence_keywords = json.load(open(args.keyword_path,'r'))
        assert len(sentences)==len(sentence_keywords), (len(sentences),len(sentence_keywords))    
    else:
        print('Skip keywords loading')
        sentence_keywords = None
    print(f'#sentences={len(sentence_embeddings)}')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    
    if args.model == 'MCLIP':
        val_transform = model.transform

    cnt = 0
    while True:
        if args.input_dir!='':
            img_dir = sorted(os.listdir(args.input_dir))
            if cnt>=len(img_dir):
                img_path = 'e'
            else:
                img_path = os.path.join(args.input_dir, img_dir[cnt])
        else: 
            img_path = input('Path/to/image_file or E(exit): \n')
        if img_path.lower() in 'e':
            return
        elif os.path.isfile(img_path):
            cnt += 1
            try:
                image = Image.open(img_path)
                image_arr = np.array(image)
            except:
                print('Open error ',img_path)
                continue
            if image_arr.shape[-1]!=3:
                if len(image_arr.shape)==2:
                    image_arr = image_arr[:,:,None]
                    image_arr = np.tile(image_arr, [1,1,3]) 
                elif image_arr.shape[-1]==4:
                    image_arr = image_arr[:,:,:3] 
            image = Image.fromarray(image_arr)     
            image = val_transform(image).unsqueeze(0)
            with torch.no_grad():
                image_features = utils.get_model(model).encode_image(image.cuda()) #B,
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_feature = image_features.squeeze(0)
            
            sims = []
            n_batch = math.ceil(len(sentence_embeddings)/args.batch_size)
            for i in tqdm(range(n_batch)):
                s,t = args.batch_size*i, min(args.batch_size*(i+1),len(sentence_embeddings)) 
                sent_features = torch.stack(sentence_embeddings[s:t], dim=0).cuda() #N,D
                # print(image_feature.shape, sent_features.shape)
                sim = image_feature@sent_features.t()
                sims.append(sim)
            sims = torch.cat(sims, dim=0)
            sorted_, indices = torch.sort(sims, descending=True)
            output_file = os.path.join(args.output_dir, img_path.replace('/','-')+'.txt')
            f = open(output_file,'w')
            f.writelines('==============Ranked by Chinese-CLIP==============\n')
            keywords = []
            for rank, i in enumerate(indices[:args.topk]):
                rank += 1
                #f.writelines(f'Top {rank}:\n')
                if type(sentences[i])==dict:
                    if 'id' in sentences[i]:
                        poem = id2poem[sentences[i]['id']]
                    elif 'poem_id' in sentences[i]:
                        poem = id2poem[sentences[i]['poem_id']]
                    else:
                        poem = None
                    if poem is not None:
                        author, dynasty, title = poem['author'], poem['dynasty'], poem['title']
                    f.writelines(f'{i}: {round(sims[i].item(),3)} {sentences[i]["content"]}  --《{title}》 {author}[{dynasty}]\n')
                    # f.writelines(f'{i}:{sentences[i]["content"]}  --《{title}》 {author}[{dynasty}]\n')
                else:
                    f.writelines(f'{i}:{sentences[i]}\n')
                if sentence_keywords is not None:
                    kws = sentence_keywords[i] #[[w1,w2,w3],[w1,w2,w3]]
                    keywords.extend(kws[0]+kws[1])
                # f.writelines(f'score:{round(sims[i].item(),3)}\n')
                # f.writelines('\n\n')
                #keywords_per_subsent = sentence_keywords[i] #[] usually consists of two subsentences

            if not args.model=='TRIPLET' or sentence_keywords is None:
                f.close()
                continue
            if args.rerank==False:
                f.close()
                continue

            f.writelines('==============Re-Ranking information==============\n')
            keywords = list(set(keywords))
            print(f'Compute keyword embedding ... #={len(keywords)}')
            keywords_prompt = []
            n_template = len(TEMPLATES)
            for TEMPLATE in TEMPLATES: #kw_id = id%n_template
                keywords_prompt.extend([TEMPLATE.replace('{}',kw) for kw in keywords])
            keywords_dataset = Sentence_dataset(keywords_prompt, tokenizer)
            keywords_loader = torch.utils.data.DataLoader(keywords_dataset, 
                                                          batch_size=args.batch_size_kw, 
                                                          shuffle=False)
            keywords_embedding = [0 for kw in keywords]
            cnt_ = 0
            for batch in tqdm(keywords_loader, total=len(keywords_loader)):
                if batch.dim()==3:
                    batch = batch.squeeze(1)
                with torch.no_grad():
                    kw_embeddings, _ ,_, _, _ = \
                        utils.get_model(model).encode_text(zh=batch.cuda(), en=None, output_attention=True)
                for kwe in kw_embeddings:
                    keywords_embedding[cnt_%len(keywords)] += kwe # D
                    cnt_ += 1
            #import ipdb; ipdb.set_trace()
            keywords_embedding = torch.stack(keywords_embedding, dim=0)/n_template #N.D
            keywords_embedding = torch.nn.functional.normalize(keywords_embedding, dim=1)

            
            keyword2embed = {kw:e for kw, e in zip(keywords, keywords_embedding)}
            keyword2sim = {}
            reranked_results = []
            for rank, i in enumerate(indices[:args.topk]):
                rank += 1
                if type(sentences[i])==dict:
                    poem = id2poem[sentences[i]['id']]
                    author, dynasty, title = poem['author'], poem['dynasty'], poem['title']
                    f.writelines(f'{i}: {round(sims[i].item(),3)} {sentences[i]["content"]}  --《{title}》 {author}[{dynasty}]\n')
                    # f.writelines(f'{i}:{sentences[i]["content"]}  --《{title}》 {author}[{dynasty}]\n')
                else:
                    f.writelines(f'{i}:{sentences[i]} \n')

                kws = sentence_keywords[i][0]+sentence_keywords[i][1]
                kw_star = 1
                for kw in kws:
                    assert kw in keyword2embed, kw
                    local_sim = image_feature@keyword2embed[kw]  #1,D D
                    keyword2sim[kw] = local_sim.cpu().item()
                    f.writelines('{}:{:.3f}\t'.format(kw,local_sim.cpu().item()))
                    kw_star = min(kw_star, keyword2sim[kw])
                f.writelines('\n')
                if kw_star>args.B0 and kw_star!=1:
                    c = args.wa*sims[i].item()+(1-args.wa)*kw_star
                    reranked_results.append({'id':i, 'b':kw_star, 'a':sims[i].item(), 'c':c})
            
            # note that split to two subsentences
            reranked_results = sorted(reranked_results, key=lambda r:r['c']*-1)
            f.writelines(f'==============Re-Ranking result (w_a={args.wa})==============\n')
            for rank, res in enumerate(reranked_results):
                rank += 1
                i = res['id']
                if type(sentences[i])==dict:
                    poem = id2poem[sentences[i]['id']]
                    author, dynasty, title = poem['author'], poem['dynasty'], poem['title']
                    f.writelines(f'rank{rank}-{i}: {sentences[i]["content"]}  --《{title}》 {author}[{dynasty}] \n')
                    f.writelines('a:{:.3f} b:{:.3f} c{:.3f} \n'.format(res['a'],res['b'],res['c']))
                    # f.writelines(f'{i}:{sentences[i]["content"]}  --《{title}》 {author}[{dynasty}]\n')
                else:
                    f.writelines(f'{i}:{sentences[i]} \n')


            f.close()


            print(f'See result in {output_file}')
            with open(output_file.replace('.txt','.keyword.pkl'),'wb') as f:
                pickle.dump(keyword2sim, f)
        else:
            print(f'{img_path} does not exist ...')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='compute sentence embeddings', add_help=False)
    parser.add_argument('--model_cfg_path', default='experiments/configs_duet/zh_large_tune_all.yaml')
    parser.add_argument('--model_path', default='') 
    parser.add_argument('--model', default='TRIPLET') 
    parser.add_argument('--rerank', action='store_true')
    parser.add_argument('--sentence_embed_path', required=True)
    parser.add_argument('--sentence_path', required=True)
    parser.add_argument('--keyword_path', default='keyword.pkl')
    parser.add_argument('--id2poem_path', required=True)
    parser.add_argument('--batch_size', default=512, type=int) 
    parser.add_argument('--batch_size_kw', default=256, type=int) 
    parser.add_argument('--input_dir', default='')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--topk', default=40, type=int)
    parser.add_argument('--gpu', default='cuda:0', help='GPU id to use.')
    parser.add_argument('--B0', default=0.135, type=float)
    parser.add_argument('--wa', default=0.5, type=float)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)