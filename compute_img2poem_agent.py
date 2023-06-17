import argparse, os, json
import torch, pickle, math
import utils, models, Datasets
from compute_sent_embedding import build_and_load_model, Sentence_dataset
from tqdm import tqdm
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import yaml
import logging
from pathlib import Path
def gaussian_weight(n, std):
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))    
    weight_map = np.exp(((xs-n/2.)*(xs-n/2.)+(ys-n/2.)*(ys-n/2.))/(-2*std*std))
    return weight_map
n_cand = 40
T, T_hat = 0.18, 0.6
s_img_threshold, s_region_threshold = 0.19, 10 
inv_temp = 0.3
std = 5/3
w_0, w_p, w_n = 100, 3, 0.5
weight_loc = torch.tensor(gaussian_weight(16, std=std))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

def setup_logger(log_level='DEBUG', output_file='log.txt'):
    """Setup a logger instance that writes to the specified log level."""
    root_logger = logging.getLogger()
    # Set up console handler with a formatter and a limited log level
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(message)s")
    console_handler.setLevel(logging.NOTSET)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    file_handler = logging.FileHandler(str(output_file))
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter("%(asctime)s [%(processName)-2d] %(levelname)s %(message)s")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    return root_logger
    # logger = setup_logger('INFO')
    # logger.info("Logging at INFO level.")
    # logger.debug("Debug message.")
    # logger.warning("Warning message.")

class CONFIG:
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.data = yaml.safe_load(f)
        for k,v in self.data.items():
            setattr(self, k, v)


class Agent():
    def __init__(self, cfg_path):
        self.config = CONFIG(cfg_path)
        self.mask_clip = getattr(self.config,'mask_clip', False)
        self.rerank, self.gpu = self.config.rerank, self.config.gpu

        self.logger = setup_logger(
            'INFO', 
            output_file=os.path.join(self.config.output_dir,'log.txt'))
        self.logger.info("Load config from "+cfg_path)

        self.init_model()
        self.init_sentence()
        
        if self.rerank:
            self.init_keyword()
        
        return
    
    def return_results(self, indices, scores):
        results = {'sens':[], 'source':{}, 'sourcecontent':[]}
        for rank, i in enumerate(indices[:self.config.return_topk]):
            if 'id' in self.sentences[i]:
                poem = self.id2poem[self.sentences[i]['id']]
            elif 'poem_id' in self.sentences[i]:
                poem = self.id2poem[self.sentences[i]['poem_id']]
            results['sens'].append(self.sentences[i]['content'])
            results['source'][str(rank)] = {'title': poem['title'], 'author': poem['author'], 'dynasty': poem['dynasty']}
            results['sourcecontent'].append(poem['content_list'])
            self.logger.info(f'Retrieve top{rank}:  {results["sens"][-1]} ' + ' '.join(['{}:{:.3f}'.format(k, v) for k,v in scores[rank].items()]))
        return results

    def compute_keywords_embedding(self, keywords, TEMPLATES):
        keywords_prompt = []
        n_template = len(TEMPLATES)
        for TEMPLATE in TEMPLATES: #kw_id = id%n_template
            keywords_prompt.extend([TEMPLATE.replace('{}',kw) for kw in keywords])
        keywords_dataset = Sentence_dataset(keywords_prompt, self.tokenizer)
        keywords_loader = torch.utils.data.DataLoader(keywords_dataset, 
                                                    batch_size=self.config.batch_size_kw, 
                                                    shuffle=False)
        keywords_embedding = [0 for kw in keywords]
        cnt_ = 0
        for batch in tqdm(keywords_loader, total=len(keywords_loader)):
            if batch.dim()==3:
                batch = batch.squeeze(1)
            with torch.no_grad():
                kw_embeddings, _ ,_, _, _ = \
                    utils.get_model(self.model).encode_text(zh=batch.to(self.config.gpu), en=None, output_attention=True)
            for kwe in kw_embeddings:
                keywords_embedding[cnt_%len(keywords)] += kwe # D
                cnt_ += 1
        keywords_embedding = torch.stack(keywords_embedding, dim=0)/n_template #N.D
        keywords_embedding = torch.nn.functional.normalize(keywords_embedding, dim=1) 
        keyword2embed = {kw:e for kw, e in zip(keywords, keywords_embedding)}
        return keyword2embed   
        
    def inference(self, image, image_name):
        #image an np.array or PIL.Image
        extra_info = {}
        self.logger.info(f'Receive image -- {image_name}')
        image_arr = np.array(image)
        if image_arr.shape[-1]!=3:
            if len(image_arr.shape)==2:
                image_arr = image_arr[:,:,None]
                image_arr = np.tile(image_arr, [1,1,3]) 
            elif image_arr.shape[-1]==4:
                image_arr = image_arr[:,:,:3] 
        image = Image.fromarray(image_arr)     
        image = val_transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs_ = utils.get_model(self.model).encode_image(image.to(self.gpu)) #B,
        if self.mask_clip:
            outputs, attention_weights = outputs_
            image_features = outputs[:,0,:]
            dense_features = outputs[:,:,:]
            dense_features = dense_features / dense_features.norm(dim=-1, keepdim=True)
            extra_info['dense_features'] = dense_features
            extra_info['attention_weights'] = attention_weights
        else:
            image_features = outputs_
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_feature = image_features.squeeze(0)
        
        sims = []
        n_batch = math.ceil(len(self.sentence_embeddings)/self.config.batch_size)
        for i in tqdm(range(n_batch)):
            s,t = self.config.batch_size*i, min(self.config.batch_size*(i+1),len(self.sentence_embeddings)) 
            sent_features = torch.stack(self.sentence_embeddings[s:t], dim=0).to(self.gpu) #N,D
            sim = image_feature@sent_features.t()
            sims.append(sim)
        sims = torch.cat(sims, dim=0)
        sorted_, indices = torch.sort(sims, descending=True)

        #record
        self.indices = indices
        self.global_sims = sims

        if self.rerank==False:
            return self.return_results(
                indices,
                scores = [{'sim': sims[i].item()} for i in indices]), extra_info
        else:
            keywords = []
            for i in indices[:self.config.topk]:
                kws = self.sentence_keywords[i]
                if len(kws)==2 and type(kws[0])==list:
                    keywords.extend(kws[0]+kws[1])
                else:
                    keywords.extend(kws)
            keywords = list(set(keywords))

            self.logger.info(f'Compute keywords embeddings (#={len(keywords)})')
            if self.mask_clip:
                keyword2embed_word = self.compute_keywords_embedding(keywords, TEMPLATES=['{}'])
                keyword2dense_sim = {}
                for kw, embed in keyword2embed_word.items():
                    dense_sim = dense_features@embed #1, 256, 1024
                    #grid_size = int(math.sqrt(dense_sim.shape[1]))
                    #dense_sim = dense_sim.squeeze(0).reshape([grid_size, grid_size,-1])
                    keyword2dense_sim[kw] = dense_sim.cpu()
                extra_info['keyword2dense_sim'] = keyword2dense_sim
            else:
                keyword2embed = self.compute_keywords_embedding(keywords, TEMPLATES=['这张图片里有{}。'])
                keyword2sim = {}
            reranked_results = []
            for rank, i in enumerate(indices[:self.config.topk]):
                rank += 1
                if len(self.sentence_keywords[i])==2 and type(self.sentence_keywords[i][0]) == list:
                    kws = self.sentence_keywords[i][0]+self.sentence_keywords[i][1]
                else:
                    kws = self.sentence_keywords[i]
                # rerank - v1
                keywords_p, keywords_n = [], []
                s_n, A = 0, torch.zeros(16,16)
                for kw in kws:
                    cam = extra_info['keyword2dense_sim'][kw][0] #257
                    sim_img, sim_patch = cam[0].item(), cam[1:].view(16,16)
                    sim_patch = sim_patch*(sim_patch>0.18)
                    sim_region = torch.sum(sim_patch).item()
                    #condition for positive/negative keywords:
                    if (sim_img>s_img_threshold) or (sim_region>s_region_threshold):
                        keywords_p.append(kw)
                        sim_patch_hat = (sim_patch-sim_patch.min())/(sim_patch.max()-sim_patch.min())
                        A += (sim_patch_hat>T_hat)
                    else:
                        keywords_n.append(kw)
                        s_n -= math.exp(inv_temp*(max(s_region_threshold-sim_region,0)))
                A = (A.bool()).int()
                s_p = torch.sum(weight_loc*A).item()
                s_0 = self.global_sims[i].item()
                s_overall = w_0*s_0 + w_p*s_p + w_n*s_n
                reranked_results.append({'id':i, 's_0':s_0, 
                                        's_p':s_p, 's_n':s_n, 's_overall':s_overall})
                ''' rerank - v0
                kw_star = 1
                for kw in kws:
                    assert kw in keyword2embed, kw
                    local_sim = image_feature@keyword2embed[kw]  #1,D D
                    keyword2sim[kw] = local_sim.cpu().item()
                    kw_star = min(kw_star, keyword2sim[kw])
                if kw_star>self.config.B0 and kw_star!=1:
                    c = self.config.wa*sims[i].item()+(1-self.config.wa)*kw_star
                    reranked_results.append({'id':i, 'b':kw_star, 'a':sims[i].item(), 'c':c})
                '''
            reranked_results = sorted(reranked_results, key=lambda r:r['s_overall']*-1)
            # reranked_results = sorted(reranked_results, key=lambda r:r['c']*-1) rerank-v0

            return self.return_results(
                [r['id'] for r in reranked_results],
                scores = [{k:v for k,v in r.items() if not k=='id'} for r in reranked_results]), \
                extra_info

    def init_model(self):
        self.logger.info("Build and load TRIPLET model from "+self.config.model_path)
        self.model = build_and_load_model('TRIPLET', 
                                          self.config.model_cfg_path, 
                                          self.config.model_path,
                                          mask_clip=self.mask_clip)
        self.tokenizer = utils.get_model(self.model).tokenize_zh

        self.logger.info("Running on GPU "+self.config.gpu)
        self.model = self.model.to(self.config.gpu)
        return

    def init_sentence(self):
        self.logger.info("Loading sentence from "+self.config.sentence_path)
        self.sentences = json.load(open(self.config.sentence_path,'r'))
        self.logger.info("Loading id2poem from "+self.config.id2poem_path)
        self.id2poem = json.load(open(self.config.id2poem_path,'r'))

        self.logger.info("Loading sentence embeddings from "+self.config.sentence_embed_path)
        self.sentence_embeddings = torch.load(self.config.sentence_embed_path,map_location='cpu')
        assert len(self.sentences)==len(self.sentence_embeddings), (len(self.sentences),len(self.sentence_embeddings))
        self.logger.info(f'Sentence #={len(self.sentences)}')
        return

    def init_keyword(self):
        self.sentence_keywords = json.load(open(self.config.keyword_path,'r'))
        assert len(self.sentences)==len(self.sentence_keywords), (len(self.sentences),len(self.sentence_keywords))            