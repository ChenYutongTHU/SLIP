import argparse, os, json
import torch, pickle
import utils, models, datasets
from compute_sent_embedding import build_and_load_model
from tqdm import tqdm
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def main(args):
    model = build_and_load_model(args.model_cfg_path, args.model_path)
    model.cuda(args.gpu)
    print('Load sentence ...')
    sentences = json.load(open(args.sentence_path,'r'))
    print('Load sentence embeddings ...')
    sentence_embeddings = torch.load(args.sentence_embed_path,map_location='cpu')
    assert len(sentences)==len(sentence_embeddings), (len(sentences),len(sentence_embeddings))
    print('Load keywords ...')
    sentence_keywords = pickle.load(open(args.sentence_path.split('.')[0]+'.keyword.json','rb'))
    assert len(sentences)==len(sentence_keywords), (len(sentences),len(sentence_keywords))    
    print(f'#sentences={len(sentence_embeddings)}')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    
    while True:
        img_path = input('Path/to/image_file or E(exit): \n')
        if img_path.lower() in 'e':
            return
        elif os.path.isfile(img_path):
            image = Image.open(img_path)
            image_arr = np.array(image)
            if image_arr.shape[-1]!=3:
                if len(image_arr.shape)==2:
                    image_arr = image_arr[:,:,None]
                image_arr = np.tile(image_arr, [1,1,3])  
            image = Image.fromarray(image_arr)     
            image = val_transform(image).unsqueeze(0)
            with torch.no_grad():
                image_features = utils.get_model(model).encode_image(image.cuda()) #B,
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_feature = image_features.squeeze(0)
            
            sims = []
            for i in tqdm(range(args.batch_size)):
                s,t = args.batch_size*i, min(args.batch_size*(i+1),len(sentence_embeddings)) 
                sent_features = torch.stack(sentence_embeddings[s:t], dim=0).cuda() #N,D
                # print(image_feature.shape, sent_features.shape)
                sim = image_feature@sent_features.t()
                sims.append(sim)
            sims = torch.cat(sims, dim=0)
            sorted, indices = torch.sort(sims, descending=True)
            output_file = os.path.join(args.output_dir, img_path.replace('/','-')+'.txt')
            f = open(output_file,'w')
            f.writelines('==============Ranked by Chinese-CLIP==============\n')
            for rank, i in enumerate(indices[:args.topk]):
                rank += 1
                f.writelines(f'Top {rank}:\n')
                f.writelines(f'{i}:{sentences[i]}\n')
                f.writelines(f'score:{round(sims[i].item(),3)}\n')
                f.writelines('\n\n')
                keywords = sentence_keywords[i] #[]

            f.writelines('==============Re-Ranking==============\n')
            # note that split to two subsentences

            f.close()


            print(f'See result in {output_file}')
        else:
            print(f'{img_path} does not exist ...')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='compute sentence embeddings', add_help=False)
    parser.add_argument('--model_cfg_path', default='experiments/configs_duet/zh_large_tune_all.yaml')
    parser.add_argument('--model_path', default='experiments/outputs_duet/cc3m_gpu8_bs16_duet_tune_all_lr1e-6_wd0.001_warmup50_steps250/checkpoint_best.pt') 
    parser.add_argument('--sentence_embed_path', required=True)
    parser.add_argument('--sentence_path', required=True)
    parser.add_argument('--batch_size', default=512, type=int) 
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--topk', default=50, type=int)
    parser.add_argument('--gpu', default='cuda:0', help='GPU id to use.')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    main(args)