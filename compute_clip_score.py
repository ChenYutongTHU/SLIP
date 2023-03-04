import argparse, os, json, pickle
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import utils, models, datasets
from utils import get_rank, get_world_size, is_main_process, dist_synchronize, dist_all_reduce
from datasets import TripletDataset
import numpy as np
def reorder(sim, num_replicas, num):
    sim_reordered = {}
    for la in sim:
        assert sim[la].shape[0]%num_replicas==0, (sim[la].shape)
        total_size = sim[la].shape[0]
        chunk_size = total_size//num_replicas 
        ids = []
        for rank in range(num_replicas):
            ids.append(np.arange(sim[la].shape[0])[rank:total_size:num_replicas])
        ids = np.concatenate(ids, axis=0)
        sim_reordered[la] = np.zeros_like(sim[la])
        for i0,i1 in enumerate(ids):
            sim_reordered[la][i1] = sim[la][i0]
        sim_reordered[la] = sim_reordered[la][:num]
    return sim_reordered
    
def get_args_parser():
    parser = argparse.ArgumentParser(description='Compute CLIP score', add_help=False)
    parser.add_argument('--model_cfg_path', default='experiments/configs_distill_contrastive/Vit_L_14_lit.yaml')
    parser.add_argument('--datasets', default='coco-cn-test,aic-val,cc3m')
    parser.add_argument('--output_dir', default='dataset_analysis/')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--load_ckpt', default='')
    parser.add_argument('--toolkit', default='torch')
    parser.add_argument('--model', default='TRIPLET')
    # Model
    parser.add_argument('--ssl-mlp-dim', default=4096, type=int,
                        help='hidden dim of SimCLR mlp projection head')
    parser.add_argument('--ssl-emb-dim', default=256, type=int,
                        help='output embed dim of SimCLR mlp projection head')
    parser.add_argument('--ssl-scale', default=1.0, type=float,
                        help='loss scale for SimCLR objective')
    parser.add_argument('--ssl-temp', default=0.1, type=float,
                        help='softmax temperature for SimCLR objective')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    return parser

def get_dataloader(args, val_transform, tokenizer):
    name2loader = {}
    catalog = json.load(open('dataset_catalog.json', 'r')) 
    for datasetname in args.datasets:
        if datasetname in ['coco-cn-test','aic-val']:
            with open(catalog[datasetname]['metadata'],'r') as f:
                img2captions = json.load(f)
            img_files, zh_caps, en_caps = [], [], []
            for img_id, c in img2captions.items():
                assert len(c['zh'])==len(c['en'])
                zh_caps.extend(c['zh'])
                en_caps.extend(c['en'])
                img_files.extend([catalog[datasetname]['img_filename'].format(img_id)]*len(c['zh']))
            dataset = datasets.TripletDataset_from_rawfile(
                img_file=img_files, zh=zh_caps, en=en_caps, 
                preprocess=val_transform, tokenizer=tokenizer,
                return_dict=True)
        elif datasetname in ['cc3m','cc12m','coco']:
            dataset = TripletDataset(names=datasetname, catalog=catalog,
                        preprocess=val_transform, tokenizer=tokenizer,
                        need_img=True, txt_from_tsv=True)
        else:
            raise ValueError
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        dataloader = torch.utils.data.DataLoader(dataset, 
            batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True, drop_last=False) 
        name2loader[datasetname] = dataloader
        print(f'{datasetname} #={len(dataset)} #batch={len(dataloader)}')
    return name2loader

def main(args):
    utils.init_distributed_mode(args)
    print("=> creating model: {}".format(args.model))
    model = getattr(models, args.model)(
        ssl_mlp_dim=args.ssl_mlp_dim, ssl_emb_dim=args.ssl_emb_dim,
        model_cfg_path=args.model_cfg_path, toolkit=args.toolkit)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200)
    if args.load_ckpt!='':
        print("=> loading checkpoint '{}'".format(args.load_ckpt))
        checkpoint = torch.load(args.load_ckpt, map_location='cpu')
        result = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(result)
    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    tokenizer = {'en':utils.get_model(model).tokenize_en, 'zh':utils.get_model(model).tokenize_zh}
    name2loader = get_dataloader(args, val_transform, tokenizer)
    for name, loader in name2loader.items():
        # ind2sim_en, ind2sim_zh = {}, {}
        print(name,f'#dataset={len(loader.dataset)}, #loader={len(loader)}' )
        cosine_sim = {'zh':[], 'en':[]}
        for batch in tqdm(loader, disable=(not is_main_process())):
            img, zh, en = batch['img'], batch['zh'], batch['en']
            indices = batch['index']
            with torch.no_grad():
                image_features = utils.get_model(model).encode_image(img.cuda()) #B,N
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_embeddings = {}
                if zh.dim()==3:
                    zh, en = zh.squeeze(1), en.squeeze(1)
                text_embeddings['zh'], text_embeddings['en'], bi_embeddings = \
                    utils.get_model(model).encode_text(zh=zh.cuda(), en=en.cuda()) #already normalize #B,N
            for lang in ['zh','en']:
                cosine_sim[lang].append(torch.bmm(image_features.unsqueeze(1), text_embeddings[lang].unsqueeze(2)))
                # print(len(indices), cosine_sim[lang][-1].shape)
                # input()
            # if len(cosine_sim['zh'])>=100:
            #     break
        for lang in ['zh','en']:
            cosine_sim[lang] = torch.cat(cosine_sim[lang], dim=0).view(-1)
            cosine_sim[lang] = utils.all_gather_tensor(cosine_sim[lang]).cpu()
            # for sim_z, sim_e, ind in zip(cosine_sim_zh, cosine_sim_en, indices):
            #     ind2sim_en[ind], ind2sim_zh[ind] = sim_e.item(), sim_z.item()
        stats = {'name':name, 'num_sample':cosine_sim[lang].shape[0]}
        for lang in ['zh','en']:
            stats[lang] = {'mean': torch.mean(cosine_sim[lang]).item(), 'std':torch.std(cosine_sim[lang]).item()}
        print(stats)
        if is_main_process():
            with open(os.path.join(args.output_dir, f'{name}_eval_log.txt'), 'a') as f:
                f.write(json.dumps(stats) + '\n')
            with open(os.path.join(args.output_dir, f'{name}_sim_shuffled.pkl'),'wb') as f:
                pickle.dump(cosine_sim, f)
            cosine_sim_reordered = reorder(sim=cosine_sim, 
                                           num_replicas=int(os.environ["WORLD_SIZE"]), num=len(loader.dataset))
            with open(os.path.join(args.output_dir, f'{name}_sim.pkl'),'wb') as f:
                pickle.dump(cosine_sim_reordered, f)
        dist_synchronize()
        # input()
        # stats = {'sim_en':{'mean':, 'std':},
        #          'sim_zh':{'mean':,'std':}}
        # with open(os.path.join(args.output_dir, f'{name}_eval_log.txt'), 'a') as f:
        #     f.write(json.dumps(stats) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compute CLIP score', parents=[get_args_parser()])
    args = parser.parse_args()
    args.datasets = args.datasets.split(',')
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)