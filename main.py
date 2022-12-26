# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from collections import OrderedDict
import json
import math
from multiprocessing import synchronize
import os
import sys
import time
import wandb

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from imagenetv2_pytorch import ImageNetValDataset
import datasets
import models
from tokenizer import SimpleTokenizer
import utils
from utils import get_rank, get_world_size, is_main_process, dist_synchronize, dist_all_reduce
from PIL import Image
from tqdm import tqdm
from model_center.dataset import DistributedDataLoader

def get_args_parser():
    parser = argparse.ArgumentParser(description='SLIP training and evaluation', add_help=False)
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--save_per_step', type=int, default=10000)
    parser.add_argument('--toolkit', default='torch', help='torch or bm')
    parser.add_argument('--toolkit_data', default='torch')
    # Data
    parser.add_argument('--dataset', default='yfcc15m', type=str)#, choices=['yfcc15m', 'cc3m', 'cc12m', 'coco', 'redcaps'])
    parser.add_argument('--read_tsv', action='store_true')
    parser.add_argument('--need_only_text', action='store_true')    
    parser.add_argument('--root', default='', type=str,
                        help='path to dataset root')
    parser.add_argument('--metadata', default='yfcc15m.pkl', type=str,
                        help='path to metadata file (see README for details)')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    # Model
    parser.add_argument('--model', default='SLIP_VITB16', type=str)
    parser.add_argument('--model_cfg_path', default='', type=str)
    parser.add_argument('--ssl-mlp-dim', default=4096, type=int,
                        help='hidden dim of SimCLR mlp projection head')
    parser.add_argument('--ssl-emb-dim', default=256, type=int,
                        help='output embed dim of SimCLR mlp projection head')
    parser.add_argument('--ssl-scale', default=1.0, type=float,
                        help='loss scale for SimCLR objective')
    parser.add_argument('--ssl-temp', default=0.1, type=float,
                        help='softmax temperature for SimCLR objective')
    parser.add_argument('--resume', type=str, default='', help='path to resume from')
    parser.add_argument('--reset_optimization', action='store_true', help='not resume epoch')
    # Training
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=64, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--eval-freq', default=1, type=int)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')
    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--evaluate', action='store_true', help='eval only')
    parser.add_argument('--evaluate_retrieval', action='store_true', help='eval only')
    parser.add_argument('--evaluate_text_retrieval', action='store_true', help='eval only')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
    return parser

best_acc1 = 0


def main(args):
    utils.init_distributed_mode(args)
    global best_acc1
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create model
    print("=> creating model: {}".format(args.model))
    model = getattr(models, args.model)(
        ssl_mlp_dim=args.ssl_mlp_dim, ssl_emb_dim=args.ssl_emb_dim,
        model_cfg_path=args.model_cfg_path, toolkit=args.toolkit)
    model.cuda(args.gpu)
    if args.toolkit=='bm':
        assert args.toolkit_data=='bm'
    if args.distributed and args.toolkit=='torch':
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200)

    # define loss function (criterion) and optimizer
    criterion = models.get_loss(args.model, args.ssl_temp, args.ssl_scale)
    if criterion is not None:
        criterion = criterion.cuda(args.gpu)
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                    eps=args.eps, weight_decay=args.wd)
    scaler = amp.GradScaler(enabled=not args.disable_amp)

    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(result)
            if args.reset_optimization:
                args.start_epoch, epoch = 0, 0
            else:
                epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
                args.start_epoch = epoch
                optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
                scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
                best_acc1 = checkpoint['best_acc1']
            print("=> load resume checkpoint '{}' (epoch {})"
                .format(args.resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # auto-resume from latest checkpoint in output directory
        latest = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            args.start_epoch = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            best_acc1 = latest_checkpoint['best_acc1']
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))

    cudnn.benchmark = True

    # Data loading code
    print("=> creating dataset")
    if args.model.startswith('TRIPLET'):
        tokenizer = {'en':utils.get_model(model).tokenize_en, 'zh':utils.get_model(model).tokenize_zh}
    else:
        tokenizer = SimpleTokenizer()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            normalize
        ])
    val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

    train_dataset = datasets.get_dataset(train_transform, tokenizer, args)
    
    '''
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, 'dataset_catalog.json')) as f:
        root = json.load(f)['imagenet']['path']
    val_dataset = ImageFolder(os.path.join(root, 'val'), val_transform)
    '''
    val_dataset = ImageNetValDataset(
        transform=val_transform, 
        location='../CLIP_distillation')

    # dist eval resamples data to pad uneven batch sizes
    # make sure num_samples = 0 mod num_gpus for exact acc

    if args.toolkit in 'torch': #toolkit! instead of toolkit_data (dist.get_rank())
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=128, shuffle=(val_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)
    elif args.toolkit=='bm': #(bm.get_rank())
        train_loader = DistributedDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DistributedDataLoader(val_dataset, batch_size=128, shuffle=False, 
            drop_last=False,num_workers=args.workers,pin_memory=True)



    if args.evaluate:
        if args.model.startswith('SIMCLR'):
            print('zero-shot evaluation not supported with ssl-only model.')
            return
        elif args.model in ['TRIPLET'] :
            if args.evaluate_retrieval:
                zero_stats = validate_retrieval_bilingual(model, val_transform, tokenizer, args)
            elif args.evaluate_text_retrieval:
                zero_stats = validate_retrieval_bilingual_text(model, tokenizer, args)
            else:
                zero_stats = validate_zeroshot_bilingual(val_loader, model, tokenizer, args)
        else:
            zero_stats = validate_zeroshot(val_loader, model, tokenizer, args)
        if utils.is_main_process():
            with open(os.path.join(args.output_dir, 'eval_log.txt'), 'a') as f:
                f.write(json.dumps(zero_stats) + '\n')
        return

    lr_schedule = utils.cosine_scheduler(args.lr, args.lr_end, args.epochs,
        len(train_loader) // args.update_freq, warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start)

    if utils.is_main_process() and args.wandb:
        wandb.login(key='5421ff43bf1e3a6e19103432d161c885d4bbeda8')
        #wandb_id = os.path.split(args.output_dir)[-1]
        wandb.init(project='slip', config=args, reinit=True)
        wandb.run.name = os.path.split(args.output_dir)[-1]
        wandb.run.save()

    print(args)
    # if args.model in ['TRIPLET']:
    #     print("=> Evaluation before training")
    #     val_stats = validate_zeroshot_bilingual(val_loader, model, tokenizer, args)
    #     print(val_stats)
    print("=> beginning training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and args.toolkit=='torch':
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train_stats = train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args)

        if (epoch + 1) % args.eval_freq != 0:
            continue

        if args.model.startswith('SIMCLR'):
            val_stats = {'acc1': -1}
            acc1 = -1
        elif args.model in ['TRIPLET']:
            val_stats = validate_zeroshot_bilingual(val_loader, model, tokenizer, args)
            acc1 = max(val_stats['zh_acc1'],val_stats['en_acc1'],
                        val_stats['zh+en_logits_acc1'], 
                        val_stats['zh^en_logits_acc1'], 
                        val_stats['zh+en_probs_acc1'])
        else:
            val_stats = validate_zeroshot(val_loader, model, tokenizer, args)
            acc1 = val_stats['acc1']

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        print("=> saving checkpoint")
        utils.save_on_master({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc1': best_acc1,
                'args': args,
            }, is_best, args.output_dir)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch}

        if utils.is_main_process():
            if args.wandb:
                wandb.log(log_stats)
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')


def train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = models.get_metric_names(args.model)
    iters_per_epoch = len(train_loader) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    if len(metrics):
        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, *metrics.values()],
            prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]
        
        if args.model not in ['TRIPLET']:
            inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]

        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            if args.model=='TRIPLET':
                if args.toolkit=='bm':
                    inputs = {k:tensor.cuda(args.gpu, non_blocking=True) for k,tensor in inputs.items()}
                loss_dict, features_dict, acc_dict = model(
                    img=inputs.get('img',None),
                    en=inputs['en'].squeeze(1), zh=inputs['zh'].squeeze(1))
                loss = loss_dict['total']
                if len(metrics)==0:
                    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in list(loss_dict.keys())+list(acc_dict.keys())])
                    progress = ProgressMeter(
                        iters_per_epoch,
                        [batch_time, data_time, mem, *metrics.values()],
                        prefix="Epoch: [{}]".format(epoch))
            else:
                outputs = model(*inputs)
                loss_dict = criterion(outputs)
                loss = loss_dict['loss']
            loss /= args.update_freq

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()

        if (data_iter + 1) % args.update_freq != 0:
            continue

        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        # clamp logit scale to [0, 100]
        if args.model.startswith('SIMCLR'):
            logit_scale = 0
        else:
            utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
            logit_scale = utils.get_model(model).logit_scale.exp().item()

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)
        for k in acc_dict:
            metrics[k].update(acc_dict[k], args.batch_size)            

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if utils.is_main_process() and args.wandb:
                wandb.log({**{f'loss/{k}': v.item() for k, v in loss_dict.items()},
                        **{f'acc/{k}': v for k, v in acc_dict.items()},
                        'scaler': scaler.get_scale(),
                        'logit': logit_scale})
            progress.display(optim_iter)

        if data_iter % args.save_per_step==0:
            print("=> saving checkpoint")
            utils.save_on_master({
                    'epoch': epoch,
                    'step': data_iter,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_acc1': best_acc1,
                    'args': args,
                }, False, args.output_dir)
    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}

def validate_retrieval_bilingual_text(model, tokenizer, args):
    model.eval()
    DATASET2FILE = {
        'coco-cn': 'data/coco_cn/coco_cn_test.json', #{'img_id':{'zh':,'en':}}
        'un':'data/text_retrieval/un10k.json'
    }    
    if utils.is_main_process(): #one gpu is enough
        for dataset in ['un', 'coco-cn']:
            print(f'Evaluate En<->Zh Retrieval on {dataset} ...')
            with open(DATASET2FILE[dataset],'r') as f:
                id2texts = json.load(f)
            zh_texts, en_texts = [], []
            for _, c in id2texts.items():
                zh_texts.append(c['zh'])
                en_texts.append(c['en'])    
            val_dataset = datasets.BilingualDataset_from_list(
                zh=zh_texts, en=en_texts, tokenizer=tokenizer)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)          
            zh_embeddings_all, en_embeddings_all = [],[]
            for batch in tqdm(val_dataloader):
                with torch.no_grad():
                    zh_embeddings, en_embeddings, bi_embeddings = \
                        utils.get_model(model).encode_text(
                            zh=batch['zh'].squeeze(1).cuda(), 
                            en=batch['en'].squeeze(1).cuda())
                zh_embeddings_all.append(zh_embeddings)
                en_embeddings_all.append(en_embeddings)
            en_embeddings_all = torch.cat(en_embeddings_all, dim=0)
            zh_embeddings_all = torch.cat(zh_embeddings_all, dim=0) #1k, 512
            
            #retrieval
            sim_en_zh = en_embeddings_all@zh_embeddings_all.t() #1k en, 1k zh
            for direction in ['En->Zh','Zh->En']:
                if direction=='Zh->En':
                    sim_ = sim_en_zh.t() #text image
                else:
                    sim_ = sim_en_zh
                pred_topk = torch.topk(sim_, k=5, dim=-1).indices #1k, 5 label: 0,1,2,3,4
                label = torch.arange(en_embeddings_all.shape[0])[:,None].cuda() #1k,1
                acc1 = torch.mean((pred_topk[:,0:1]==label).float())
                acc5 = torch.mean(torch.max((pred_topk==label).float(),dim=-1).values)
                acc1, acc5 = (acc1*100).item(), (acc5*100).item()
                print('{} acc1={:.2f} acc5={:.2f}'.format(direction,  acc1, acc5))  
                
                src_texts = zh_texts if direction=='Zh->En' else en_texts
                tgt_texts = en_texts if direction=='Zh->En' else zh_texts 
                with open(os.path.join(args.output_dir, f'retrieve_{dataset}_{direction}.txt'),'w') as f:
                    for si in range(50):
                        f.writelines(src_texts[si][0]+'\n')   
                        f.writelines('Correct Answer: '+tgt_texts[si][0]+'\n')
                        f.writelines('Retrieve Results:\n') 
                        for ti in pred_topk[si]:
                            f.writelines(tgt_texts[ti.item()][0]+'\n')
                        f.writelines('\n')    
                print('Write some samples in '+os.path.join(args.output_dir, f'retrieve_{dataset}_{direction}.txt'))

def validate_retrieval_bilingual(model, val_transform, tokenizer, args):
    model.eval()
    DATASET2FILE = {
        'coco-cn': 'data/coco_cn/coco_cn_test.json', #{'img_id':{'zh':,'en':}}
    }
    DATASET2IMG = {
        'coco-cn': 'data/coco_cn/coco_cn_test_images/{}.jpg' #.format(img_id)
    }
    if utils.is_main_process(): #one gpu is enough
        for dataset in ['coco-cn']:
            print(f'Evaluate Img<->Text Retrieval on {dataset} ...')
            with open(DATASET2FILE[dataset],'r') as f:
                img2captions = json.load(f)

            img_ids, zh_caps, en_caps = [], [], []
            img_files = []
            for img_id, c in img2captions.items():
                img_ids.append(img_id)
                img_files.append(DATASET2IMG[dataset].format(img_id))
                zh_caps.append(c['zh'])
                en_caps.append(c['en'])

            val_dataset = datasets.TripletDataset_from_rawfile(
                img_file=img_files, zh=zh_caps, en=en_caps, preprocess=val_transform, tokenizer=tokenizer)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=128, shuffle=False,
                    num_workers=args.workers, pin_memory=True, drop_last=False)

            print('Compute embeddings ... ')
            image_embeddings_all = []
            en_embeddings_all, zh_embeddings_all, bi_embeddings_all = [], [], []
            for img, en, zh in tqdm(val_dataloader):
                en, zh = en.squeeze(1), zh.squeeze(1)
                with torch.no_grad():
                    image_features = utils.get_model(model).encode_image(img.cuda())
                    zh_embeddings, en_embeddings, bi_embeddings = \
                        utils.get_model(model).encode_text(zh=zh.cuda(), en=en.cuda()) #already normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                image_embeddings_all.append(image_features)
                en_embeddings_all.append(en_embeddings)
                zh_embeddings_all.append(zh_embeddings)
                bi_embeddings_all.append(bi_embeddings)
            image_embeddings_all = torch.cat(image_embeddings_all, dim=0)
            en_embeddings_all = torch.cat(en_embeddings_all, dim=0)
            zh_embeddings_all = torch.cat(zh_embeddings_all, dim=0) #1k, 512
            text_embeddings = {'zh': zh_embeddings_all, 'en':en_embeddings_all}
            if bi_embeddings is not None:
                bi_embeddings_all = torch.cat(bi_embeddings_all, dim=0)
                text_embeddings['bi'] = bi_embeddings_all

            for lang in ['zh','en','bi']:
                if not lang in text_embeddings:
                    continue
                sim = image_embeddings_all@text_embeddings[lang].t() #1k img, 1k text
                for direction in ['Image->Text','Text->Image']:
                    if direction=='Text->Image':
                        sim_ = sim.t() #text image
                    else:
                        sim_ = sim
                    pred_topk = torch.topk(sim_, k=5, dim=-1).indices #1k, 5 label: 0,1,2,3,4
                    label = torch.arange(image_embeddings_all.shape[0])[:,None].cuda() #1k,1
                    acc1 = torch.mean((pred_topk[:,0:1]==label).float())
                    acc5 = torch.mean(torch.max((pred_topk==label).float(),dim=-1).values)
                    acc1, acc5 = (acc1*100).item(), (acc5*100).item()
                    print('{}({}) acc1={:.2f} acc5={:.2f}'.format(direction, lang, acc1, acc5))
    #dist.barrier()
    dist_synchronize()
    model.train()

def validate_zeroshot_bilingual(val_loader, model, tokenizer, args):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    top1, top5 = {},{}
    keys = ['zh','en', 'zh+en_logits', 'zh^en_logits', 'zh+en_probs',
        'bi','zh+en+bi_logits','zh+en+bi_probs']#, 'zh+en_feature']
    for k in keys:
        top1[k] = AverageMeter(f'Acc@1_{k}', ':6.2f')
        top5[k] = AverageMeter(f'Acc@5_{k}', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time]+[top1[k] for k in keys] + [top5[k] for k in keys],
        prefix='Test: ')

    cwd = os.path.dirname(os.path.realpath(__file__))
    with torch.no_grad():
        id2lang_text,lang2text_features = {}, {'zh':[],'en':[]}
        for lang in ['zh','en']:
            lang2text_features[lang] = []
            templates = open(os.path.join(cwd,f'{lang}_templates.txt'),'r').readlines()
            templates = [t.strip() for t in templates]
            labels = json.load(open(os.path.join(cwd,f'imagenet_class_name_{lang}.json'),'r'))
            for id in sorted(labels):
                l = labels[id]
                texts = [t.format(l) for t in templates]
                if not id in id2lang_text:
                    id2lang_text[id] = {}
                id2lang_text[id][lang] = tokenizer[lang](texts)
        print('=> encoding captions')
        for id in sorted(labels):
            zh_embeddings, en_embeddings, bi_embeddings = utils.get_model(model).encode_text(
                    zh=id2lang_text[id]['zh'].cuda(args.gpu, non_blocking=True), 
                    en=id2lang_text[id]['en'].cuda(args.gpu, non_blocking=True)) 
            #print('beform norm', torch.nn.functional.mse_loss(zh_embeddings, en_embeddings, reduction='mean'))
            zh_embeddings, en_embeddings = zh_embeddings.mean(dim=0), en_embeddings.mean(dim=0)
            en_embeddings = en_embeddings / en_embeddings.norm(dim=-1, keepdim=True)
            lang2text_features['en'].append(en_embeddings)
            zh_embeddings = zh_embeddings / zh_embeddings.norm(dim=-1, keepdim=True)
            lang2text_features['zh'].append(zh_embeddings)
            #print('after norm', torch.nn.functional.mse_loss(zh_embeddings, en_embeddings, reduction='mean'))
            if bi_embeddings is not None:
                if not 'bi' in lang2text_features:
                    lang2text_features['bi'] = []
                bi_embeddings = bi_embeddings.mean(dim=0)
                bi_embeddings = bi_embeddings / bi_embeddings.norm(dim=-1, keepdim=True)
                lang2text_features['bi'].append(bi_embeddings)
        for k in lang2text_features:
            if len(lang2text_features[k])==0:
                continue
            lang2text_features[k] = torch.stack(lang2text_features[k], dim=0)
        end = time.time()
        SAVE = True
        if SAVE:
            fname2logits = {}
        logit_scale = utils.get_model(model).logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)
        print('=> encoding images and compute similarities')
        for i, (images, target, fnames) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # encode images
            image_features = utils.get_model(model).encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_image, probs_per_image = {}, {}
            for lang, text_features in lang2text_features.items():
                #logits_per_image[lang] = logit_scale*image_features @ text_features.t() #B,V
                logits_per_image[lang] = logit_scale*image_features @ text_features.t() 
                probs_per_image[lang] = torch.nn.functional.softmax(logits_per_image[lang],dim=-1) #B,V
                acc1, acc5 = accuracy(logits_per_image[lang], target, topk=(1, 5))
                acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
                top1[lang].update(acc1.item(), images.size(0))
                top5[lang].update(acc5.item(), images.size(0))
            if 'bi' not in lang2text_features:
                top1['bi'].update(0, images.size(0))
                top5['bi'].update(0, images.size(0))

            #bilingual ensemble
            if SAVE:
                for fname, logits_zh, logits_en in zip(fnames, logits_per_image['zh'], logits_per_image['en']):
                    fname2logits[fname] = {'zh':logits_zh.cpu(), 'en':logits_en.cpu()}

            logits_per_image_bilingual = logits_per_image['zh']+logits_per_image['en'] #B,V
            acc1, acc5 = accuracy(logits_per_image_bilingual, target, topk=(1, 5))
            acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
            top1['zh+en_logits'].update(acc1.item(), images.size(0))
            top5['zh+en_logits'].update(acc5.item(), images.size(0))    

            if 'bi' in logits_per_image:
                logits_per_image_bilingual = logits_per_image['zh']+logits_per_image['en']+logits_per_image['bi'] #B,V
                acc1, acc5 = accuracy(logits_per_image_bilingual, target, topk=(1, 5))
                acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
                top1['zh+en+bi_logits'].update(acc1.item(), images.size(0))
                top5['zh+en+bi_logits'].update(acc5.item(), images.size(0))
            else:
                top1['zh+en+bi_logits'].update(0, images.size(0))
                top5['zh+en+bi_logits'].update(0, images.size(0))                        

            logits_per_image_bilingual =  torch.stack([logits_per_image['zh'],logits_per_image['en']], dim=-1)  #B,V,2
            logits_per_image_bilingual = torch.max(logits_per_image_bilingual, dim=-1).values  
            acc1, acc5 = accuracy(logits_per_image_bilingual, target, topk=(1, 5))
            acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
            top1['zh^en_logits'].update(acc1.item(), images.size(0))
            top5['zh^en_logits'].update(acc5.item(), images.size(0))

            probs_per_image_bilingual = probs_per_image['zh']+probs_per_image['en'] #B,V
            acc1, acc5 = accuracy(probs_per_image_bilingual, target, topk=(1, 5))
            acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
            top1['zh+en_probs'].update(acc1.item(), images.size(0))
            top5['zh+en_probs'].update(acc5.item(), images.size(0)) 

            if 'bi' in probs_per_image:
                probs_per_image_bilingual = probs_per_image['zh']+probs_per_image['en']+probs_per_image['bi'] #B,V
                acc1, acc5 = accuracy(probs_per_image_bilingual, target, topk=(1, 5))
                acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
                top1['zh+en+bi_probs'].update(acc1.item(), images.size(0))
                top5['zh+en+bi_probs'].update(acc5.item(), images.size(0)) 
            else:
                top1['zh+en+bi_probs'].update(0, images.size(0))
                top5['zh+en+bi_probs'].update(0, images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        if SAVE:
            torch.save(fname2logits, os.path.join(args.output_dir,f'fname2logits_rank{get_rank()}.bin'))

    progress.synchronize()
    return_dict = {}
    for k in keys:
        top1_, top5_ = top1[k], top5[k]
        print(k+': 0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1_, top5=top5_))
        return_dict[f'{k}_acc1'], return_dict[f'{k}_acc5'] = top1_.avg, top5_.avg
    return return_dict

def validate_zeroshot(val_loader, model, tokenizer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    print('=> encoding captions')
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, 'templates.json')) as f:
        templates = json.load(f)['imagenet']

    with open(os.path.join(cwd, 'labels.json')) as f:
        labels = json.load(f)['imagenet']

    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer['en'](texts).cuda(args.gpu, non_blocking=True)
            if args.model.startswith('TRIPLET'):
                class_embeddings = utils.get_model(model).model_en.encode_text(texts)
            else:
                class_embeddings = utils.get_model(model).encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # encode images
            image_features = utils.get_model(model).encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_image = image_features @ text_features.t()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits_per_image, target, topk=(1, 5))
            acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    progress.synchronize()
    print('0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    return {'acc1': top1.avg, 'acc5': top5.avg}


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        #dist.barrier()
        dist_synchronize()
        #dist.all_reduce(t)
        dist_all_reduce(t)
        
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        if self.count==0:
            print(self.name, 'count=0!')
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SLIP training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.isfile(args.root) and '/data11/private/chenyutong/data' in args.root:
        args.root = args.root.replace('/data11/private/chenyutong/data','../../../data')
        print(f'training on itp instead of on thunlp, args.root={args.root}')
    main(args)
