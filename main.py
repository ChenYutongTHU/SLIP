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
    parser.add_argument('--eval_metric', default='zh_acc1')
    parser.add_argument('--save_per_step', type=int, default=10000)
    parser.add_argument('--toolkit', default='torch', help='torch or bm')
    parser.add_argument('--toolkit_data', default='torch')
    # Data
    parser.add_argument('--dataset', default='yfcc15m', type=str)#, choices=['yfcc15m', 'cc3m', 'cc12m', 'coco', 'redcaps'])
    parser.add_argument('--read_tsv', action='store_true')
    parser.add_argument('--need_only_text', action='store_true')  
    parser.add_argument('--training_mode', default='auto')  
    parser.add_argument('--dataset_distill', default='aic,coco_sbu_vg,cc3m,cc12m,tsl2019,newsv16,wikititles,wikimatrix,thunmt') 
    parser.add_argument('--dataset_contrastive', default='cc3m') 
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
    parser.add_argument('--training_unit', default='epoch')
    parser.add_argument('--steps', default=1000, type=int)
    parser.add_argument('--warmup-steps', default=100, type=int)
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--warmup-epochs', default=1, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=64, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--batch-size_eval', default=128, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--batch_size_distill', default=64, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--batch_size_contrastive', default=64, type=int,
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

    train_dataset = {}
    if args.training_mode == 'auto':
        if args.need_only_text:
            args.training_mode = ['distill']
            args.batch_size_distill = args.batch_size
        else:
            args.training_mode = ['contrastive']
            args.batch_size_contrastive = args.batch_size
    else:
        args.training_mode = args.training_mode.split(',')
    for k in args.training_mode:
        if k=='distill':
            train_dataset[k] = datasets.get_dataset(
                    train_transform, tokenizer, args.dataset_distill, 
                    args=args, need_only_text=True, read_tsv=True)
        elif k=='contrastive':
            train_dataset[k] = datasets.get_dataset(
                    train_transform, tokenizer, args.dataset_contrastive, 
                    args=args, need_only_text=False, read_tsv=True)
        else:
            raise ValueError
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
        dataset2iteration = {
                'train_loader': {}, 'train_sampler': {},
                'train_iter': {}, 'train_step':{}}
        longest_dataset = None
        args.mode2batch_size = {'distill':args.batch_size_distill, 'contrastive':args.batch_size_contrastive}
        for k, d in train_dataset.items():
            if args.distributed:
                dataset2iteration['train_sampler'][k] = torch.utils.data.distributed.DistributedSampler(d)
            else:
                dataset2iteration['train_sampler'][k] = None
            dataset2iteration['train_loader'][k] = torch.utils.data.DataLoader(
                d, batch_size=args.mode2batch_size[k], shuffle=(dataset2iteration['train_sampler'][k] is None),
                num_workers=args.workers, pin_memory=True, sampler=dataset2iteration['train_sampler'][k], drop_last=True)
            dataset2iteration['train_iter'][k] = iter(dataset2iteration['train_loader'][k])
            dataset2iteration['train_step'][k] = [0, len(dataset2iteration['train_loader'][k]), 0] #current, total steps per epoch, cur_epoch
            if longest_dataset==None or len(dataset2iteration['train_loader'][k])>len(dataset2iteration['train_loader'][longest_dataset]):
                longest_dataset = k
            print(f'Dataset {k}, #={len(train_dataset[k])}, #batch={len(dataset2iteration["train_loader"][k])}, sampler-len{len(dataset2iteration["train_sampler"][k])}')
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size_eval, shuffle=(val_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)
    elif args.toolkit=='bm': #(bm.get_rank())
        raise ValueError
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

    if args.training_unit=='epoch':
        lr_schedule = utils.cosine_scheduler_epoch(args.lr, args.lr_end, args.epochs,
            len(dataset2iteration['train_loader'][longest_dataset]) // args.update_freq, warmup_epochs=args.warmup_epochs, start_warmup_value=args.lr_start)
    elif args.training_unit=='step':
        lr_schedule = utils.cosine_scheduler_step(args.lr, args.lr_end, args.steps,
             warmup_steps=args.warmup_steps, start_warmup_value=args.lr_start)
    else:
        raise ValueError
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
    #     val_stats = validate_retrieval_bilingual(model, val_transform, tokenizer, args)
        # print(val_stats)
    print("=> beginning training")
    if args.training_unit=='step':
        args.epochs = math.ceil(args.steps/len(dataset2iteration['train_loader'][longest_dataset]))
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and args.toolkit=='torch':
            dataset2iteration['train_sampler'][longest_dataset].set_epoch(epoch)
        # train for one epoch
        if args.training_unit=='step':
            end_step = args.steps-epoch*len(dataset2iteration['train_loader'][longest_dataset])
        else:
            end_step = len(dataset2iteration['train_loader'][longest_dataset])
        train_stats = train(dataset2iteration,
            longest_dataset, model, criterion, optimizer, scaler, epoch, lr_schedule, args, 
            val_loader, tokenizer, val_transform, end_step)

        if (epoch + 1) % args.eval_freq != 0:
            continue

        if args.model.startswith('SIMCLR'):
            val_stats = {'acc1': -1}
            acc1 = -1
        elif args.model in ['TRIPLET']:
            val_stats = validate_zeroshot_bilingual(val_loader, model, tokenizer, args)
            if args.eval_metric=='max':
                acc1 = max(val_stats['zh_acc1'],val_stats['en_acc1'],
                            val_stats['zh+en_logits_acc1'], 
                            val_stats['zh^en_logits_acc1'], 
                            val_stats['zh+en_probs_acc1']) 
            else:
                acc1 = val_stats[args.eval_metric]
            val_stats_retrieval = validate_retrieval_bilingual(model, val_transform, tokenizer, args)
            val_stats = {**val_stats, **val_stats_retrieval}
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
                     'epoch': epoch, 'step':len(dataset2iteration['train_loader'][longest_dataset])}

        if utils.is_main_process():
            if args.wandb:
                wandb.log(log_stats)
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

def get_next_batch(dataset2iteration, k):
    cur_step, step_per_epoch ,cur_epoch = dataset2iteration['train_step'][k]
    if cur_step>=step_per_epoch:
        dataset2iteration['train_sampler'][k].set_epoch(cur_epoch+1)
        dataset2iteration['train_step'][k][2] += 1
        dataset2iteration['train_iter'][k] = iter(dataset2iteration['train_loader'][k])
        # print('Reset loader', k, dataset2iteration['train_step'][k][2])
        dataset2iteration['train_step'][k][0] = 0 
    dataset2iteration['train_step'][k][0] += 1
    try:
        data = next(dataset2iteration['train_iter'][k])
    except:
        dataset2iteration['train_sampler'][k].set_epoch(cur_epoch+1)
        dataset2iteration['train_step'][k][2] += 1
        dataset2iteration['train_iter'][k] = iter(dataset2iteration['train_loader'][k])    
        dataset2iteration['train_step'][k][0] = 0 
        # print('Reset loader', k, dataset2iteration['cur_epoch'])  
    return data

def train(dataset2iteration, longest_dataset, model, criterion, optimizer, scaler, epoch, lr_schedule, args, val_loader, tokenizer, val_transform, end_step):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = models.get_metric_names(args.model)
    iters_per_epoch = len(dataset2iteration['train_iter'][longest_dataset]) // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    if len(metrics):
        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, *metrics.values()],
            prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    #train_iter = {k:iter(loader) for k, loader in train_loader.items()}
    for data_iter in range(len(dataset2iteration['train_iter'][longest_dataset])):
        #inputs
        optim_iter = data_iter // args.update_freq
        if optim_iter>=end_step:
            break
        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]

        loss_dict, acc_dict, loss = {}, {}, 0 
        loss_dict_per_key, acc_dict_per_key = {}, {}   
        for k in dataset2iteration['train_iter']:
            inputs = get_next_batch(dataset2iteration,k)#next(iterator)
            if args.model not in ['TRIPLET']:
                inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]

            # compute output
            with amp.autocast(enabled=not args.disable_amp):
                if args.model=='TRIPLET':
                    if args.toolkit=='bm':
                        inputs = {k:tensor.cuda(args.gpu, non_blocking=True) for k,tensor in inputs.items()}
                    loss_dict_per_key[k], features_dict, acc_dict_per_key[k] = model(
                        mode=k,
                        img=inputs.get('img',None),
                        en=inputs['en'].squeeze(1), zh=inputs['zh'].squeeze(1))
                    loss_dict = {**loss_dict, **loss_dict_per_key[k]}
                    acc_dict = {**acc_dict, **acc_dict_per_key[k]}
                    loss += loss_dict_per_key[k][f'{k}_total']
                else:
                    raise ValueError
                    outputs = model(*inputs)
                    loss_dict = criterion(outputs)
                    loss = loss_dict['loss']
                loss /= args.update_freq
        if len(metrics)==0:
            metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in list(loss_dict.keys())+list(acc_dict.keys())])
            progress = ProgressMeter(
                iters_per_epoch,
                [batch_time, data_time, mem, *metrics.values()],
                prefix="Epoch: [{}]".format(epoch))
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

        for mode, loss_dict_ in loss_dict_per_key.items():
            for kk in loss_dict_:
                metrics[kk].update(loss_dict_[kk].item(), args.mode2batch_size[mode])
        for mode, acc_dict_ in acc_dict_per_key.items():
            for kk in acc_dict_:
                metrics[kk].update(acc_dict_[kk], args.mode2batch_size[mode])            

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

        if data_iter % args.save_per_step==0 and data_iter!=0:
            if args.model.startswith('SIMCLR'):
                val_stats = {'acc1': -1}
                acc1 = -1
            elif args.model in ['TRIPLET']:
                val_stats = validate_zeroshot_bilingual(val_loader, model, tokenizer, args)
                acc1 = max(val_stats['zh_acc1'],val_stats['en_acc1'],
                            val_stats['zh+en_logits_acc1'], 
                            val_stats['zh^en_logits_acc1'], 
                            val_stats['zh+en_probs_acc1'])
                val_stats_re = validate_retrieval_bilingual(model, val_transform, tokenizer, args)
                val_stats = {**val_stats, **val_stats_re}
            else:
                val_stats = validate_zeroshot(val_loader, model, tokenizer, args)
                acc1 = val_stats['acc1']

            global best_acc1
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            print("=> saving checkpoint")
            utils.save_on_master({
                    'epoch': epoch,
                    'step': data_iter,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'best_acc1': best_acc1,
                    'args': args,
                }, is_best, args.output_dir)

            log_stats = {**{f'test_{k}': v for k, v in val_stats.items()},
                        'epoch': epoch, 'step':data_iter}

            if utils.is_main_process():
                if args.wandb:
                    wandb.log(log_stats)
                with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                    f.write(json.dumps(log_stats) + '\n')

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
            for batch in tqdm(val_dataloader, disable=(not is_main_process())):
                with torch.no_grad():
                    zh_embeddings, en_embeddings, bi_embeddings = \
                        utils.get_model(model).encode_text(
                            zh=batch['zh'].squeeze(1).cuda(), 
                            en=batch['en'].squeeze(1).cuda())
                if zh_embeddings is not None:
                    zh_embeddings_all.append(zh_embeddings)
                if en_embeddings is not None:
                    en_embeddings_all.append(en_embeddings)
            if en_embeddings_all != []:
                en_embeddings_all = torch.cat(en_embeddings_all, dim=0)
            if zh_embeddings_all != []:
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
    val_stats = {}
    DATASET2FILE = {
        'coco-cn': 'data/coco_cn/coco_cn_test.json', #{'img_id':{'zh':,'en':}}
        'aic':  '/data11/private/chenyutong/data/aic/aic_validation.json'
    }
    DATASET2IMG = {
        'coco-cn': 'data/coco_cn/coco_cn_test_images/{}.jpg', #.format(img_id)
        'aic': '/data11/private/chenyutong/data/aic/caption_validation_images_20170910/{}.jpg'
    }
    for dataset in ['coco-cn','aic']:
        print(f'Evaluate Img<->Text Retrieval on {dataset} ...')
        with open(DATASET2FILE[dataset],'r') as f:
            img2captions = json.load(f)

        img_ids, zh_caps, en_caps, pt = [], [], [], 0
        img_files = []
        imgid2capid, capid2imgid = [], []
        for img_id, c in img2captions.items():
            capid2imgid.extend([len(img_ids)]*len(c['zh']))
            img_ids.append(img_id)
            img_files.append(DATASET2IMG[dataset].format(img_id))
            #zh_caps.append(c['zh']) #a list?
            #en_caps.append(c['en'])
            imgid2capid.append(np.arange(pt, pt+len(c['zh'])))
            pt += len(c['zh'])
            assert len(c['zh'])==len(c['en'])
            zh_caps.append(c['zh'])
            en_caps.append(c['en'])
        print('#Images={}, #Captions={}'.format(len(img_files), pt))
        #manually split
        chunk_size = math.ceil(len(img_files)/get_world_size())
        left, right = get_rank()*chunk_size, min((get_rank()+1)*chunk_size, len(img_files))
        img_files_, zh_caps_, en_caps_ = img_files[left:right], zh_caps[left:right], en_caps[left:right]
        if right-left<chunk_size:
            img_files_ +=[img_files[-1] for _ in range(right-left, chunk_size)]
            zh_caps_ +=[zh_caps[-1] for _ in range(right-left, chunk_size)]
            en_caps_ +=[en_caps[-1] for _ in range(right-left, chunk_size)]
        val_dataset = datasets.TripletDataset_from_rawfile(
            img_file=img_files_, zh=zh_caps_, en=en_caps_, preprocess=val_transform, tokenizer=tokenizer)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=128, shuffle=False,
                num_workers=args.workers, pin_memory=True, drop_last=False)

        image_embeddings_all = []
        en_embeddings_all, zh_embeddings_all, bi_embeddings_all = [], [], []
        for img, en, zh in tqdm(val_dataloader, disable=(not is_main_process())):
            #B,k,L
            en, zh = en.reshape(-1,en.shape[-1]), zh.reshape(-1,zh.shape[-1])
            #en, zh = en.squeeze(1), zh.squeeze(1) #
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

        dist_synchronize()
        image_embeddings_all = utils.all_gather_tensor(image_embeddings_all)
        image_embeddings_all = image_embeddings_all[:len(img_files)]
        text_embeddings = {}
        for k,v in [['zh',zh_embeddings_all],['en',en_embeddings_all],['bi',bi_embeddings_all]]:
            if v[-1] is not None:
                text_embeddings[k] = torch.cat(v, dim=0)
                text_embeddings[k] = utils.all_gather_tensor(text_embeddings[k])
                text_embeddings[k] = text_embeddings[k][:pt]
                print(k, len(text_embeddings[k]))
        #gather
        for lang in ['zh','en','bi']:
            if not lang in text_embeddings:
                continue
            #direction Image->Text
            acc1, acc5 = [], []
            for i, (image_embedding, capids) in enumerate(zip(image_embeddings_all, imgid2capid)):
                i2t_sim = image_embedding@text_embeddings[lang].t() #D,(D,nT)
                pred_topk = (torch.topk(i2t_sim, k=5, dim=-1).indices).tolist()
                acc1.append(pred_topk[0] in capids)
                acc5.append(np.max([cid in pred_topk for cid in capids]))
            acc1, acc5 = np.mean(acc1)*100, np.mean(acc5)*100
            val_stats[f'{dataset}_I2T_acc1'] = acc1
            val_stats[f'{dataset}_I2T_acc5'] = acc5
            print('{}({}) acc1={:.2f} acc5={:.2f}'.format('Image->Text', lang, acc1, acc5))
            #direction Text->Image
            acc1, acc5 = [], []
            for i, (text_embedding, imgid) in enumerate(zip(text_embeddings[lang], capid2imgid)):
                t2i_sim = text_embedding@image_embeddings_all.t()
                pred_topk = (torch.topk(t2i_sim, k=5, dim=-1).indices).tolist()
                acc1.append(pred_topk[0]==imgid)
                acc5.append(imgid in pred_topk)    
            acc1, acc5 = np.mean(acc1)*100, np.mean(acc5)*100
            val_stats[f'{dataset}_T2I_acc1'] = acc1
            val_stats[f'{dataset}_T2I_acc5'] = acc5
            print('{}({}) acc1={:.2f} acc5={:.2f}'.format('Text->Image', lang, acc1, acc5))                                
    #dist.barrier()
    dist_synchronize()
    model.train()
    return val_stats

def validate_retrieval_bilingual_single(model, val_transform, tokenizer, args):
    model.eval()
    DATASET2FILE = {
        'coco-cn': 'data/coco_cn/coco_cn_test.json', #{'img_id':{'zh':,'en':}}
        'aic':  '/data11/private/chenyutong/data/aic/aic_validation.json'
    }
    DATASET2IMG = {
        'coco-cn': 'data/coco_cn/coco_cn_test_images/{}.jpg', #.format(img_id)
        'aic': '/data11/private/chenyutong/data/aic/caption_validation_images_20170910/{}.jpg'
    }
    if utils.is_main_process(): #one gpu is enough
        for dataset in ['coco-cn','aic']:
            print(f'Evaluate Img<->Text Retrieval on {dataset} ...')
            with open(DATASET2FILE[dataset],'r') as f:
                img2captions = json.load(f)

            img_ids, zh_caps, en_caps, pt = [], [], [], 0
            img_files = []
            imgid2capid, capid2imgid = [], []
            for img_id, c in img2captions.items():
                capid2imgid.extend([len(img_ids)]*len(c['zh']))
                img_ids.append(img_id)
                img_files.append(DATASET2IMG[dataset].format(img_id))
                #zh_caps.append(c['zh']) #a list?
                #en_caps.append(c['en'])
                imgid2capid.append(np.arange(pt, pt+len(c['zh'])))
                pt += len(c['zh'])
                assert len(c['zh'])==len(c['en'])
                zh_caps.append(c['zh'])
                en_caps.append(c['en'])
            print('#Images={}, #Captions={}'.format(len(img_files), pt))
            val_dataset = datasets.TripletDataset_from_rawfile(
                img_file=img_files, zh=zh_caps, en=en_caps, preprocess=val_transform, tokenizer=tokenizer)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=128, shuffle=False,
                    num_workers=args.workers, pin_memory=True, drop_last=False)

            print('Compute embeddings ... ')
            image_embeddings_all = []
            en_embeddings_all, zh_embeddings_all, bi_embeddings_all = [], [], []
            for img, en, zh in tqdm(val_dataloader, disable=(not is_main_process())):
                #B,k,L
                en, zh = en.reshape(-1,en.shape[-1]), zh.reshape(-1,zh.shape[-1])
                #en, zh = en.squeeze(1), zh.squeeze(1) #
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
            text_embeddings = {}
            for k,v in [['zh',zh_embeddings_all],['en',en_embeddings_all],['bi',bi_embeddings_all]]:
                if v[-1] is not None:
                    text_embeddings[k] = torch.cat(v, dim=0)
            for lang in ['zh','en','bi']:
                if not lang in text_embeddings:
                    continue
                #direction Image->Text
                acc1, acc5 = [], []
                for i, (image_embedding, capids) in enumerate(zip(image_embeddings_all, imgid2capid)):
                    i2t_sim = image_embedding@text_embeddings[lang].t() #D,(D,nT)
                    pred_topk = (torch.topk(i2t_sim, k=5, dim=-1).indices).tolist()
                    acc1.append(pred_topk[0] in capids)
                    acc5.append(np.max([cid in pred_topk for cid in capids]))
                acc1, acc5 = np.mean(acc1)*100, np.mean(acc5)*100
                print('{}({}) acc1={:.2f} acc5={:.2f}'.format('Image->Text', lang, acc1, acc5))
                #direction Text->Image
                acc1, acc5 = [], []
                for i, (text_embedding, imgid) in enumerate(zip(text_embeddings[lang], capid2imgid)):
                    t2i_sim = text_embedding@image_embeddings_all.t()
                    pred_topk = (torch.topk(t2i_sim, k=5, dim=-1).indices).tolist()
                    acc1.append(pred_topk[0]==imgid)
                    acc5.append(imgid in pred_topk)    
                acc1, acc5 = np.mean(acc1)*100, np.mean(acc5)*100
                print('{}({}) acc1={:.2f} acc5={:.2f}'.format('Text->Image', lang, acc1, acc5))                                
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
            for k,v in [['zh', zh_embeddings],['en', en_embeddings]]:
                if v is not None:
                    v = v.mean(dim=0)
                    v = v/v.norm(dim=-1, keepdim=True)
                    lang2text_features[k].append(v)
            
        for k in lang2text_features:
            if len(lang2text_features[k])==0:
                continue
            lang2text_features[k] = torch.stack(lang2text_features[k], dim=0)
        lang2text_features = {k:v for k,v in lang2text_features.items() if len(v)}
        end = time.time()
        SAVE = True
        if SAVE:
            fname2logits = {}
        logit_scale = utils.get_model(model).logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)
        print('=> encoding images and compute similarities')
        for i, (images, target, fnames) in enumerate(val_loader):
            # if i>=5:
            #     break
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
            if 'zh' in lang2text_features and 'en' in lang2text_features:
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
            else:
                for k in ['zh+en_logits', 'zh^en_logits', 'zh+en_probs','bi','zh+en+bi_logits','zh+en+bi_probs','zh','en']:
                    if k in lang2text_features:
                        continue
                    top1[k].update(0, images.size(0))
                    top5[k].update(0, images.size(0)) 
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
