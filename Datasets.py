# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict
import json
import os
import pickle
import zipfile

import numpy as np
from PIL import Image, ImageFile

import torch
from torchvision import transforms
from torchvision import datasets as t_datasets
from read_tsv import TSVFile, img_from_base64
import utils
# from model_center.dataset import DistributedMMapIndexedDataset, MMapIndexedDataset
# import bmtrain as bmt
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def yfcc_loader(root, index):
    index = format(index, "0>8d")
    repo = index[:2]
    z = index[2: 5]
    file_img = index[5:] + '.jpg'
    path_zip = os.path.join(root, 'images', repo, z) + '.zip'
    with zipfile.ZipFile(path_zip, 'r') as myzip:
        img = Image.open(myzip.open(file_img))
    return img.convert('RGB')


class ImageCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset, root, metadata):
        self.dataset = dataset
        self.root = root
        if self.dataset == 'yfcc15m':
            with open(metadata, 'rb') as f:
                self.samples = pickle.load(f)
        elif self.dataset == 'coco':
            samples = defaultdict(list)
            with open(metadata) as f:
                annotations = json.load(f)['annotations']
            for ann in annotations:
                samples[ann['image_id']].append(ann['caption'])
            self.samples = [(k, v) for k, v in samples.items()]
        elif self.dataset == 'cc12m' or self.dataset == 'cc3m':
            #self.samples = np.load(metadata, allow_pickle=True)
            self.tsv = TSVFile(root)
        elif self.dataset == 'redcaps':
            with open(metadata) as f:
                annotations = json.load(f)
            self.samples = [(ann['image_id'], ann['subreddit'], ann['caption']) for ann in annotations]

    def get_raw_item(self, i):
        if self.dataset == 'yfcc15m':
            index, title, desc = self.samples[i]
            caption = np.random.choice([title, desc])
            img = yfcc_loader(self.root, index)
        elif self.dataset == 'coco':
            index, captions = self.samples[i]
            path = os.path.join(self.root, 'train2017', '{:012d}.jpg'.format(index))
            img = pil_loader(path)
            caption = np.random.choice(captions)
        elif self.dataset in ['cc3m', 'cc12m']:
            row = self.tsv.seek(i)
            img = img_from_base64(row[-1])
            caption = row[-2]     
        # elif self.dataset == 'cc3m':
        #     ann = self.samples[i]
        #     filename, captions = ann['image_id'], ann['captions']
        #     path = os.path.join(self.root, str(filename))
        #     img = pil_loader(path)
        #     caption = np.random.choice(captions)
        # elif self.dataset == 'cc12m':
        #     ann = self.samples[i]
        #     filename, captions = ann['image_name'], ann['captions']
        #     path = os.path.join(self.root, filename)
        #     img = pil_loader(path)
        #     caption = np.random.choice(captions)

        elif self.dataset == 'redcaps':
            image_id, subreddit, caption = self.samples[i]
            path = os.path.join(self.root, subreddit, f"{image_id}.jpg")
            img = pil_loader(path)

        return img, caption

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        if self.dataset in ['cc3m', 'cc12m']:
            return self.tsv.num_rows()
        else:
            return len(self.samples)


class ImageCaptionDatasetCLIP(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform=None, tokenizer=None):
        super().__init__(dataset, root, metadata)

        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, caption = self.get_raw_item(i)

        # apply transformation
        if self.transform is not None:
            image = self.transform(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return image, caption


class TextDistillDataset(torch.utils.data.Dataset):
    def __init__(self, names, catalog):#zh : MMapIndexedDataset, en : MMapIndexedDataset):
        super().__init__()
        self.names = names.split(',')
        self.name2IndexedDataset = {}
        self.index2data = []
        for name in self.names:
            self.name2IndexedDataset[name] = {
                'zh': MMapIndexedDataset(catalog[name]['indexed_dataset']['zh']),
                'en': MMapIndexedDataset(catalog[name]['indexed_dataset']['en']),
            }
            print(name, f'#={len(self.name2IndexedDataset[name]["zh"])}')
            self.index2data.extend([(name,i) for i in range(len(self.name2IndexedDataset[name]['zh']))])
        #assert len(self.zh)==len(self.en), (len(self.zh), len(self.en))
        print(f'Total number={len(self.index2data)}')

    def __len__(self):
        return len(self.index2data)
    
    def __getitem__(self, ith):
        name, ith = self.index2data[ith]
        zh_input_ids = self.name2IndexedDataset[name]['zh'][ith] # get the i-th data from DistributedMMapIndexedDataset]
        en_input_ids = self.name2IndexedDataset[name]['en'][ith]
        return {'zh': zh_input_ids, 'en': en_input_ids}

class BilingualDataset_from_TSV(torch.utils.data.Dataset):
    def __init__(self, names, catalog, tokenizer):
        super().__init__()
        self.names = names.split(',')
        self.name2tsv = {}
        self.index2data = []
        for name in self.names:
            self.name2tsv[name] = TSVFile(catalog[name]["tsv_txt"])
            self.index2data.extend([(name,i) for i in range(len(self.name2tsv[name]))])
            print(f'Load {name}(#={len(self.name2tsv[name])}) from {catalog[name]["tsv_txt"]} ...')
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.index2data)

    def __getitem__(self, index):
        name, ith = self.index2data[index]
        en, zh = self.name2tsv[name].seek(ith)
        en = self.tokenizer['en'](en)
        zh = self.tokenizer['zh'](zh)        
        return {'en':en, 'zh':zh}

class ParallelDataset_from_TSV(torch.utils.data.Dataset):
    def __init__(self, names, catalog, tokenizer):
        super().__init__()
        self.names = names.split(',')
        self.name2tsv = {}
        self.index2data = []
        for name in self.names:
            self.name2tsv[name] = TSVFile(catalog[name]["tsv_txt"])
            self.index2data.extend([(name,i) for i in range(len(self.name2tsv[name]))])
            print(f'Load {name}(#={len(self.name2tsv[name])}) from {catalog[name]["tsv_txt"]} ...')
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.index2data)

    def __getitem__(self, index):
        name, ith = self.index2data[index]
        zh1, zh2 = self.name2tsv[name].seek(ith)
        zh1 = self.tokenizer(zh1)
        zh2 = self.tokenizer(zh2)       
        return {'zh1':zh1, 'zh2':zh2}
    

class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, names, catalog, preprocess, tokenizer, 
                 need_img=True, txt_from_tsv=False, lang=['zh','en'],
                 poems_mode=False) -> None:
        super().__init__()
        self.names = names.split(',')
        self.name2triplets, self.name2tsv, self.name2tsv_txt = {},{},{}
        self.index2data = []
        self.txt_from_tsv = txt_from_tsv
       # self.poems_mode = poems_mode
        for name in self.names:
            if self.txt_from_tsv==False:
                self.name2triplets[name] = json.load(open(catalog[name]['metadata'], 'r'))
                print(f'Load {name}(#={len(self.name2triplets[name])}) from {catalog[name]["metadata"]} ...')
                self.index2data.extend([(name,i) for i in range(len(self.name2triplets[name]))])
            else:
                self.name2tsv_txt[name] = TSVFile(catalog[name]['tsv_txt'])
                print(f'Load {name}-txt(#={len(self.name2tsv_txt[name])}) from {catalog[name]["tsv_txt"]} ...')
                self.index2data.extend([(name,i) for i in range(len(self.name2tsv_txt[name]))])
            if need_img:
                self.name2tsv[name] = TSVFile(catalog[name]["tsv_root"])
                print(f'Load {name}-images(#={len(self.name2tsv[name])}) from {catalog[name]["tsv_root"]} ...')

        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.need_img = need_img
        self.lang = lang
        
    def __len__(self):
        return len(self.index2data)
    
    def __getitem__(self, index):
        name, index = self.index2data[index]
        if self.txt_from_tsv:
            id = index
            en_, zh_ = self.name2tsv_txt[name].seek(index)
            if 'poem' in name:
                #randomly select en_ or zh_ as zh_
                if en_ != 'empty'  and zh_ != 'empty':
                    if random.random() > 0.5:
                        zh_ = en_
                elif en_ != 'empty' and zh_ == 'empty':
                    zh_ = en_
                elif en_ == 'empty' and zh_ != 'empty':
                    pass
                else:
                    raise ValueError 
        else:
            img_name, id, en_, zh_ = self.name2triplets[name][index]
        if self.tokenizer is not None:
            if 'en' in self.lang:
                en = self.tokenizer['en'](en_)
            else:
                en = self.tokenizer['en'](zh_)
            if 'zh' in self.lang:
                zh = self.tokenizer['zh'](zh_)
            else:
                zh = self.tokenizer['zh'](en_)
        else:
            en, zh = en_, zh_
        if self.need_img:
            row = self.name2tsv[name].seek(id)
            img_id = self.name2tsv[name]._lineidx[id]
            image = img_from_base64(row[-1])
            if self.preprocess is not None:
                image = self.preprocess(image)
            return {'img':image, 'en':en, 'zh':zh, 'index': index, 'img_lineid': img_id}
        else:
            return {'en':en, 'zh':zh}


class BilingualDataset_from_list(torch.utils.data.Dataset):
    def __init__(self, zh, en, tokenizer):
        super().__init__()
        self.zh, self.en = zh, en
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.zh)
    def __getitem__(self, index):
        en = self.tokenizer['en'](self.en[index])
        zh = self.tokenizer['zh'](self.zh[index])        
        return {'en':en, 'zh':zh}

class TripletDataset_from_rawfile(torch.utils.data.Dataset):
    def __init__(self, img_file, zh, en, preprocess, tokenizer, return_dict=False):
        super().__init__()
        self.img_file = img_file
        self.zh, self.en = zh, en
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        assert len(self.img_file)==len(self.zh) and len(self.zh)==len(self.en)
        self.return_dict = return_dict
    def __len__(self):
        return len(self.img_file)
    
    def __getitem__(self, index):
        image = Image.open(self.img_file[index])
        image_arr = np.array(image)
        if image_arr.shape[-1]!=3:
            if len(image_arr.shape)==2:
                image_arr = image_arr[:,:,None]
                image_arr = np.tile(image_arr, [1,1,3]) 
            elif image_arr.shape[-1]==4:
                image_arr = image_arr[:,:,:3] 
        image = Image.fromarray(image_arr)     
        image = self.preprocess(image)
        
        en = self.tokenizer['en'](self.en[index])
        zh = self.tokenizer['zh'](self.zh[index])   
        #print(f'--> en {en} zh{zh}')
        if self.return_dict:
            return {'img':image, 'en':en, 'zh':zh, 'index': index}
        else:    
            return image, en, zh

class ImageCaptionDatasetSLIP(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, transform, augment, tokenizer=None):
        super().__init__(dataset, root, metadata)

        self.transform = transform
        self.augment = augment
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        img, caption = self.get_raw_item(i)

        image = self.transform(img)
        aug1 = self.augment(img)
        aug2 = self.augment(img)

        # tokenize caption
        if self.tokenizer is not None:
            caption = self.tokenizer(caption)

        return image, caption, aug1, aug2


class ImageCaptionDatasetSSL(ImageCaptionDatasetBase):
    def __init__(self, dataset, root, metadata, augment):
        super().__init__(dataset, root, metadata)

        self.augment = augment

    def __getitem__(self, i):
        img, _ = self.get_raw_item(i)

        aug1 = self.augment(img)
        aug2 = self.augment(img)

        return aug1, aug2


class FileListDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.images = np.load(images)
        self.labels = np.load(labels)

    def __getitem__(self, index):
        img = pil_loader(self.images[index])
        target = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)


def get_downstream_dataset(catalog, name, is_train, transform):
    entry = catalog[name]
    root = entry['path']
    if entry['type'] == 'imagefolder':
        dataset = t_datasets.ImageFolder(os.path.join(root, entry['train'] if is_train else entry['test']),
            transform=transform)
    elif entry['type'] == 'special':
        if name == 'cifar10':
            dataset = t_datasets.CIFAR10(root, train=is_train,
                transform=transform, download=True)
        elif name == 'cifar100':
            dataset = t_datasets.CIFAR100(root, train=is_train,
                transform=transform, download=True)
        elif name == 'stl10':
            dataset = t_datasets.STL10(root, split='train' if is_train else 'test',
                transform=transform, download=True)
        elif name == 'mnist':
            dataset = t_datasets.MNIST(root, train=is_train,
                transform=transform, download=True)
    elif entry['type'] == 'filelist':
        path = entry['train'] if is_train else entry['test']
        val_images = os.path.join(root, path + '_images.npy')
        val_labels = os.path.join(root, path + '_labels.npy')
        if name == 'clevr_counts':
            target_transform = lambda x: ['count_10', 'count_3', 'count_4', 'count_5', 'count_6', 'count_7', 'count_8', 'count_9'].index(x)
        else:
            target_transform = None
        dataset = FileListDataset(val_images, val_labels, transform, target_transform)
    else:
        raise Exception('Unknown dataset')

    return dataset


def get_dataset(train_transform, tokenizer, dataset_names, args, 
                need_only_text=True, read_tsv=True, lang=['en','zh'],
                is_am=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augment = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([utils.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    if args.model.startswith('SIMCLR'):
        return ImageCaptionDatasetSSL(args.dataset, args.root, args.metadata, augment)
    elif args.model.startswith('CLIP'):
        return ImageCaptionDatasetCLIP(args.dataset, args.root, args.metadata, train_transform, tokenizer)
    elif args.model.startswith('SLIP'):
        return ImageCaptionDatasetSLIP(args.dataset, args.root, args.metadata, train_transform, augment, tokenizer)
    elif args.model.startswith('TRIPLET') or args.model.startswith('ALTCLIP') or args.model.startswith('MCLIP'):
        need_img = not need_only_text #(not args.need_only_text)
        catalog = json.load(open('dataset_catalog.json', 'r'))
        if read_tsv:
            if need_img==False:
                if is_am:
                    return ParallelDataset_from_TSV(names=dataset_names, catalog=catalog, tokenizer=tokenizer)
                else:
                    return BilingualDataset_from_TSV(names=dataset_names, catalog=catalog, tokenizer=tokenizer)
            else:
                return TripletDataset(names=dataset_names, catalog=catalog,
                        preprocess=train_transform, tokenizer=tokenizer,
                        need_img=True, txt_from_tsv=True, lang=lang)
        if args.toolkit_data=='torch':
            return TripletDataset(names=dataset_names, catalog=catalog, preprocess=train_transform, 
                              tokenizer=tokenizer, need_img=need_img) # a dict
        elif args.toolkit_data=='bm' and need_img==False:
            return TextDistillDataset(names=dataset_names, catalog=catalog)
        else:
            raise ValueError