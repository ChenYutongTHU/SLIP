import argparse, os, json, math
import sys
from tqdm import tqdm

def output_txt(chunk, filename):
    os.makedirs(os.path.dirname(filename),exist_ok=True)
    with open(filename, 'w') as f:
        for c in chunk:
            f.writelines(c+'\n')
    print('Output ', filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Split trio.json -> .txt')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--chunk_number', type=int, default=1)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    catalog = json.load(open('dataset_catalog.json','r'))
    trio = json.load(open(catalog[args.dataset]['metadata'],'r'))

    en_chunk, zh_chunk = [], []
    i = 0
    args.chunk_size = math.ceil(len(trio)/args.chunk_number)
    for _,_,en,zh in tqdm(trio):
        en_chunk.append(en)
        zh_chunk.append(zh)
        if len(en_chunk)>=args.chunk_size:
            output_txt(en_chunk, filename=os.path.join(args.output_dir,'en',f'data_{i}.txt'))
            en_chunk = []
            output_txt(zh_chunk, filename=os.path.join(args.output_dir,'zh',f'data_{i}.txt'))
            zh_chunk = []
            i+=1

    if len(en_chunk):
        output_txt(en_chunk, filename=os.path.join(args.output_dir,'en',f'data_{i}.txt'))
        en_chunk = []
        output_txt(zh_chunk, filename=os.path.join(args.output_dir,'zh',f'data_{i}.txt'))
        zh_chunk = []        



