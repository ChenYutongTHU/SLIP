import argparse, os, json, math
# from tqdm import tqdm
import multiprocessing, torch
import numpy as np
#from model_center.tools import indexed_dataset
import indexed_dataset, sys
sys.path.append('/home/chenyutong/code/SLIP')
import clip
from wukong import wukong_tokenizer

# 1. Implement an Encoder, which gives it a line of input data and it returns you the tokenized result.
class Encoder_en(object): 
    def initializer(self):
        Encoder_en.tokenize = clip.tokenize
    def encode(self, line):
        data = line.strip() 
        doc_ids = Encoder_en.tokenize(data)
        return [doc_ids]

class Encoder_zh(object): 
    def initializer(self):
        Encoder_zh.tokenize = wukong_tokenizer('zh')
    def encode(self, line):
        data = line.strip() 
        doc_ids = Encoder_zh.tokenize(data).int()
        return [doc_ids]

    # def encode(self, line):
    #     data = line.strip().replace("<n>", "\n") # replace back line break symbol

    #     doc_ids = self.tokenize(data)

    #     max_length = 512 # model's maximum input length

    #     pieces = []
    #     while i < len(doc_ids): # split document into chunks with model's maximum length
    #         piece = doc_ids[i:i+max_length]
    #         if len(piece) < 32: # drop too short chunks
    #             break
    #         i += max_length

    #         pieces.append(piece)

        # return pieces

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Split trio.json -> .txt')
    parser.add_argument('--chunk_number', type=int, default=1)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()
    if args.output_dir==None:
        args.output_dir = args.input_dir
    # assumes that there are 100 raw data files, named `data_0.txt` to `data_99.txt`
    for lang in ['zh','en']:
        for ith in range(0, args.chunk_number):
            fin = open(os.path.join(args.input_dir,lang,f"data_{ith}.txt"), "r", encoding="utf-8")
            # encoder use the tokenizer to encode data
            
            if lang=='en':
                encoder = Encoder_en()
            else:
                encoder = Encoder_zh()
            # encoder.initializer()
            # for lin in fin.readlines():
            #     print(lin)
            #     ids = encoder.encode(lin)
            #     print(ids, ids[0].dtype)
            #     input()
            #     break
            # break

            # 2. Mapping all datas with Encoder, with the help of multiprocessing
            pool = multiprocessing.Pool(processes=64, initializer=encoder.initializer)
            encoded_docs = pool.imap_unordered(encoder.encode, fin, chunksize=10)

            # 3. tool `indexed_dataset` compress the tokenized data into binary format `bin_file`
            # it will also generate another small `idx_file` for saving meta information in order to decode `bin_file`.
            output_dir = os.path.join(args.output_dir, f'{lang}_tokenized')
            os.makedirs(output_dir, exist_ok=True)
            bin_file = os.path.join(output_dir, f"tokenized_{ith}.bin")
            idx_file = os.path.join(output_dir, f"tokenized_{ith}.idx")

            binary_builder = indexed_dataset.make_builder(bin_file, impl="mmap")

            # put tokenized data into binary_builder
            for pieces in encoded_docs:
                for doc_ids in pieces:
                    binary_builder.add_item(torch.IntTensor(doc_ids))

            # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
            binary_builder.finalize(idx_file)

            # close multiproceessing mapping
            pool.close()

