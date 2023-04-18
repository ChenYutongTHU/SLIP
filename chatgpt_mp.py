import requests
import json
import argparse, os, pickle
from multiprocessing import Pool
from tqdm import tqdm
from collections import defaultdict


def check(txt, number=5):
    txt = txt.replace('。','')
    cnt = 0
    for s in txt.split(','):
        cnt += len(s)-number
    return cnt
        
def chat_api_request(context):
    response = requests.post("http://freegpt.club/sls01", json={
        "model": "gpt-3.5-turbo",
        "messages": context,
        "max_tokens": 1024,
        "presence_penalty": 0,
        "frequency_penalty": 0
    })
    json = response.json()
    return json['choices'][0]['message']['content'].strip()
def process_func(i):
    maxT = args.max_T
    number = args.word_number
    number2 = '十' if number=='五' else '十四'
    prompt = f'请根据输入现代文创作一句{number}言诗。例如：'
    number_int = 5 if number=='五' else 7
    if number=='五':
        demo = [
        ['白亮的月光下有一张床。 ','There is a bed under the moonlight.','床前明月光，疑是地上霜。'],
        ['沙漠中有烟气垂直升起，河流之上太阳正在下山。','A column of smoke is rising in the desert and a sun is setting above the river.', '大漠孤烟直，长河落日圆。']
        ]
    elif number=='七':
        demo = [
            ['夜空中有云和月亮，非常安静',None,'暮云收尽溢清寒，银汉无声转玉盘。'],
            ['在竹林之外有几只桃花，江水中有几只鸭子',None,'竹外桃花三两枝，春江水暖鸭先知。']
        ]
    for z, e, o in demo:
        prompt += f'输入："{z}" 输出："{o}" '
    prompt += f'输入：{texts[i]} '
    prompt += '输出：{text}'
    prompt += '（输出须由两句组成，每句各{}个字，总共{}个字）'.format(number,number2)
    # print('Prompt:', prompt)
    results = None
    for t in range(maxT):
        try:
            result = chat_api_request([{"role": "user", "content": prompt}])
        except:
            result = 'Error'
        if result!='Error':
            n = check(result, number_int)
            if n==0:
                results = result
                break
            else:
                if results==None or n < results[1]:
                    results = [result, n]
    if type(results) == list:
        results = results[0]
    return i, results

args = argparse.ArgumentParser()
args.add_argument("--input", type=str, required=True)
args.add_argument("--output", type=str, required=True)
args.add_argument("--process_number", type=int, required=True)
args.add_argument("--max_T", type=int, required=True)
args.add_argument("--word_number", type=str, required=True)
args.add_argument("--save_frequency", type=int, required=True)

args = args.parse_args()
#make outputdir of args.output 
os.makedirs(os.path.dirname(args.output), exist_ok=True)
#load the pickle file args.output
if os.path.exists(args.input):
    with open(args.input, 'rb') as f:
        texts = pickle.load(f)
else:
    raise ValueError('input file not exist')

# texts = texts[:10]
 
id2results = {}
import math
save_total = math.ceil(len(texts)/args.save_frequency)
save_cnt = 0
with Pool(args.process_number) as p:
    #func = lambda x:process_func(x,max_T=args.max_T, number=args.word_number)
    for i, result in tqdm(p.imap_unordered(process_func, range(len(texts))), total=len(texts)):
        print(i, texts[i], result)
        id2results[i] = result
        if len(id2results)%args.save_frequency==0:
            print('save', len(id2results), 'results')
            with open(args.output+f'.{save_cnt}_{save_total}', 'wb') as f:
                pickle.dump(id2results, f)
            save_cnt += 1
            id2results = {}
    if len(id2results) > 0:
        print('save', len(id2results), 'results')
        with open(args.output+f'.{save_cnt}_{save_total}', 'wb') as f:
            pickle.dump(id2results, f)
        save_cnt += 1
with open(args.output, 'wb') as f:
    pickle.dump(id2results, f)

# print(id2results)
# prompt = '请根据输入现代文创作古诗句。例如：输入：无边的树木向地面脱落，长江的水势滔滔不绝地向前流淌 。输出：无边落木萧萧下，不尽长江滚滚来。 输入：明亮的月光洒在床前的窗户纸上，好像地上泛起了一层霜。输出：床前明月光，疑是地上霜。 输入：在灿烂的阳光照耀下，西湖水微波粼粼，波光艳丽，看起来很美；雨天时，在雨幕的笼罩下，西湖周围的群山迷迷茫茫，若有若无，也显得非常奇妙。 输出：水光潋滟晴方好，山色空蒙雨亦奇。 输入：{} 输出：'



# if check(result1, number):
#     print('OK')
# else:
#     prompt2 = f'请将"{result1}"改成由两句组成的诗，每句各{number}个字，总共{number2}个字。'
#     #'生成的诗必须由两句组成，每句各{number}个字，总共{number2}个字。输入：'+i+' 输出：'
#     print('Prompt:', prompt2)
#     result2 = chat_api_request([{
#         "role": "user", "content": prompt,
#         "role": "system", "content": result1,
#         "role": "user", "content": prompt2}])
#     print('Response:', result2)