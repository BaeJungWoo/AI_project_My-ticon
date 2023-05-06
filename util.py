from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead, AdamW
import re
import emoji
from soynlp.normalizer import repeat_normalize
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn


def clean(x): 
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    x = pattern.sub(' ', x)
    x = emoji.replace_emoji(x, replace='') #emoji 삭제
    x = url_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)

    return x

data = pd.read_csv("./data.csv",encoding='cp949')
sentence, label = list(), list()
max_len = 15
for idx in range(len(data)):
    sen, lbl = data.iloc[idx,:]
    tok = tokenizer(clean(sen))['input_ids']
    if len(tok) >= max_len:
        sentence.append(tok[:max_len])
    else:
        sentence.append([0]* (max_len - len(tok))+tok)
    label.append(label_dic[lbl])
sentence = np.array(sentence)
label = np.array(label).reshape(-1,1)
print(sentence.shape)
print(label.shape)
