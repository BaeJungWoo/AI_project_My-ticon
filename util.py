from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead, AdamW, AutoModel
import re
import emoji
from soynlp.normalizer import repeat_normalize
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import argparse


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

class Preprocessing:
    def __init__(self, max_len, padding = 'left'):
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-large")
        self.sentence = list()
        self.label = list()
        self.max_len = max_len
        self.label_dic = {"공포":0,"혐오":1,"행복":2,"놀람":3,"분노":4,"슬픔":5,"중립":6}
        self.padding = padding

        self.load_data()

    def load_data(self):
        data = pd.read_csv("./data.csv",encoding='cp949')
        for idx in range(len(data)):
            sen, lbl = data.iloc[idx,:]
            tok = self.tokenizer(clean(sen))['input_ids']
            if len(tok) >= self.max_len:
                self.sentence.append(tok[:self.max_len])
            else:
                if self.padding == 'left':
                    self.sentence.append([0]*(self.max_len - len(tok)) + tok)
                else:
                    self.sentence.append(tok + [0]*(self.max_len - len(tok)))
        self.label.append(self.label_dic[lbl])
        self.sentence = np.array(self.sentence)
        self.label = np.array(self.label).reshape(-1,1)

    def split_data(self, val_set = False):
        X_train, X_test, y_train, y_test = train_test_split(self.sentence, self.label, test_size=0.2)
        
        if val_set:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
            return X_train, X_val, y_train, y_val
        else:
            return X_train, X_test, y_train, y_test

