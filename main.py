import os
import pandas as pd
import numpy as np
import torch
import sys
from pprint import pprint
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_lightning import LightningModule, Trainer, seed_everything
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import emoji
from soynlp.normalizer import repeat_normalize
import transformers
import emoji
import soynlp
import pytorch_lightning
from utils.emotion_model import Model
from utils.infer_emotion import infer
from utils.load_data import load_dataset, idx2emoticon, emoticon2idx, convert_sentence_emotion, add_emotion
from sentence_transformers import SentenceTransformer, util
from sklearn.datasets import make_classification
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from Recommend import *


def recommend(knn, input_sentence, num_recommend,weight_emotion = 1):
    sentence_emotion = infer(emotion_model, input_sentence)
    sentence_emotion = sentence_emotion.cpu().detach().numpy()

    sentence_vector = add_emotion(sentence_model, sentence_emotion * weight_emotion, input_sentence)
    print(sentence_emotion)
    sentence_emotion = np.argmax(sentence_emotion)
    recommend_num = num_recommend
    selected = {idx : i for idx, i in enumerate(knn.predict_proba([sentence_vector]).reshape(-1)) if i!=0}
    sorted_dict = sorted(selected.items(), key = lambda item: item[1], reverse = True)
    loss = 0
    for idx, i in enumerate(sorted_dict):
        if idx >= recommend_num : break
        print('Rank{} recommended emoticon: {}'.format(idx+1, idx2emoticon[int(i[0])]))
        vote = np.array([0,0,0,0])
        for v in [train_emotions[j] for j in np.where(train_labels == i[0])[0]]:
            vote[v] +=1
        record = np.argmax(vote)
        if record != sentence_emotion : loss += 1

    if idx < recommend_num:
        print("Please train model with larger neighbors")
    else:
        print("Accuracy: ", (recommend_num - loss) / recommend_num)


if __name__=="__main__":
    emotion_model_path = '/content/drive/MyDrive/epoch9-val_acc0.7202.ckpt'
    sentence_model_path = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'
    emoticon_sentence_path = '/content/dataset/sentence_emoticon.txt'
    idx2emoticon_path = '/content/dataset/idx2emoticon.txt'
    emoticon2idx_path = '/content/dataset/emoticon2idx.txt'
    idx2emoticon = idx2emoticon(idx2emoticon_path)
    emoticon2idx = emoticon2idx(emoticon2idx_path)
    emotion_model = Model.load_from_checkpoint(emotion_model_path)
    sentence_model = SentenceTransformer(sentence_model_path)

    train_sentences, train_emotions, train_labels = load_dataset(emoticon_sentence_path)
    train = (train_sentences, train_emotions, train_labels)
    model = Recommend(sentence_model, emotion_model ,train, idx2emoticon)
    model.recommend(input_sentence=['귀여운 강아지'], emotion_weight=1, threshold=3, recommend_num=10)

    #for Analysis bw Emotion weight and Sentence embedding Similarity
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    value = []
    x = np.arange(1,101)
    for i in tqdm(range(1,101)):
        value.append(model.sentence_analysis(['귀여운 강아지'],i,5))
    plt.plot(x,np.array(value),marker = 'o')
    plt.show()

    
    # train_sentences, train_emotions, train_labels = load_dataset(emoticon_sentence_path)
    # train_sentences = convert_sentence_emotion(sentence_model, train_emotions, train_sentences)
    # knn = KNeighborsClassifier(n_neighbors=10, algorithm='brute', metric='cosine')
    # knn.fit(train_sentences, train_labels)

    #label_dic = {"슬픔":0, "당황":1, "기쁨": 2, "분노":3}
    # input_sentence = ['오늘 기분 진짜 개짜증나네!']
    # input_sentence = None
    # while True:
    #     print("Enter Input Sentence : ",end="")
    #     input_sentence = sys.stdin.readline()
    #     print(input_sentence)
    #     if input_sentence == "-1\n":
    #         print("Bye!")
    #         break
    #     recommend(knn = knn, input_sentence=[str(input_sentence)], num_recommend = 7, weight_emotion=768/4)
