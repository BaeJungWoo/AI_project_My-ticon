import pandas as pd
import numpy as np

def load_dataset(path):

    df = pd.read_csv(path, sep='\t')

    train_sentence = df['sentence']
    train_emotion = df['emotion']
    train_label = df['label']

    return train_sentence.to_numpy(), train_emotion.to_numpy(), train_label.to_numpy()


def idx2emoticon(path):
    newdic = dict()
    with open(path,'r') as f:
        while True:
            line = f.readline()
            if not line : break
            newdic[int(line.split(':')[0])] = line.split(':')[1][:-1]
    return newdic


def emoticon2idx(path):
    newdic = dict()
    with open(path,'r') as f:
        while True:
            line = f.readline()
            if not line : break
            newdic[line.split(':')[0]] = line.split(':')[1][:-1]
    return newdic


def convert_sentence_emotion(model, emotion, sentence):

    sentence_vector = []

    sentence = model.encode(sentence)

    for idx in range(len(sentence)):

        emotion_label = emotion[idx]
        distance = np.sqrt(np.sum(sentence[idx]**2))
        sentence /= distance

        tmp_vector = np.array([0, 0, 0, 0])
        tmp_vector = np.append(tmp_vector, sentence[idx])
        tmp_vector[emotion_label] = 1
        sentence_vector.append(tmp_vector)

    return sentence_vector


def add_emotion(model, weight, emotion, sentence):

    sentence = model.encode(sentence)

    distance = np.sqrt(np.sum(sentence**2))
    sentence /= distance

    emo_distance = np.sqrt(np.sum(emotion**2))
    emo_added = np.append(weight*emotion/emo_distance, sentence)

    return emo_added