import torch
import numpy as np
from tqdm import tqdm

class Recommend:
    def __init__(self, sentence_model, emotion_model, train_data, idx2emoticon):
        self.sen_model = sentence_model
        self.emo_model = emotion_model
        self.train_sen   = train_data[0]
        self.train_emo   = train_data[1]
        self.train_label = train_data[2]
        self.idx2emoticon = idx2emoticon
        self.train_vector = self.convert_sentence_emotion(self.sen_model, self.train_emo, self.train_sen)
        self.train_norm = np.linalg.norm(self.train_vector)
        self.rank_index = None
        self.recommendation = None
        self.input = None
        self.emotion_weight = None

    def convert_sentence_emotion(self, sen_model, emotion, sentence):
        sentence_vector = []
        sentence = sen_model.encode(sentence)
        for idx in range(len(sentence)):
            emotion_label = emotion[idx] #0,1,2,3
            distance = np.sqrt(np.mean(sentence[idx]**2))
            sentence[idx] /= distance
            tmp_vector = np.array([0, 0, 0, 0])
            tmp_vector = np.append(tmp_vector, sentence[idx])
            tmp_vector[emotion_label] = 1
            sentence_vector.append(tmp_vector)

        return np.array(sentence_vector)
    
    def infer(self, x):
        x_tokens = self.emo_model.tokenizer(x, return_tensors='pt')
        x_tokens = x_tokens.to('cuda')
        return torch.softmax(self.emo_model(**x_tokens).logits, dim=-1)

    def add_emotion(self, emotion, sentence, weight):
        sentence = self.sen_model.encode(sentence)
        distance = np.sqrt(np.sum(sentence**2))
        sentence /= distance

        emo_distance = np.sqrt(np.sum(emotion**2))
        emo_added = np.append(weight*emotion/emo_distance, sentence)
        
        return emo_added

    def recommend(self, input_sentence, emotion_weight = 1, threshold = 2, recommend_num = 5, view = True): #threshold must be located 1 to 5
        sentence_emotion = self.infer(input_sentence).cpu().detach().numpy() #emotion analysis
        input_vector = self.add_emotion(sentence_emotion, input_sentence, emotion_weight).reshape(-1,1)
        input_norm = np.linalg.norm(input_vector)
        cos_sim = np.dot(self.train_vector, input_vector).squeeze()
        self.rank_index = np.argsort(cos_sim)[::-1]

        recommendation = []
        selected = {idx : 0 for idx, _ in self.idx2emoticon.items()}
        for r in self.rank_index:
            emoticon = self.train_label[r]
            selected[emoticon] += 1
            if selected[emoticon] == threshold:
                recommendation.append(emoticon)
        self.recommendation = recommendation

        if view:
            for i in range(recommend_num):
                print('Rank{} recommended emoticon: {}'.format(i+1, self.idx2emoticon[recommendation[i]]))
            
    def sentence_analysis(self, input_sentence, emotion_weight = 1, top_num = 5): #input sentences => [List,...]
        self.recommend(input_sentence = input_sentence, emotion_weight = emotion_weight, view = False)
        candidate = []
        for i in range(top_num):
            tmp = self.sen_model.encode(self.train_sen[self.rank_index[i]])
            mean = np.mean(tmp)
            std = np.std(tmp)
            candidate.append((tmp - mean) / std)
        analysis = np.mean(np.array(candidate),axis = 0).reshape(-1,1)
        input_vec = self.sen_model.encode(input_sentence).reshape(1,-1)

        return np.dot(input_vec,analysis).item()
    
    def weight_analysis(self, input_sentence, recommend_num = 5, srt_weight = 0, dst_weight = 100, xlim = 1, vector_top = 5, alpha = 1, beta = 1): 
        # xlim must be positive integer alpha and beta is weight of sen metric and emotion metric must be positive float value
        sen_metric, emo_metric = list(), list()
        for w in tqdm(range(srt_weight, dst_weight, xlim),desc = "Analysing..."):
            sen_metric.append(self.sentence_analysis(input_sentence = input_sentence, emotion_weight = w, top_num = vector_top))

            input_emo = np.argmax(self.infer(input_sentence).cpu().detach().numpy())
            emo_sim = []
            for i in range(recommend_num):
                emo_sim.append(self.train_emo[np.where(self.train_label == self.recommendation[i])[0][0]])
            emo_metric.append(len(np.array(emo_sim)[np.array(emo_sim) == input_emo]) / recommend_num)

        sen_metric = np.array(sen_metric)
        sen_metric /= sen_metric[0]
        emo_metric = np.array(emo_metric)
        return (emo_metric * alpha + sen_metric * beta) / (alpha + beta)

