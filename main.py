from util import *
from RNN import RNN

if __name__=="__main__":
    data = pd.read_csv("./data.csv",encoding='cp949')
    label_dic = {"공포":0,"혐오":1,"행복":2,"놀람":3,"분노":4,"슬픔":5,"중립":6}
    tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-large")
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
    X_train, X_test, y_train, y_test = train_test_split(sentence, label, test_size=0.3)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RNN(device=device, embed_dim = 15, hidden_dim = 64)
    model.train_model(X_train, y_train, 10, 100, 0.005)
    acc, tot_loss = model.predict(X_test, y_test, 100)
    print(acc)
    print(tot_loss)
