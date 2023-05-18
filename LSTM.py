from util import *

class LSTM(nn.Module):
  def __init__(self, device, embed_dim, hidden_dim, num_classes = 4):
    super(LSTM, self).__init__()
    self.device = device
    self.embed_dim = embed_dim
    self.hidden_dim = hidden_dim
    self.num_classes = num_classes
    self.vocab_size = 30000

    self.build_model()
    self.to(device)


  def build_model(self):
    self.word_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
    self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_dim, batch_first = True, dropout=0.3)
    self.layer1 = nn.Linear(in_features = self.hidden_dim, out_features = 16)
    self.layer2 = nn.Linear(in_features = 16,  out_features = self.num_classes)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim = 1)
    self.loss_function = nn.CrossEntropyLoss()

  def forward(self, x):
    x = self.word_embedding(x)
    h_0 = torch.zeros(1, x.size(0), self.hidden_dim).to(self.device)
    c_0 = torch.zeros(1, x.size(0), self.hidden_dim).to(self.device)
    x, (h_t,c_t) = self.lstm(x, (h_0,c_0))
    h_t = x[:, -1, :]
    output = self.layer1(h_t)
    output = self.relu(output)
    output = self.layer2(output)
    
    return output.squeeze()

  def train_model(self, x_train, y_train, num_epochs, batch_size, learning_rate):
    self.optimizer = AdamW(self.parameters(), lr=learning_rate)
    loss_log = []
    for e in range(num_epochs):
      epoch_loss = 0
      iter_per_epoch = max(int(len(x_train) / batch_size), 1)
      for batch_idx in range(iter_per_epoch):
        self.optimizer.zero_grad()
        out = x_train[batch_idx*batch_size : (batch_idx+1)*(batch_size),:]
        out = torch.Tensor(out).long().to(self.device)
        out = self.forward(out)
        batch_labels = torch.Tensor(y_train[batch_idx*batch_size:(batch_idx+1)*(batch_size),:].reshape(-1)).long().to(self.device)
        loss = self.loss_function(out, batch_labels)
        epoch_loss += loss.item()
        loss.backward()
        self.optimizer.step()
      loss_log.append(epoch_loss)
      print(f'>> [Epoch {e+1}] Total epoch loss: {epoch_loss:.2f}')

  def predict(self, X, y, batch_size):
    y = np.array(y)
    preds = torch.zeros(len(X)).to(self.device)
    total_loss = 0
    with torch.no_grad():
      iter_per_epoch = max(int(len(X) / batch_size), 1)
      for batch_idx in range(iter_per_epoch):
        out = X[batch_idx*batch_size:(batch_idx+1)*(batch_size),:]
        
        out = torch.Tensor(out).long().to(self.device)
        out = self.forward(out)
        batch_labels = torch.Tensor(y[batch_idx*batch_size:(batch_idx+1)*(batch_size),:].reshape(-1)).long().to(self.device)
        loss = self.loss_function(out, batch_labels)
        total_loss += loss
        max_vals, max_indices = torch.max(out,1)
        preds[batch_idx*batch_size:(batch_idx+1)*(batch_size)] = max_indices.float()
      labels = torch.Tensor(y).to(self.device)
      accuracy = (preds == labels.reshape(-1)).sum().data.cpu().numpy() / y.shape[0]
      
    return accuracy, total_loss