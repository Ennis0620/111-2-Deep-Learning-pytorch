import torch
import torch.nn as nn  
#from torchsummary import summary
from torchinfo import summary

class RNN_IMDB_EMBEDDING(nn.Module):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 embedding_dim,
                 seq_len,
                 num_classes):
        super(RNN_IMDB, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Embedding(vocab_size, embedding_dim, padding_idx=seq_len)
        self.rnn = nn.RNN(input_size=embedding_dim, 
                          hidden_size=hidden_size, 
                          num_layers=2, 
                          batch_first=True)
        self.linear = nn.Linear(hidden_size, 
                                num_classes)
    def forward(self,x):
        '''
        x (batch_size, seq_length, input_size)
        '''
        x,_ = self.rnn(x)
        #print(x[:,-1,:].shape)
        x = self.linear(x[:,-1,:])
        return x

class RNN_IMDB(nn.Module):
    def __init__(self,hidden_size,embedding_dim,num_classes):
        super(RNN_IMDB, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=embedding_dim, 
                          hidden_size=hidden_size, 
                          num_layers=2, 
                          batch_first=True)
        self.linear = nn.Linear(hidden_size, 
                                num_classes)
    def forward(self,x):
        '''
        x (batch_size, seq_length, input_size)
        '''
        x,_ = self.rnn(x)
        #print(x[:,-1,:].shape)
        x = self.linear(x[:,-1,:])
        return x

class LSTM_IMDB():
    pass

if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU state:', device)
    seq_len = 300
    hidden_size = 128
    embedding_dim = 100 #glove:100維
    model = RNN_IMDB(hidden_size = hidden_size,
                    embedding_dim=embedding_dim,
                    num_classes=2).to(device)
    print(model)
    summary(model, (1, seq_len, embedding_dim))