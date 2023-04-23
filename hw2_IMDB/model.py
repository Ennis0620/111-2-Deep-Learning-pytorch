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
    def __init__(self,hidden_size,embedding_dim,num_classes,num_layers):
        super(RNN_IMDB, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=embedding_dim, 
                          hidden_size=hidden_size, 
                          num_layers=num_layers, 
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
    
class RNN2_IMDB(nn.Module):
    def __init__(self,hidden_size,embedding_dim,num_classes):
        super(RNN2_IMDB, self).__init__()
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
        x = self.linear(x[:,-1,:])
        return x

class RNN_IMDB_BCE(nn.Module):
    def __init__(self,hidden_size,embedding_dim,num_classes):
        super(RNN_IMDB_BCE, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=embedding_dim, 
                          hidden_size=hidden_size, 
                          num_layers=2, 
                          batch_first=True)
        self.linear = nn.Linear(hidden_size, 
                                1)
        
    def forward(self,x):
        '''
        x (batch_size, seq_length, input_size)
        '''
        x,_ = self.rnn(x)
        x = self.linear(x[:,-1,:])
        x = torch.sigmoid(x)
        return x

class LSTM_IMDB(nn.Module):
    def __init__(self,hidden_size,embedding_dim,num_classes,num_layers) -> None:
        super(LSTM_IMDB, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            batch_first=True,
                            dropout = 0.3)
        self.linear = nn.Linear(hidden_size, 
                                num_classes)
    def forward(self,x):
        '''
        x (batch_size, seq_length, input_size)
        '''
        x,_ = self.lstm(x)
        #print(x[:,-1,:].shape)
        x = self.linear(x[:,-1,:])
        return x

class LSTM2_IMDB(nn.Module):
    def __init__(self,hidden_size,embedding_dim,num_classes) -> None:
        super(LSTM2_IMDB, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_size, 
                            num_layers=2, 
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 
                                num_classes)
    def forward(self,x):
        '''
        x (batch_size, seq_length, input_size)
        '''
        x,_ = self.lstm(x)
        #print(x[:,-1,:].shape)
        x = self.linear(x[:,-1,:])
        return x

class GRU_IMDB(nn.Module):
    def __init__(self,hidden_size,embedding_dim,num_classes,num_layers) -> None:
        super(GRU_IMDB, self).__init__()
        self.gru = nn.GRU(input_size=embedding_dim, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True,
                            dropout = 0.3)
        self.linear = nn.Linear(hidden_size, 
                                num_classes)
    def forward(self,x):
        '''
        x (batch_size, seq_length, input_size)
        '''
        x,_ = self.gru(x)
        #print(x[:,-1,:].shape)
        x = self.linear(x[:,-1,:])
        return x

class GRU2_IMDB(nn.Module):
    def __init__(self,hidden_size,embedding_dim,num_classes) -> None:
        super(GRU2_IMDB, self).__init__()
        self.gru = nn.GRU(input_size=embedding_dim, 
                            hidden_size=hidden_size, 
                            num_layers=2, 
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 
                                num_classes)
    def forward(self,x):
        '''
        x (batch_size, seq_length, input_size)
        '''
        x,_ = self.gru(x)
        #print(x[:,-1,:].shape)
        x = self.linear(x[:,-1,:])
        return x

if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('GPU state:', device)
    seq_len = 180
    hidden_size = 256
    embedding_dim = 100 #glove:100維
    model = GRU_IMDB(hidden_size = hidden_size,
                    embedding_dim=embedding_dim,
                    num_classes=2,
                    num_layers=2
                    ).to(device)
    print(model)
    summary(model, (1, seq_len, embedding_dim))