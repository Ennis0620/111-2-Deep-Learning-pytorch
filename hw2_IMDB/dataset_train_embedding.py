import torch 
from torch.utils.data import Dataset
import numpy as np
import os 
import pandas as pd
import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class dataset_embedding(Dataset):
    def __init__(self,
                 data
                 ):
        super(dataset_embedding,self).__init__()

        self.CURRENT_PATH = os.path.dirname(__file__)
        self.data = data
        #glove
        self.data_length = len(data[0])
        
    def __getitem__(self, index):
        #glove
        review = self.data[0][index]
        sentiment = self.data[1][index]

        return review,torch.from_numpy(np.asarray(sentiment))
    
    def __len__(self):
        return self.data_length


    