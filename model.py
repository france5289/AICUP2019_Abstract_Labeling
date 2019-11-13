'''
Every model definition should implement in here
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Net1(nn.Module):
    ''' 
    Args:
        vocabulary_size(int) : size of vocabulary
        embedding_dim(int) : dimentionality of embedding layer
        hidden_dim(int) : dimentionality of hidden layer
        embedding_matrix(Tensor) : pretrained word vector matrix
        layer_num(int) : depth of GRU layer
        drop_pb(float) : drop out probability
        bidirect(bool) : bidirectional GRU or not 
    
    Network Architecture:
        Embedding Layer( vocab_size, embedding_dim, embedding_matrix ) \n
        GRU Layer( embedding_dim, hidden_dim, layer_num, drop_pb, bidirect) \n
        Linear1(hiddem_dim*2, hiddem_dim) with xavier_normal \n
        Dropout( drop_pb ) \n
        ReLU() \n
        Linear( hidden_dim, 6 ) \n
        Sigmoid() \n   
    '''
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, embedding_matrix, layer_num=1, drop_pb=0.5, bidirect=False):
        super(Net1, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, _weight=embedding_matrix)
        self.sent_rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=layer_num, dropout=drop_pb, bidirectional=bidirect, batch_first=True)
        self.FCLayer = nn.Sequential(OrderedDict([
            ('FC1', nn.Linear(hidden_dim*2, hidden_dim)),
            ('DropOut1', nn.Dropout(drop_pb)),
            ('ReLU1', nn.ReLU()),
            ('FC2', nn.Linear(hidden_dim, 6)),
            ('Sigmoid', nn.Sigmoid())
        ]))
        torch.nn.init.xavier_normal_(self.FCLayer[0].weight)
        
    def forward(self, x):
        '''
        Args:
            x(Tensor) : input tensor
        Return:
            y(Tensor) : output tensor
        '''
        # b: batch_size
        # s: number of sentences
        # w: number of words
        # e: embedding_dim
        x = self.embedding(x)
        b,s,w,e = x.shape
        x = x.view(b,s*w,e)
        x, __ = self.sent_rnn(x)
        x = x.view(b,s,w,-1)
        x = torch.max(x,dim=2)[0]
        y = self.FCLayer(x)
        return y
