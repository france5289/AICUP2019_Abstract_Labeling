'''
Every model definition should implement in here
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# TODO: residual connections
class Net(nn.Module):
    def __init__(self, config, vocab_size, embedding_matrix):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.embedding.weight = torch.nn.Parameter(embedding_matrix)
        self.sent_rnn = nn.GRU( self.embedding_dim,
                                self.hidden_dim,
                                bidirectional=True,
                                batch_first=True)
        self.FCLayers = nn.Sequential(
                            OrderedDict(
                                [
                                    ('FC1', nn.Linear(self.hidden_dim*2, self.hidden_dim)),
                                    ('ReLU', nn.ReLU()),
                                    ('FC2', nn.Linear(self.hidden_dim, 6)),
                                    ('Sigmoid', nn.Sigmoid())
                                ]
                            )
        )

        torch.nn.init.xavier_normal_(self.FCLayers[0].weight)
        torch.nn.init.xavier_normal_(self.FCLayers[2].weight)

    def forward(self, x):
        x = self.embedding(x)
        b,s,w,e = x.shape 
        x = x.view(b, s*w, e)
        x, _ = self.sent_rnn(x)
        x = x.view(b,s,w,-1)
        x = torch.max(x, dim=2)[0]
        y = self.FCLayers(x)
        return y


