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
                                    ('Sigmoid'), nn.Sigmoid()
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


class F1():
    '''
    This object provide some method to evaluate F1 score 
    '''
    def __init__(self):
        self.threshold = 0.5
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0
        self.name = 'F1'

    def reset(self):
        self.n_precision = 0
        self.n_recall = 0
        self.n_corrects = 0

    def update(self, predicts, groundTruth):
        predicts = predicts > self.threshold
        self.n_precision += torch.sum(predicts).data.item()
        self.n_recall += torch.sum(groundTruth).data.item()
        self.n_corrects += torch.sum(groundTruth.type(torch.bool) *predicts).data.item()

    def get_score(self):
        recall = self.n_corrects / self.n_recall
        precision = self.n_corrects / (self.n_precision + 1e-20)
        return 2 * (recall * precision) / (recall + precision + 1e-20)

    def print_score(self):
        score = self.get_score()
        return '{:.5f}'.format(score)
