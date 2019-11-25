'''
Every model definition should implement in here
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class GRUNet(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 embedding_matrix,
                 hidden_dim,
                 layer_num=1,
                 drop_pb=0.5,
                 bidirect=False):
        super(GRUNet, self).__init__()
        GRU_drop_pb = drop_pb
        if layer_num == 1:
            GRU_drop_pb = 0
        # TODO : use Glove pre trained word embedding to init embedding layet weight
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = torch.nn.Parameter(embedding_matrix)
        self.sent_rnn = nn.GRU(embedding_dim,
                               hidden_dim,
                               num_layers=layer_num,
                               dropout=GRU_drop_pb,
                               bidirectional=bidirect,
                               batch_first=True)
        self.FCLayer = nn.Sequential(
            OrderedDict([('FC1', nn.Linear(hidden_dim * 2, hidden_dim)),
                         ('DropOut1', nn.Dropout(drop_pb)),
                         ('LayerNorm2', nn.LayerNorm(hidden_dim)),
                         ('ReLU1', nn.ReLU()),
                         ('FC2', nn.Linear(hidden_dim, hidden_dim // 2)),
                         ('DropOut2', nn.Dropout(drop_pb)),
                         ('LayerNorm3', nn.LayerNorm(hidden_dim // 2)),
                         ('ReLU2', nn.ReLU()),
                         ('FC3', nn.Linear(hidden_dim // 2, 6)),
                         ('Sigmoid', nn.Sigmoid())]))
        self.layernorm1 = nn.LayerNorm(hidden_dim * 2)
        torch.nn.init.xavier_normal_(self.FCLayer[0].weight)
        torch.nn.init.xavier_normal_(self.FCLayer[4].weight)
        torch.nn.init.xavier_normal_(self.FCLayer[8].weight)

    def forward(self, x, eos_indices):
        '''
        Args:
            x(Tensor): input tensor with shape b*s
            eos_indexes(list): list which record positions of every eos tokens
        '''
        x = self.embedding(x)
        x, _ = self.sent_rnn(x)
        b, s, h = x.size()
        x = x.contiguous().view(-1, h)  # (b*s)*(hidden_dim * direction_num)
        x = torch.index_select(x, 0, eos_indices)
        x = self.layernorm1(x)
        y = self.FCLayer(x)
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
