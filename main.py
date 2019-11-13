'''
Run this file to train, valid and test model
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, re
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from tqdm import trange

from gensim.parsing import remove_stopwords

from model import * # import all model object
from config_writer import write_config


class F1():
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
        self.n_corrects += torch.sum(groundTruth.type(torch.bool) * predicts).data.item()

    def get_score(self):
        recall = self.n_corrects / self.n_recall
        precision = self.n_corrects / (self.n_precision + 1e-20)
        return 2 * (recall * precision) / (recall + precision + 1e-20)

    def print_score(self):
        score = self.get_score()
        return '{:.5f}'.format(score)

if __name__ == '__main__':
    CWD = os.getcwd()
    DATA_PATH = os.path.join(CWD, 'data')
    CPUNUM = os.cpu_count() // 2