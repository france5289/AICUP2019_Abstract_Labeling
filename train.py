import json
import multiprocessing
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange


from model import F1, GRUNet
from tokenizer import NLTKTokenizer, RegTokenizer

# ============ uncomment the following line to use VSCode python debugger! ================
# multiprocessing.set_start_method('spawn', True) 



# set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data path and cpu num
CWD = os.getcwd()
TRAIN_DATA_PATH = os.path.join(CWD, 'data', 'trainset.csv')
VALID_DATA_PATH = os.path.join(CWD, 'data', 'validset.csv')
TEST_DATA_PATH = os.path.join(CWD, 'data', 'testset.csv')
DICT_PATH = os.path.join(CWD, 'data', 'dictionary.pkl')


WORKERS = os.cpu_count() 

if __name__ == "__main__":
    pass
