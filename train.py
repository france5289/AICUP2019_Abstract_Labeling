import pandas as pd
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from tqdm import trange

from model import Net1, F1
from DataPreprocessor import Download_Glove, Create_Glove_embedding_matrix

# set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data path and cpu num
CWD = os.getcwd()
TRAIN_DATA_PATH = os.path.join(CWD, 'data', 'trainset.csv')
VALID_DATA_PATH = os.path.join(CWD, 'data', 'validset.csv')
TEST_DATA_PATH = os.path.join(CWD, 'data', 'testset.csv')
WORKERS = os.cpu_count() // 2


if __name__ == '__main__':
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(VALID_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        os.system('python3 gendata.py')

    
