import pandas as pd
import os
import pickle

import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from tqdm import trange

from model import Net1, F1
from DataPreprocessor import Download_Glove, Create_Glove_embedding_matrix
from tokenizer import NLTKTokenizer

# set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data path and cpu num
CWD = os.getcwd()
TRAIN_DATA_PATH = os.path.join(CWD, 'data', 'trainset.csv')
VALID_DATA_PATH = os.path.join(CWD, 'data', 'validset.csv')
TEST_DATA_PATH = os.path.join(CWD, 'data', 'testset.csv')
DICT_PATH = os.path.join(CWD, 'data', 'dictionary.pkl')

WORKERS = os.cpu_count() // 2
# default Tokenizer
Tokenizer = NLTKTokenizer()


def SplitSent(doc):
    return doc.split('$$$')


def GenDict(train, valid):
    global Tokenizer
    if os.path.exists(DICT_PATH):
        Tokenizer = NLTKTokenizer.load_from_file(DICT_PATH)
    else:
        for item in train['Abstract']:
            Tokenizer.build_dict(item)

        for item in valid['Abstract']:
            Tokenizer.build_dict(item)
        Tokenizer.save_to_file(DICT_PATH)


if __name__ == '__main__':
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(VALID_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        os.system('python3 gendata.py')

    train = pd.read_csv(TRAIN_DATA_PATH)
    valid = pd.read_csv(VALID_DATA_PATH)
    test = pd.read_csv(TEST_DATA_PATH)

    train['Abstract'] = train['Abstract'].apply(func=SplitSent)
    valid['Abstract'] = valid['Abstract'].apply(func=SplitSent)
    GenDict(train, valid)