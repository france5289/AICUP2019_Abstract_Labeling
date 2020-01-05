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

from config import Config
from DataPreprocessor import (  Create_Vocabulary, Get_dataset,
                                Get_Pretrained_embedding_matrix, Load_Vocabulary)
from model import F1, Net

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
HPARAMS_PATH = os.path.join(CWD, 'hyperparameters.json')

WORKERS = os.cpu_count() 

PAD_IDX = 0
UNK_IDX = 1

# ======= Custom Dataset Definition ==============================================================================
class AbstractDataset(Dataset):
    def __init__(self, data, pad_idx, max_len = 64):
        self.data = data
        self.pad_idx = pad_idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
        
    def collate_fn(self, datas):
        # get max length in this batch
        max_sent = max([len(data['Abstract']) for data in datas])
        max_len = max([min(len(sentence), self.max_len) for data in datas for sentence in data['Abstract']])
        batch_abstract = []
        batch_label = []
        sent_len = []
        for data in datas:
            # padding abstract to make them in same length
            pad_abstract = []
            for sentence in data['Abstract']:
                if len(sentence) > max_len:
                    pad_abstract.append(sentence[:max_len])
                else:
                    pad_abstract.append(sentence+[self.pad_idx]*(max_len-len(sentence)))
            sent_len.append(len(pad_abstract))
            pad_abstract.extend([[self.pad_idx]*max_len]*(max_sent-len(pad_abstract)))
            batch_abstract.append(pad_abstract)
            # gather labels
            if 'Label' in data:
                pad_label = data['Label']
                pad_label.extend([[0]*6]*(max_sent-len(pad_label)))
                
                batch_label.append(pad_label)
        return torch.LongTensor(batch_abstract), torch.FloatTensor(batch_label), sent_len
# ================================================================================================================










if __name__ == "__main__":
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(VALID_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        os.system('python3 gendata.py')
    # ======== Hyperparameters settings ========
    myconfig = Config.load_from_json(HPARAMS_PATH)
    # ==========================================

    # ======== Create Vocabulary from training set ========
    myvocab = Create_Vocabulary(TEST_DATA_PATH, pad_idx=PAD_IDX, unk_idx=UNK_IDX ,workers=WORKERS)
    # =====================================================

    # ======== Get Pretrained embedding matrix ========
    embedding_matrix = torch.FloatTensor(Get_Pretrained_embedding_matrix(myconfig.pretrained_embedding_path, myvocab, myconfig.embedding_dim))
    # =================================================

    # ======== Preprocess training set, validation set and testing set =======
    print('[INFO] Start processing trainset...')
    train = Get_dataset(TRAIN_DATA_PATH, myvocab, WORKERS)
    print('[INFO] Start processing validset...')
    valid = Get_dataset(VALID_DATA_PATH, myvocab, WORKERS)
    print('[INFO] Start processing testset...')
    test = Get_dataset(TEST_DATA_PATH, myvocab, WORKERS)
    # ========================================================================

    # ======== construct custom dataset ===================================
    trainData = AbstractDataset(train, PAD_IDX, max_len=64)
    validData = AbstractDataset(valid, PAD_IDX, max_len=64)
    testData = AbstractDataset(test, PAD_IDX, max_len=64)
    # ========================================================================