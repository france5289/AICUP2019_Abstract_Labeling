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

def labels_to_onehot(labels):
    '''
    Convert labels to one-hot encoding

    Args : 
        labels:( DataFrame column item ) 
    Return :
        one_hot_labels: ( DataFrame column item )
    '''
    one_hot_labels = []
    label_list = labels.split(' ')
    label_dict = {'BACKGROUND': 0, 'OBJECTIVES':1, 'METHODS':2, 'RESULTS':3, 'CONCLUSIONS':4, 'OTHERS':5}
    for label in label_list:
        onehot = [0,0,0,0,0,0]
        for l in label.split('/'):
            onehot[label_dict[l]] = 1
        one_hot_labels.append(onehot)
    
    return one_hot_labels

def encode_data(dataset):
    '''
    encode 'Abstract' and convert label to one_hot


    Args:
        dataset(pd.DataFrame)
    Return:
        abstract(pd.Series)
        labels(pd.Series) : if dataset dosen't contain label, then return None 
    '''
    global Tokenizer
    abstract = dataset['Abstract'].apply(func=Tokenizer.encode)
    labels = None
    if 'Task 1' in dataset.columns:
        labels = dataset['Task 1'].apply(func=labels_to_onehot)

    return abstract, labels


if __name__ == '__main__':
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(VALID_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        os.system('python3 gendata.py')

    train = pd.read_csv(TRAIN_DATA_PATH)
    valid = pd.read_csv(VALID_DATA_PATH)
    test = pd.read_csv(TEST_DATA_PATH)

    train['Abstract'] = train['Abstract'].apply(func=SplitSent)
    valid['Abstract'] = valid['Abstract'].apply(func=SplitSent)
    test['Abstract'] = test['Abstract'].apply(func=SplitSent)
    GenDict(train, valid)

    # encode 'Abstract' and convert label to one_hot
    train_encode, train_label = encode_data(train)
    valid_encode, valid_label = encode_data(valid)
    test_encode, _ = encode_data(test)