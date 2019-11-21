import pandas as pd
import os
import pickle

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
from tqdm import trange

from model import GRUNet, F1
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
PAD_TOKEN = '[PAD]'
PAD_TOKEN_ID = 0
EOS_TOKEN = '[EOS]'
EOS_TOKEN_ID = 3


Tokenizer = NLTKTokenizer(pad_token=PAD_TOKEN, pad_token_id=PAD_TOKEN_ID,
                          eos_token=EOS_TOKEN, eos_token_id=EOS_TOKEN_ID)


class Abstract(Dataset):
    def __init__(self, data, pad_idx, eos_id):
        self.data = data
        self.pad_idx = pad_idx
        self.eos_token = eos_id

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, index):
        return self.data.iloc[index]

    def collate_fn(self, datas):
        abstracts = [torch.as_tensor(abstract, dtype=torch.long)
                     for data in datas for abstract in data['Abstract']]
        batch_abstracts = pad_sequence(
            abstracts, batch_first=True, padding_value=self.pad_idx)
        
        b, s = batch_abstracts.size() # b: batch, s:sequence length
        batch_eos = batch_abstracts == 3
        eos_index_matrix = batch_eos.nonzero()
        eos_index_list = list()
        prev = 0
        for row in eos_index_matrix:
            eos_index_list.append(row[1].item()+prev)
            prev = prev + s    

        batch_labels = None
        labels = [
            label for data in datas if 'Task 1' in data for label in data['Task 1']]
        if len(labels) != 0:
            batch_labels = torch.as_tensor(labels, dtype=torch.float)
            batch_labels = batch_labels.view(-1, 6)

        return batch_abstracts, batch_labels, eos_index_list


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
    label_dict = {'BACKGROUND': 0, 'OBJECTIVES': 1, 'METHODS': 2,
                  'RESULTS': 3, 'CONCLUSIONS': 4, 'OTHERS': 5}
    for label in label_list:
        onehot = [0, 0, 0, 0, 0, 0]
        for l in label.split('/'):
            onehot[label_dict[l]] = 1
        one_hot_labels.append(onehot)

    return one_hot_labels


def encode_data(dataset):
    '''
    encode 'Abstract' and convert label to one_hot


    Args:
        dataset(pd.DataFrame)
    '''
    global Tokenizer
    dataset['Abstract'] = dataset['Abstract'].apply(func=Tokenizer.encode)
    if 'Task 1' in dataset.columns:
        dataset['Task 1'] = dataset['Task 1'].apply(func=labels_to_onehot)


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
    print('Start encoding train, valid, test dataset')
    encode_data(train)
    encode_data(valid)
    encode_data(test)
    print('Encoding process complete!')

    trainset = Abstract(data=train, pad_idx=PAD_TOKEN_ID, eos_id=EOS_TOKEN_ID)
    validset = Abstract(data=valid, pad_idx=PAD_TOKEN_ID, eos_id=EOS_TOKEN_ID)
    testset = Abstract(data=test, pad_idx=PAD_TOKEN_ID, eos_id=EOS_TOKEN_ID)

    #-----------------------hyperparameter setting block-------------------
    # TO-DO : use a object or other data structure to pack hyperparameters
    embedding_dim = 100
    hidden_dim = 512
    lrate = 1e-4
    max_epoch = 10
    batch = 16
    drop_pb = 0.25
    layers = 1
    expname = 'modelVer4_test1'
    #-----------------------hyperparameter setting block-------------------
