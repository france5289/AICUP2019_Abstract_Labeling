'''
Run this file to train, valid and test model
'''
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from tqdm import trange

from gensim.parsing import remove_stopwords

from model import Net1, F1 
from config_writer import write_config
from DataPreprocessor import Download_Glove, Create_Vocabulary, Create_Glove_embedding_matrix, Get_dataset



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


def Run_Epoch(epoch, mode, model, dataset, workers=4):
    '''
    run this function to start training or validation process

    Args:
        epoch(int) : num of epoch
        mode(string) : train or validate
        model(nn.Module) : your model
        dataset(AbstracDataset obj.) : your dataset object
        workers(int) : how many CPU are used when handle data
    Return:
        history(dict) : a dictionary which record f1 score and loss
    '''
    model.train(True)
    if mode == "train":
        description = 'Train'
        shuffle = True
    else:
        description = 'Valid'
        shuffle = False
    
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=dataset.collate_fn,
                            num_workers=8)
    
    



if __name__ == '__main__':
    PAD_TOKEN = 0
    UNK_TOKEN = 1
    
    # Get data path and cpu num
    CWD = os.getcwd()
    DATA_PATH = os.path.join(CWD, 'data')
    WORKERS = os.cpu_count() // 2

    # set hyperparameter
    embedding_dim = 100 # word embedding dim for Glove
    hidden_dim = 512
    learning_rate = 1e-4
    max_epoch = 10
    batch_size = 16
    drop_p = 0.3
    layer_num = 2

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set config file name and write out 
    
    config_fname = 'Experiment1_config'
    write_config(config_fname, embd_dim=embedding_dim, hidden_dim=hidden_dim, lrate=learning_rate, epoch=max_epoch, 
                batch_size=batch_size, drop=drop_p, layer_num=layer_num)
    
    # read training set
    dataset = pd.read_csv( os.path.join( DATA_PATH,'task1_trainset.csv' ), dtype=str )
    
    # Remove redundant columns
    dataset.drop('Title',axis=1,inplace=True)
    dataset.drop('Categories',axis=1,inplace=True)
    dataset.drop('Created Date',axis=1, inplace=True)
    dataset.drop('Authors',axis=1,inplace=True)
    dataset['Abstract'] = dataset['Abstract'].str.lower()
    
    # Remove stop words
    dataset['Abstract'] = dataset['Abstract'].apply(func=remove_stopwords)

    # split training and validation set
    trainset, validset = train_test_split(dataset, test_size=0.1, random_state=42)
    trainset.to_csv(os.path.join(DATA_PATH,'trainset.csv'),index=False)
    validset.to_csv(os.path.join(DATA_PATH,'validset.csv'),index=False)

    # read testing set and remove redundant columns 
    dataset = pd.read_csv(os.path.join(DATA_PATH, 'task1_public_testset.csv'), dtype=str)
    dataset.drop('Title', axis=1, inplace=True)
    dataset.drop('Categories', axis=1, inplace=True)
    dataset.drop('Created Date', axis=1, inplace=True)
    dataset.drop('Authors', axis=1, inplace=True)
    dataset['Abstract'] = dataset['Abstract'].str.lower()
    # remove stop words
    dataset['Abstract'] = dataset['Abstract'].apply(func=remove_stopwords)

    dataset.to_csv(os.path.join(DATA_PATH, 'testset.csv'), index=False)

    #---------------now we have generate training, validation, testing set-----------

    # Collect words and create the vocabulary set
    word_dict = Create_Vocabulary(os.path.join(DATA_PATH, 'trainset.csv'))

    # download Glove pretrained word embedding
    Download_Glove()
    
    # Generate word embedding matrix
    embedding_matrix = Create_Glove_embedding_matrix('glove.6B.100d.txt', word_dict, embedding_dim)

    # Data formatting
    print('[INFO] Start processing trainset...')
    train = Get_dataset(os.path.join(DATA_PATH,'trainset.csv'), word_dict, n_workers=WORKERS)
    print('[INFO] Start processing validset...')
    valid = Get_dataset(os.path.join(DATA_PATH,'validset.csv'), word_dict, n_workers=WORKERS)
    print('[INFO] Start processing testset...')
    test = Get_dataset(os.path.join(DATA_PATH,'testset.csv'), word_dict, n_workers=WORKERS)

    # Create AbstractDataset object
    trainData = AbstractDataset(train, PAD_TOKEN, max_len = 64)
    validData = AbstractDataset(valid, PAD_TOKEN, max_len = 64)
    testData = AbstractDataset(test, PAD_TOKEN, max_len = 64)

    # create model
    model = Net1(len(word_dict), embedding_dim, hidden_dim, embedding_matrix, layer_num, drop_p, True)

    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criteria = torch.nn.BCELoss()
    model.to(device)
    
