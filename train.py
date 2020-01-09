import json

import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from config import Config
from DataPreprocessor import (  Create_Vocabulary, Get_dataset,
                                Get_Pretrained_embedding_matrix, Load_Vocabulary)
from dataset import AbstractDataset
from metrics import F1
from model import Net

# ============ uncomment the following line to use VSCode python debugger! ================
# import multiprocessing
# multiprocessing.set_start_method('spawn', True) 



# set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data path and cpu num
CWD = os.getcwd()
TRAIN_DATA_PATH = os.path.join(CWD, 'data', 'trainset.csv')
VALID_DATA_PATH = os.path.join(CWD, 'data', 'validset.csv')
# TEST_DATA_PATH = os.path.join(CWD, 'data', 'testset.csv')
# DICT_PATH = os.path.join(CWD, 'dictionary.pkl')
HPARAMS_PATH = os.path.join(CWD, 'hyperparameters.json')

WORKERS = os.cpu_count() 

PAD_IDX = 0
UNK_IDX = 1


def Setup_seed(seed):
    '''
    Use this function to setup random seep for reproducibility

    Args:
    --------
        seed(int) : random seed
    Returns:
    --------
        No return value
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def Run_Epoch(epoch, mode, shuffle, data, config, model, opt, criteria, writer, history):
    '''
    Use this function to run an epoch of training or validation

    Args:
    --------
        epoch(int) : epoch number
        mode(string) : 'Train' or 'Validation'
        shuffle(boolean) : shuffle data or not, this should be True during training
        data(torch Dataset) : which dataset do you want to use
        config(Config object) : hyperparameters
        opt(torch Optimizer) : what optimizer do you use
        criteria(torch Loss function) : waht loss function do you use
        writer(tf.SummaryWriter) : tensorboard writer
        history(dict) : dictionary to log loss and f1 score
    Returns:
    --------
        No return values
    '''
    def run_iter(x,y):
        abstract = x.to(DEVICE)
        labels = y.to(DEVICE)
        o_labels = model(abstract)
        l_loss = criteria(o_labels, labels)
        return o_labels, l_loss
    
    model.train(True)
    dataloader = DataLoader(dataset=data,
                            batch_size=config.batch_size,
                            shuffle=shuffle,
                            collate_fn=data.collate_fn,
                            num_workers=WORKERS)
    
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=mode)
    loss = 0
    f1_score = F1()
    for i,(x, y, sent_len) in trange:
        o_labels, batch_loss = run_iter(x, y)
        if mode == 'Train':
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
        
        loss += batch_loss.item()
        f1_score.update(o_labels.cpu(), y)
        trange.set_postfix(loss=loss / (i+1), f1=f1_score.print_score())
    
    # ======== Log f1 and loss ========
    if mode == 'Train':
        history['train_F1'].append(f1_score.get_score())
        history['train_loss'].append(loss / len(trange))
        writer.add_scalar('Loss/train', loss / len(trange), epoch)
        writer.add_scalar('F1_score/train', f1_score.get_score(), epoch)
    else:
        history['valid_F1'].append(f1_score.get_score())
        history['valid_loss'].append(loss / len(trange))
        writer.add_scalar('Loss/valid', loss / len(trange), epoch)
        writer.add_scalar('F1_score/valid', f1_score.get_score(), epoch)
    # =================================
    trange.close()

def Save(epoch, model, history, filename):
    '''
    Save model status to a pikcle file and dump f1 and loss history to JSON

    Args:
    --------
        epoch( int ) : epoch number
        model( nn.Module ) : your model
        history( dictionary obj. ) : a dictionary which record f1 and loss history
        config_fname( string ) : JSON filename
    Returns:
    --------
        No return values
    '''
    if not os.path.exists(os.path.join(CWD, 'model', filename)):
        os.makedirs(os.path.join(CWD, 'model', filename))
    torch.save(model.state_dict(),os.path.join(CWD, f'model/{filename}/model.pkl.' + str(epoch)))
    with open(os.path.join(CWD, f'model/{filename}/history.json'),'w') as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(VALID_DATA_PATH):
        os.system('python3 gendata.py')
    # ======== Hyperparameters settings ========
    myconfig = Config.load_from_json(HPARAMS_PATH)
    # ==========================================

    # ======== Random seed setup ===============
    Setup_seed(myconfig.seed)
    # ==========================================

    # ======== Create Vocabulary from training set ========
    myvocab = Create_Vocabulary(TRAIN_DATA_PATH, pad_idx=PAD_IDX, unk_idx=UNK_IDX ,workers=WORKERS)
    # =====================================================

    # ======== Get Pretrained embedding matrix ========
    embedding_matrix = torch.FloatTensor(Get_Pretrained_embedding_matrix(myconfig.pretrained_embedding_path, myvocab, myconfig.embedding_dim))
    # =================================================

    # ======== Preprocess training set, validation set and testing set =======
    print('[INFO] Start processing trainset...')
    train = Get_dataset(TRAIN_DATA_PATH, myvocab, UNK_IDX, WORKERS)
    print('[INFO] Start processing validset...')
    valid = Get_dataset(VALID_DATA_PATH, myvocab, UNK_IDX, WORKERS)
    # print('[INFO] Start processing testset...')
    # test = Get_dataset(TEST_DATA_PATH, myvocab, UNK_IDX, WORKERS)
    # ========================================================================

    # ======== construct custom dataset ===================================
    trainData = AbstractDataset(train, PAD_IDX, max_len=64)
    validData = AbstractDataset(valid, PAD_IDX, max_len=64)
    # testData = AbstractDataset(test, PAD_IDX, max_len=64)
    # ========================================================================

    # ======== Construct model, choose optimizer and loss function ===========
    Model = Net(myconfig, len(myvocab), embedding_matrix)
    Opt = torch.optim.AdamW(Model.parameters(), lr=myconfig.learning_rate)
    Criteria = torch.nn.BCELoss()
    Model.to(DEVICE)
    # ========================================================================

    # ======== Tensorboard and Logging dictionary Setup ===================================
    history = {'train_F1':[], 'train_loss':[], 'valid_F1':[], 'valid_loss':[]}
    tf_path = os.path.join(CWD, 'test_experiment')
    if not os.path.exists(tf_path):
        os.mkdir(tf_path)
    writer = SummaryWriter(os.path.join(tf_path, myconfig.expname))
    # ==============================================================

    # ======== Start training and validation process ================
    for epoch in range(myconfig.max_epoch):
        print(f'Epoch:{epoch}')
        Run_Epoch(epoch, 'Train', True, trainData, myconfig, Model, Opt, Criteria, writer, history)
        Run_Epoch(epoch, 'Valid', False, validData, myconfig, Model, Opt, Criteria, writer, history)
        Save(epoch, Model, history, myconfig.expname)
    # ===============================================================

    # ======== Log hyperparameters to tensorboard ===================
    hparams = {
        'seed':myconfig.seed,
        'embedding_dim':myconfig.embedding_dim,
        'hidden_dim':myconfig.hidden_dim,
        'lrate':myconfig.learning_rate,
        'epoch':myconfig.max_epoch,
        'batch_size':myconfig.batch_size,
        'removeOOV':myconfig.removeOOV,
        'removepunct':myconfig.removepunct
    }
    for key, value in history.items():
        if 'loss' in key:
            history[key] = min(value)
        else:
            history[key] = max(value)
    writer.add_hparams(hparams, history)
    writer.close()
    # ===============================================================