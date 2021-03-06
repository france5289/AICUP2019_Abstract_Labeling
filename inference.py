
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from config import Config
from DataPreprocessor import (Get_dataset, Get_Pretrained_embedding_matrix,
                              Load_Vocabulary)
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
TEST_DATA_PATH = os.path.join(CWD, 'data', 'testset.csv')
DICT_PATH = os.path.join(CWD, 'dictionary.pkl')
HPARAMS_PATH = os.path.join(CWD, 'hyperparameters.json')

WORKERS = os.cpu_count() 

PAD_IDX = 0
UNK_IDX = 1


### Helper function for creating a csv file following the submission format

def SubmitGenerator(prediction, sampleFile, public=True, filename='prediction.csv'):
    sample = pd.read_csv(sampleFile)
    submit = {}
    submit['order_id'] = list(sample.order_id.values)
    redundant = len(sample) - prediction.shape[0]
    if public:
        submit['BACKGROUND'] = list(prediction[:,0]) + [0]*redundant
        submit['OBJECTIVES'] = list(prediction[:,1]) + [0]*redundant
        submit['METHODS'] = list(prediction[:,2]) + [0]*redundant
        submit['RESULTS'] = list(prediction[:,3]) + [0]*redundant
        submit['CONCLUSIONS'] = list(prediction[:,4]) + [0]*redundant
        submit['OTHERS'] = list(prediction[:,5]) + [0]*redundant
    else:
        submit['BACKGROUND'] = [0]*redundant + list(prediction[:,0])
        submit['OBJECTIVES'] = [0]*redundant + list(prediction[:,1])
        submit['METHODS'] = [0]*redundant + list(prediction[:,2])
        submit['RESULTS'] = [0]*redundant + list(prediction[:,3])
        submit['CONCLUSIONS'] = [0]*redundant + list(prediction[:,4])
        submit['OTHERS'] = [0]*redundant + list(prediction[:,5])
    df = pd.DataFrame.from_dict(submit) 
    df.to_csv(filename,index=False)


if __name__ == "__main__":
    # You should set BEST_MODEL_PATH by yourself
    BEST_MODEL_PATH = 'model/BestModel/model.pkl.18'
    
    # ======== Hyperparameters settings ========
    myconfig = Config.load_from_json(HPARAMS_PATH)
    # ==========================================

    # ======== Load vocabulary ===================
    myvocab = Load_Vocabulary(DICT_PATH)
    # ============================================

    # ======== Get Pretrained embedding matrix ========
    embedding_matrix = torch.FloatTensor(Get_Pretrained_embedding_matrix(myconfig.pretrained_embedding_path, myvocab, myconfig.embedding_dim))
    # =================================================

    # ======== Preprocess testing dataset ========
    print('[INFO] Start processing testset...')
    test = Get_dataset(TEST_DATA_PATH, myvocab, UNK_IDX, WORKERS)
    # ============================================

    # ======== Construct custom dataset ==========
    testData = AbstractDataset(test, PAD_IDX, max_len=64)
    # ============================================

    # ======== Contruct custom dataloader ========
    dataloader = DataLoader(dataset=testData,
                            batch_size=32,
                            shuffle=False,
                            collate_fn=testData.collate_fn,
                            num_workers=WORKERS)
    # ============================================

    # ======== Load Model ========================
    model = Net(myconfig, len(myvocab), embedding_matrix)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    model.to(DEVICE)
    # ============================================

    # ======== Start inference process ===========
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='Predict')
    prediction = []
    for i, (x, y, sent_len) in trange:
        o_labels = model(x.to(DEVICE))
        o_labels = o_labels > 0.4
        for idx, o_label in enumerate(o_labels):
            prediction.append(o_label[:sent_len[idx]].to('cpu'))
    prediction = torch.cat(prediction).detach().numpy().astype(int)
    # ============================================

    # ======== Output csv for submission ========
    SubmitGenerator(prediction,
                    os.path.join(CWD,'data/task1_sample_submission.csv'), 
                    True, 
                    os.path.join(CWD,'submission_1230(0.684).csv'))
    # ===========================================
