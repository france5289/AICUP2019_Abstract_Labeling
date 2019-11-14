'''
Run this file to train, valid and test model
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle, re
import os


from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from tqdm import trange

from gensim.parsing import remove_stopwords

from model import Net1, F1 # import all model object
from config_writer import write_config




if __name__ == '__main__':
    CWD = os.getcwd()
    DATA_PATH = os.path.join(CWD, 'data')
    CPUNUM = os.cpu_count() // 2