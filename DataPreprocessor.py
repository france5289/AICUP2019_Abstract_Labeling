'''
This module provide some helper function to preprocess dataset
'''
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
import os
import pickle
import pandas as pd
import requests, zipfile, io
import torch
import numpy as np

def _collect_words(data_path, n_workers=4):
    df = pd.read_csv(data_path, dtype=str)
    # create a list for storing sentences
    sent_list = []
    for _ , row in df.iterrows():
        # remove $$$ and append to sent_list
        sent_list.extend(row['Abstract'].split('$$$'))

    chunks = [
        ' '.join(sent_list[i:i + len(sent_list) // n_workers])
        for i in range(0, len(sent_list), len(sent_list) // n_workers)
    ]
    with Pool(n_workers) as pool:
        # word_tokenize for word-word separation
        chunks = pool.map_async(word_tokenize, chunks)
        words = set(sum(chunks.get(), []))

    return words

def Create_Vocabulary(data_path):
    '''
    Given input dataset path and create relative word vocabulary and dump it to picke file \n
    Args:
        data_path(string) : path of input dataset
    '''
    assert data_path is not '', 'Please give input dataset path!'
    CPUNUM = os.cpu_count() // 2
    words = set()
    words |= _collect_words(data_path, n_workers=CPUNUM)
    PAD_TOKEN = 0
    UNK_TOKEN = 1
    word_dict = {'<pad>':PAD_TOKEN, '<unk>':UNK_TOKEN}
    for word in words:
        word_dict[word] = len(word_dict)

    with open('dictionary.pkl', 'wb') as f:
        pickle.dump(word_dict, f)

    return word_dict



def Download_Glove():
    '''
    Download Glove pretrained word embedding from Web
    '''
    if not os.path.exists('glove'):
        os.mkdir('glove')
        r = requests.get('http://nlp.stanford.edu/data/glove.6B.zip')
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path='glove')
        print('Extract Glove pretrained word embedding to {working dir}/glove successfully')
    else:
        print('Glove pretrained word embedding has already existed')

def Create_Glove_embedding_matrix(filename, word_dict, embedding_dim):
    '''
    Given word vocabulary, embedding dimension and pretrained Glove word vector path \n
    return relative word embedding matrix \n
    Args:
        filename( string ): glove word-embedding filename e.g glove.6B.100d.txt
        word_dict( dictionary obj ) : word vocabulary
        embedding_dim( int ) : embedding dimension
    Return:
        embedding_matrix(torch.FloatTensor) : a Float Tensor with dimension ( word_dict_size x embedding_dim )  
    '''
    # Parse the unzipped file (a .txt file) to build an index that maps words (as strings) to their vector representation (as number vectors)
    CWD = os.getcwd()
    wordvector_path = os.path.join(CWD, 'glove', filename)
    embedding_index = {}
    with open(wordvector_path) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coef = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coef
    # Preparing the GloVe word-embeddings matrix
    max_words = len(word_dict)
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_dict.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return torch.FloatTensor(embedding_matrix)
