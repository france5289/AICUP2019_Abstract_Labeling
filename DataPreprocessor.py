'''
This module provide some helper function to preprocess dataset
'''
import io
import os
import pickle
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
import requests
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm


def collect_words(data_path, n_workers=4):
    df = pd.read_csv(data_path, dtype=str)
        
    # create a list for storing sentences
    sent_list = []
    for i in df.iterrows():
        # remove $$$ and append to sent_list
        sent_list += i[1]['Abstract'].split('$$$')

    chunks = [
        ' '.join(sent_list[i:i + len(sent_list) // n_workers])
        for i in range(0, len(sent_list), len(sent_list) // n_workers)
    ]
    with Pool(n_workers) as pool:
        # word_tokenize for word-word separation
        chunks = pool.map_async(RegexpTokenizer(r'\w+|(?:\[NUM\])').tokenize, chunks)
        words = set(sum(chunks.get(), []))

    return words


def Create_Vocabulary(data_path, pad_idx = 0, unk_idx = 1, workers = 4):
    '''
    Given input dataset path and create relative word vocabulary and dump it to picke file 
    
    Args:
        data_path(string) : path of input dataset
    '''
    assert data_path is not '', 'Please give input dataset path!'
    words = set()
    words |= collect_words(data_path, n_workers=workers)
    word_dict = {'<pad>':pad_idx, '<unk>':unk_idx}
    for word in words:
        word_dict[word] = len(word_dict)

    with open('dictionary.pkl', 'wb') as f:
        pickle.dump(word_dict, f)

    return word_dict

def Load_Vocabulary(vocab_path):
    '''
    Read vocabulary from .pkl

    Args:
    --------
        vocab_path(string): path of .pkl file
    
    Return:
    --------
        vocab(dict)
    '''
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    return vocab


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

def Get_Pretrained_embedding_matrix(filepath, word_dict, embedding_dim):
    '''
    Given word vocabulary, embedding dimension and pretrained  word vector path and
    return relative word embedding matrix 

    Args:
    --------
        filename( string ):  word-embedding filename e.g glove.6B.100d.txt
        word_dict( dictionary obj ) : word vocabulary
        embedding_dim( int ) : embedding dimension
    Return:
    --------
        embedding_matrix(numpy matrix) : a numpy matrix with dimension ( word_dict_size x embedding_dim )  
    '''
    # Parse the unzipped file (a .txt file) to build an index that maps words (as strings) to their vector representation (as number vectors)
    embedding_index = {}
    with open(filepath) as f:
        for line in f:
            values = line.replace(',','').split()
            word = values[0]
            coef = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coef
    print(f'Found {len(embedding_index)} word vectors')
    
    max_words = len(word_dict)
    embedding_matrix = np.random.randn(max_words, embedding_dim)
    
    unk_count = 0
    for word, i in word_dict.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            unk_count += 1
    
    print(f'Found {unk_count} words that did not exist in pre-trained word embeddings')
    
    return embedding_matrix

def label_to_onehot(labels):
    """ Convert label to onehot .

        Args:
            labels (string): sentence's labels.
        Return:
            outputs (onehot list): sentence's onehot label.
    """
    label_dict = {'BACKGROUND': 0, 'OBJECTIVES':1, 'METHODS':2, 'RESULTS':3, 'CONCLUSIONS':4, 'OTHERS':5}
    onehot = [0,0,0,0,0,0]
    for l in labels.split('/'):
        onehot[label_dict[l]] = 1
    return onehot
        
def sentence_to_indices(sentence, word_dict, unk_idx):
    """ Convert sentence to its word indices.

    Args:
        sentence (str): One string.
    Return:
        indices (list of int): List of word indices.
    """
    return [word_dict.get(word,unk_idx) for word in RegexpTokenizer(r'\w+|(?:\[NUM\])').tokenize(sentence)]
    
def Get_dataset(data_path, word_dict, unk_idx, n_workers=4):
    """ Load data and return dataset for training and validating.

    Args:
        data_path (str): Path to the data.
    """
    dataset = pd.read_csv(data_path, dtype=str)

    results = [None] * n_workers
    with Pool(processes=n_workers) as pool:
        for i in range(n_workers):
            batch_start = (len(dataset) // n_workers) * i
            if i == n_workers - 1:
                batch_end = len(dataset)
            else:
                batch_end = (len(dataset) // n_workers) * (i + 1)
            
            batch = dataset[batch_start: batch_end]
            results[i] = pool.apply_async(preprocess_samples, args=(batch,word_dict, unk_idx))

        pool.close()
        pool.join()

    processed = []
    for result in results:
        processed += result.get()
    return processed

def preprocess_samples(dataset, word_dict, unk_idx):
    """ Worker function.

    Args:
        dataset (list of dict)
    Returns:
        list of processed dict.
    """
    processed = []
    for sample in tqdm(dataset.iterrows(), total=len(dataset)):
        processed.append(preprocess_sample(sample[1], word_dict, unk_idx))

    return processed

def preprocess_sample(data, word_dict, unk_idx):
    """
    Args:
        data (dict)
    Returns:
        dict
    """
    ## clean abstracts by removing $$$
    processed = {}
    processed['Abstract'] = [sentence_to_indices(sent, word_dict, unk_idx) for sent in data['Abstract'].split('$$$')]
    
    ## convert the labels into one-hot encoding
    if 'Task 1' in data:
        processed['Label'] = [label_to_onehot(label) for label in data['Task 1'].split(' ')]
        
    return processed
