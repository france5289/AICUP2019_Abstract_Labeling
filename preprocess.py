'''
This file do following jobs:
1. generate testing, validation and training dataset meanwhile write them to seperate csv files
2. Create vocabulary and dump to a pickle file

'''
from tokenizer import NLTKTokenizer
import os
import pandas as pd
from sklearn.model_selection import train_test_split

CWD = os.getcwd()
DATA_PATH = os.path.join(CWD, 'data')


def Remove_Redundant_Columns(dataset):
    '''
    Read dataset and remove Title, Categories, Created Date, Authors columns

    Args:
        dataset(DataFrame) : original dataset
    '''
    dataset.drop('Title', axis=1, inplace=True)
    dataset.drop('Categories', axis=1, inplace=True)
    dataset.drop('Created Date', axis=1, inplace=True)
    dataset.drop('Authors', axis=1, inplace=True)


def SplitSent(doc):
    '''
    Given abstract content and split sentences by '$$$'

    '''
    return doc.split('$$$')


if __name__ == '__main__':
    tokenizer = NLTKTokenizer()
    dataset_path = os.path.join(DATA_PATH, 'task1_trainset.csv')
    dataset = pd.read_csv(dataset_path, dtype=str)
    # Remove redundant columns
    Remove_Redundant_Columns(dataset)
    # Split sentences
    dataset['Abstract'] = dataset['Abstract'].apply(func=SplitSent)
    # bulid dictionary
    for item in dataset['Abstract']:
        tokenizer.build_dict(item)
    # dump dictionary
    filename = os.path.join(DATA_PATH, 'dictionary')
    tokenizer.save_to_file(file_path=filename)
    # train validation split
    trainset, validset = train_test_split(
        dataset, test_size=0.1, random_state=42)
    trainset.to_csv(os.path.join(DATA_PATH, 'trainset.csv'), index=False)
    validset.to_csv(os.path.join(DATA_PATH, 'validset.csv'), index=False)

    test_path = os.path.join(DATA_PATH, 'task1_public_testset.csv')
    testset = pd.read_csv(test_path, dtype=str)
    Remove_Redundant_Columns(testset)
    testset.to_csv(os.path.join(DATA_PATH, 'testset.csv'), index=False)
