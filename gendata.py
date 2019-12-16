import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
DATA_PATH = os.path.join(os.getcwd(), 'data')


def Remove_Redundant_Columns(dataset):
    '''
    Read dataset and remove Title, Categories, Created Date, Authors columns

    Args:
        dataset(DataFrame) : original dataset
    '''
    dataset.drop('Id', axis=1, inplace=True)
    dataset.drop('Title', axis=1, inplace=True)
    dataset.drop('Categories', axis=1, inplace=True)
    dataset.drop('Created Date', axis=1, inplace=True)
    dataset.drop('Authors', axis=1, inplace=True)

# TODO : use multi-processing to speed up data preprocessing!

def Extract_Sentences(dataset):
    '''
    Extract Abstract sentences to form a new dataframe which every row contains a sentences and relative label
    
    Args:
        dataset(DataFrame) : dataframe which has been removed some redundatn columns
    Return:
        newframe(DataFrame) : new dataframe
    '''
    if 'Task 1' in dataset:
        newframe = pd.DataFrame(columns=['Abstract', 'Task 1'])
        for _, row in dataset.iterrows():
            for sent, label in zip(row['Abstract'], row['Task 1']):
                newframe = newframe.append({'Abstract' : sent, 'Task 1' : label}, ignore_index=True)
    else:
        newframe = pd.DataFrame(columns=['Abstract'])
        for _, row in dataset.iterrows():
            for sent in row['Abstract']:
                newframe = newframe.append({'Abstract' : sent}, ignore_index=True)
    return newframe        

def CleanData(sent):
    '''
    Try to remove website link, and use [NUM] to replace numbers
    '''
    sent = re.sub(r'([--:\w?@%&+~#=]*\.[a-z]{2,4}\/{0,2})((?:[?&](?:\w+)=(?:\w+))+|[--:\w?@%&+~#=]+)?', '', sent)
    sent = re.sub(r'[0-9]+\.[0-9]+', ' [NUM] ', sent)
    sent = re.sub(r'[0-9]+\.[0-9]+', ' [NUM] ', sent)
    return sent



if __name__ == '__main__':
    tqdm.pandas()
    dataset = pd.read_csv(os.path.join(DATA_PATH,'task1_trainset.csv'), dtype=str)
    print('Preprocessing task1_trainset.csv')
    Remove_Redundant_Columns(dataset)
    dataset['Abstract'] = dataset['Abstract'].progress_apply(func = CleanData)
    dataset['Abstract'] = dataset['Abstract'].progress_apply(func = lambda doc : doc.split('$$$'))
    dataset['Task 1'] = dataset['Task 1'].progress_apply(func = lambda labels : labels.split(' '))
    #dataset = Extract_Sentences(dataset)
    dataset.dropna(inplace=True)
    print('Split to train and valid')
    trainset, validset = train_test_split(dataset, test_size=0.1, random_state=42)
    trainset.to_csv(os.path.join(DATA_PATH, 'trainset.csv'), index=False)
    validset.to_csv(os.path.join(DATA_PATH, 'validset.csv'), index=False)
    print('Preprocessing task1_public_testset.csv')
    testset = pd.read_csv(os.path.join(DATA_PATH, 'task1_public_testset.csv'), dtype=str)
    Remove_Redundant_Columns(testset)
    testset['Abstract'] = testset['Abstract'].progress_apply(func= CleanData)
    testset['Abstract'] = testset['Abstract'].progress_apply(func = lambda doc : doc.split('$$$'))
    #testset = Extract_Sentences(testset)
    testset.dropna(inplace=True)
    testset.to_csv(os.path.join(DATA_PATH, 'testset.csv'), index=False)
    print('Preprocessing completed!')