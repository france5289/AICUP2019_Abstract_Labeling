import os
import pandas as pd
from sklearn.model_selection import train_test_split


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


if __name__ == '__main__':
    dataset = pd.read_csv(os.path.join(DATA_PATH,'task1_trainset.csv'), dtype=str)
    Remove_Redundant_Columns(dataset)
    trainset, validset = train_test_split(
        dataset, test_size=0.1, random_state=42)
    trainset.to_csv(os.path.join(DATA_PATH, 'trainset.csv'), index=False)
    validset.to_csv(os.path.join(DATA_PATH, 'validset.csv'), index=False)

    testset = pd.read_csv(os.path.join(DATA_PATH, 'task1_public_testset.csv'), dtype=str)
    Remove_Redundant_Columns(testset)
    testset.to_csv(os.path.join(DATA_PATH, 'testset.csv'), index=False)