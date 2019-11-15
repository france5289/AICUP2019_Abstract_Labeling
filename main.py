'''
Run this file to train, valid and test model
'''
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import argparse

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
from DataPreprocessor import Download_Glove, Create_Vocabulary, Create_Glove_embedding_matrix, Get_dataset, Remove_Redundant_Columns

# set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_TOKEN = 0
UNK_TOKEN = 1

# Get data path and cpu num
CWD = os.getcwd()
DATA_PATH = os.path.join(CWD, 'data')
WORKERS = os.cpu_count() // 2

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


def Run_Epoch(epoch, mode, model, criteria, opt, dataset, writer, history, workers=4):
    '''
    run this function to start training or validation process \n
    if given tensorboard summary writer object, then it will write loss and F1_Score to tf board \n

    Args:
        epoch(int) : num of epoch
        mode(string) : train or validate
        model(nn.Module) : your model
        criteria : your loss evaluation criterion
        opt : your optimizer 
        dataset(AbstracDataset obj.) : your dataset object
        workers(int) : how many CPU are used when handle data
        writer(SummaryWriter) : tensorboard summary writer object
        history(dictionary) : a dictionary object which record f1 and loss
    '''

    def run_iter(x, y):
        abstract = x.to(DEVICE)
        labels = y.to(DEVICE)
        o_labels = model(abstract)
        l_loss = criteria(o_labels, labels)
        return o_labels, l_loss

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
    
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)
    loss = 0
    f1_score = F1()
    
    for i, (x, y, sent_len) in trange:
        if epoch == 0:
            writer.add_graph(model, x.to(DEVICE))
        
        o_labels, batch_loss = run_iter(x, y)
        if mode=="train":
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
        
        loss += batch_loss.item()
        f1_score.update(o_labels.cpu(), y)

        trange.set_postfix( loss = loss / ( i + 1 ), f1 = f1_score.print_score() )
    
    if mode == "train":
        history['train'].append({'f1':f1_score.get_score(), 'loss':loss/ len(trange)})
        writer.add_scalar('Loss/train', loss/ len(trange), epoch)
        writer.add_scalar('F1_score/train', f1_score.get_score(), epoch)
    else:
        history['valid'].append({'f1':f1_score.get_score(), 'loss':loss/ len(trange)})
        writer.add_scalar('Loss/valid', loss/ len(trange), epoch)
        writer.add_scalar('F1_score/valid', f1_score.get_score(), epoch)
    trange.close()

def Save(epoch, model, history):
    '''
    Save model status to a pikcle file and dump f1 and loss history to JSON

    Args:
        epoch( int ) : epoch number
        model( nn.Module ) : your model
        history( dictionary obj. ) : a dictionary which record f1 and loss history
    '''
    if not os.path.exists(os.path.join(CWD,'model')):
        os.makedirs(os.path.join(CWD,'model'))
    torch.save(model.state_dict(), os.path.join( CWD,'model/model.pkl.'+str(epoch) ))
    with open( os.path.join( CWD,'model/history.json'), 'w') as f:
        json.dump(history, f, indent=4)

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

def Plot_Figure(history):
    '''
    Pass in training and validation loss and f1 score and plot a figure

    Args:
        history ( dictionary obj ) : a dictionary object that record f1 and loss 
    '''
    train_loss = [l['loss'] for l in history['train']]
    valid_loss = [l['loss'] for l in history['valid']]
    train_f1 = [l['f1'] for l in history['train']]
    valid_f1 = [l['f1'] for l in history['valid']]

    plt.figure(figsize=(7,5))
    plt.title('Loss')
    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='valid')
    plt.legend()
    plt.show()

    plt.figure(figsize=(7,5))
    plt.title('F1 Score')
    plt.plot(train_f1, label='train')
    plt.plot(valid_f1, label='valid')
    plt.legend()
    plt.show()

    print('Best F1 score ', max([[l['f1'], idx] for idx, l in enumerate(history['valid'])]))

def Run_Predict(best_model, model):
    '''
    use the best model status to run prediction and return result

    Args:
        best_model(int) : which epoch has the best model
        model(nn.Module) : your model 
    Return:
        prediction(np.array) : prediction result
    '''
    model.load_state_dict(state_dict=torch.load(os.path.join(CWD,f'model/model.pkl.{best_model}')))
    model.train(False)
    # start testing
    dataloader = DataLoader(dataset=testData,
                            batch_size=16,
                            shuffle=False,
                            collate_fn=testData.collate_fn,
                            num_workers=8)
    trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='Predict')
    prediction = []
    for i, (x, y, sent_len) in trange:
        o_labels = model(x.to(DEVICE))
        o_labels = o_labels > 0.5
        for idx, o_label in enumerate(o_labels):
            prediction.append(o_label[:sent_len[idx]].to('cpu'))
    prediction = torch.cat(prediction).detach().numpy().astype(int)

    return prediction

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfname', '--config_file_name', help='config filename', type=str, default='experiment1_config')
    parser.add_argument('-ebd', '--embedding_dim', help='embedding layer dimension', type=int, default=100)
    parser.add_argument('-hid', '--hidden_dim', help='hidden layer dimension', type=int, default=512)
    parser.add_argument('-lrate', '--learning_rate', help='learning rate', type=float, default=1e-4)
    parser.add_argument('-mepoch', '--max_epoch', help='Max epoch number', type=int, default=10)
    parser.add_argument('-bsize', '--batch_size', help='batch size', type=int, default=16)
    parser.add_argument('-drop', '--drop_prob', help='drop probability', type=float, default=0.3)
    parser.add_argument('-lnum', '--layer_num', help='GRU layer num', type=int, default=1)
    args = parser.parse_args()

    # set hyperparameter
    embedding_dim = args.embedding_dim # word embedding dim for Glove
    hidden_dim = args.hidden_dim
    learning_rate = args.learning_rate
    max_epoch = args.max_epoch
    batch_size = args.batch_size
    drop_p = args.drop_prob
    layer_num = args.layer_num
    # set config file name and write out 
    config_fname = args.config_file_name
    write_config(config_fname, embd_dim=embedding_dim, hidden_dim=hidden_dim, lrate=learning_rate, epoch=max_epoch, 
                batch_size=batch_size, drop=drop_p, layer_num=layer_num)
    
    # read training set
    dataset = pd.read_csv( os.path.join( DATA_PATH,'task1_trainset.csv' ), dtype=str )
    
    # Remove redundant columns
    Remove_Redundant_Columns(dataset)
    dataset['Abstract'] = dataset['Abstract'].str.lower()
    # Remove stop words
    dataset['Abstract'] = dataset['Abstract'].apply(func=remove_stopwords)

    # split training and validation set
    trainset, validset = train_test_split(dataset, test_size=0.1, random_state=42)
    trainset.to_csv(os.path.join(DATA_PATH,'trainset.csv'),index=False)
    validset.to_csv(os.path.join(DATA_PATH,'validset.csv'),index=False)

    # read testing set and remove redundant columns 
    dataset = pd.read_csv(os.path.join(DATA_PATH, 'task1_public_testset.csv'), dtype=str)
    Remove_Redundant_Columns(dataset)
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
    model.to(DEVICE)

    # Tensorboard 
    # save path: test_experiment/
    tf_path = os.path.join(CWD, 'test_experiment')
    if not os.path.exists(tf_path):
        os.mkdir(tf_path)
    tf_writer = SummaryWriter(os.path.join(tf_path,config_fname))
    history = {'train':[],'valid':[]}

    for epoch in range(max_epoch):
        print(f'Epoch:{epoch}')
        Run_Epoch(epoch, 'train', model, criteria, opt, trainData, history, tf_writer, WORKERS)
        Run_Epoch(epoch, 'valid', model, criteria, opt, validData, history, tf_writer, WORKERS)
        Save(epoch, model, history)
    
    # Plot the training results
    Plot_Figure(history)
    # run prediction process
    best_model = int(input('Please insert epoch for best model'))
    prediction = Run_Predict(best_model, model)

    # Output csv for submission
    SubmitGenerator(prediction, os.path.join(DATA_PATH, 'task1_sample_submission.csv'),
                    True,
                    os.path.join(CWD, f'submission_{config_fname}.csv'))