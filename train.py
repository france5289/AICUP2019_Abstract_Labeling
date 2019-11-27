import numpy as np
import pandas as pd
import os
import pickle
import json

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
from tqdm import trange

from model import GRUNet, F1
from DataPreprocessor import Download_Glove, Create_Glove_embedding_matrix
from tokenizer import NLTKTokenizer

# set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data path and cpu num
CWD = os.getcwd()
TRAIN_DATA_PATH = os.path.join(CWD, 'data', 'trainset.csv')
VALID_DATA_PATH = os.path.join(CWD, 'data', 'validset.csv')
TEST_DATA_PATH = os.path.join(CWD, 'data', 'testset.csv')
DICT_PATH = os.path.join(CWD, 'data', 'dictionary.pkl')

WORKERS = os.cpu_count() // 2
# default Tokenizer
PAD_TOKEN = '[PAD]'
PAD_TOKEN_ID = 0
EOS_TOKEN = '[EOS]'
EOS_TOKEN_ID = 3

Tokenizer = NLTKTokenizer(  pad_token=PAD_TOKEN,
                            pad_token_id=PAD_TOKEN_ID,
                            eos_token=EOS_TOKEN,
                            eos_token_id=EOS_TOKEN_ID  )


class Abstract(Dataset):
    def __init__(self, data, pad_idx, eos_id):
        self.data = data
        self.pad_idx = pad_idx
        self.eos_token = eos_id

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, index):
        return self.data.iloc[index]

    def collate_fn(self, datas):
        abstracts = [
            torch.as_tensor(abstract, dtype=torch.long) for data in datas
            for abstract in data['Abstract']
        ]
        batch_abstracts = pad_sequence(abstracts, batch_first=True, padding_value=self.pad_idx)

        _, s = batch_abstracts.size()  # b: batch, s:sequence length
        batch_eos = batch_abstracts == self.eos_token
        eos_index_matrix = batch_eos.nonzero()
        eos_index_list = list()
        prev = 0
        for row in eos_index_matrix:
            eos_index_list.append(row[1].item() + prev)
            prev = prev + s

        batch_labels = None
        labels = [
            label for data in datas if 'Task 1' in data
            for label in data['Task 1']
        ]
        if len(labels) != 0:
            batch_labels = torch.as_tensor(labels, dtype=torch.float)
            batch_labels = batch_labels.view(-1, 6)

        return batch_abstracts, batch_labels, torch.as_tensor(eos_index_list, dtype=torch.long)


def SplitSent(doc):
    return doc.split('$$$')


def GenDict(train, valid):
    global Tokenizer
    if os.path.exists(DICT_PATH):
        Tokenizer = NLTKTokenizer.load_from_file(DICT_PATH)
    else:
        for item in train['Abstract']:
            Tokenizer.build_dict(item)

        for item in valid['Abstract']:
            Tokenizer.build_dict(item)
        Tokenizer.save_to_file(DICT_PATH)


def labels_to_onehot(labels):
    '''
    Convert labels to one-hot encoding

    Args : 
        labels:( DataFrame column item ) 
    Return :
        one_hot_labels: ( DataFrame column item )
    '''
    one_hot_labels = []
    label_list = labels.split(' ')
    label_dict = {
        'BACKGROUND': 0,
        'OBJECTIVES': 1,
        'METHODS': 2,
        'RESULTS': 3,
        'CONCLUSIONS': 4,
        'OTHERS': 5
    }
    for label in label_list:
        onehot = [0, 0, 0, 0, 0, 0]
        for l in label.split('/'):
            onehot[label_dict[l]] = 1
        one_hot_labels.append(onehot)

    return one_hot_labels


def encode_data(dataset):
    '''
    encode 'Abstract' and convert label to one_hot


    Args:
        dataset(pd.DataFrame)
    '''
    global Tokenizer
    tqdm.pandas()
    dataset['Abstract'] = dataset['Abstract'].progress_apply(func=Tokenizer.encode)
    if 'Task 1' in dataset.columns:
        dataset['Task 1'] = dataset['Task 1'].progress_apply(func=labels_to_onehot)


def Run_Epoch(epoch, mode, model, criteria, opt, dataset, batch, writer, history, workers=WORKERS):
    '''
    run this function to start training or validation process

    Args:
        epoch(int) : current epoch
        mode(string) : train or validate
        model(nn.Module) : your model
        criteria : yout loss evaluation criterion
        opt : your optimizer
        dataset(Dataset obj.) : dataset object
        batch : batch size
        writer(SummaryWriter) : tensorboard summary writer object
        history(dict obj) : a dictionary record loss and f1 score
        workers(int) : how many CPU cores are used when processing data
    '''
    def run_iter(x, y, indexlist):
        abstract = x.to(DEVICE)
        labels = y.to(DEVICE)
        indexes = indexlist.to(DEVICE)
        o_labels = model(abstract, indexes)
        l_loss = criteria(o_labels, labels)
        return o_labels, l_loss

    model.train(True)
    if mode == 'train':
        description = 'Train'
        shuffle = True
    else:
        description = 'Valid'
        shuffle = False
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch,
                            shuffle=shuffle,
                            collate_fn=dataset.collate_fn,
                            num_workers=workers)

    trange = tqdm(enumerate(dataloader),total=len(dataloader),desc=description)
    loss = 0
    f1_score = F1()

    for i, (x, y, index_list) in trange:
        if epoch == 0:
            writer.add_graph(model, [x.to(DEVICE), index_list.to(DEVICE)])
        o_labels, batch_loss = run_iter(x, y, index_list)
        if mode == 'train':
            opt.zero_grad()
            batch_loss.backward()
            opt.step()

        loss += batch_loss.item()
        f1_score.update(o_labels.cpu(), y)

        trange.set_postfix(loss=loss / (i + 1), f1=f1_score.print_score())
    if mode == 'train':
        # history['train'].append({'f1': f1_score.get_score(),'loss': loss / len(trange)})
        history['train_F1'].append(f1_score.get_score())
        history['train_loss'].append(loss / len(trange))
        writer.add_scalar('Loss/train', loss / len(trange), epoch)
        writer.add_scalar('F1_score/train', f1_score.get_score(), epoch)
    else:
        # history['valid'].append({'f1': f1_score.get_score(),'loss': loss / len(trange)})
        history['valid_F1'].append(f1_score.get_score())
        history['valid_loss'].append(loss / len(trange))
        writer.add_scalar('Loss/valid', loss / len(trange), epoch)
        writer.add_scalar('F1_score/valid', f1_score.get_score(), epoch)
    trange.close()


def Save(epoch, model, history, config_fname):
    '''
    Save model status to a pikcle file and dump f1 and loss history to JSON

    Args:
        epoch( int ) : epoch number
        model( nn.Module ) : your model
        history( dictionary obj. ) : a dictionary which record f1 and loss history
        config_fname( string ) : JSON filename
    '''
    if not os.path.exists(os.path.join(CWD, 'model', config_fname)):
        os.makedirs(os.path.join(CWD, 'model', config_fname))
    torch.save(
        model.state_dict(),
        os.path.join(CWD, f'model/{config_fname}/model.pkl.' + str(epoch)))
    with open(os.path.join(CWD, f'model/{config_fname}/history.json'),'w') as f:
        json.dump(history, f, indent=4)


def SubmitGenerator(prediction,sampleFile,public=True,filename='prediction.csv'):
    sample = pd.read_csv(sampleFile)
    submit = {}
    submit['order_id'] = list(sample.order_id.values)
    redundant = len(sample) - prediction.shape[0]
    if public:
        submit['BACKGROUND'] = list(prediction[:, 0]) + [0] * redundant
        submit['OBJECTIVES'] = list(prediction[:, 1]) + [0] * redundant
        submit['METHODS'] = list(prediction[:, 2]) + [0] * redundant
        submit['RESULTS'] = list(prediction[:, 3]) + [0] * redundant
        submit['CONCLUSIONS'] = list(prediction[:, 4]) + [0] * redundant
        submit['OTHERS'] = list(prediction[:, 5]) + [0] * redundant
    else:
        submit['BACKGROUND'] = [0] * redundant + list(prediction[:, 0])
        submit['OBJECTIVES'] = [0] * redundant + list(prediction[:, 1])
        submit['METHODS'] = [0] * redundant + list(prediction[:, 2])
        submit['RESULTS'] = [0] * redundant + list(prediction[:, 3])
        submit['CONCLUSIONS'] = [0] * redundant + list(prediction[:, 4])
        submit['OTHERS'] = [0] * redundant + list(prediction[:, 5])
    df = pd.DataFrame.from_dict(submit)
    df.to_csv(filename, index=False)


# TODO:implement Run_Predict function
# TODO: dataset include some words that Glove don't have! maybe we should pre-trained our owen word embedding!
def get_glove_matrix(word_dict, wordvector_path, embedding_dim):
    embeddings_index = {}
    f = open(wordvector_path)
    for line in f:
        values = line.split()
        token = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[token] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    max_words = Tokenizer.vocab_size()
    embedding_matrix = np.random.randn(max_words, embedding_dim)
    for token, index in word_dict.items():
        embedding_vector = embeddings_index.get(token)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
        # else:
        #     print('Found a unknown word!')    
    return embedding_matrix

if __name__ == '__main__':
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(
            VALID_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        os.system('python3 gendata.py')
    # ---------------- Hyperparameter setting -------------------------
    hparams_path = os.path.join(CWD, 'hparams_setting.json')
    with open(hparams_path, 'r') as f:
        hyper_params = f.read()
    obj = json.loads(hyper_params)
    expname = obj['expname']
    embedding_dim = obj['embedding_dim']
    hidden_dim = obj['hidden_dim']
    lrate = obj['lrate']
    max_epoch = obj['max_epoch']
    batch = obj['batch']
    drop_pb = obj['drop_pb']
    layers = obj['RNN_layers']
    bidirect = obj['bidirect'] 
    # ---------------- Hyperparameter setting -------------------------
    train = pd.read_csv(TRAIN_DATA_PATH)
    valid = pd.read_csv(VALID_DATA_PATH)
    test = pd.read_csv(TEST_DATA_PATH)

    train['Abstract'] = train['Abstract'].apply(func=SplitSent)
    valid['Abstract'] = valid['Abstract'].apply(func=SplitSent)
    test['Abstract'] = test['Abstract'].apply(func=SplitSent)
    GenDict(train, valid)

    # encode 'Abstract' and convert label to one_hot
    print('Start encoding train, valid, test dataset')
    encode_data(train)
    encode_data(valid)
    encode_data(test)
    print('Encoding process complete!')
    trainset = Abstract(data=train, pad_idx=PAD_TOKEN_ID, eos_id=EOS_TOKEN_ID)
    validset = Abstract(data=valid, pad_idx=PAD_TOKEN_ID, eos_id=EOS_TOKEN_ID)
    testset = Abstract(data=test, pad_idx=PAD_TOKEN_ID, eos_id=EOS_TOKEN_ID)
    embedding_matrix = torch.FloatTensor(get_glove_matrix(Tokenizer.get_token_to_id(), 'glove/glove.6B.300d.txt', embedding_dim))
    # -----------------------Model configuration----------------------------
    model = GRUNet(vocab_size=Tokenizer.vocab_size(),
                   embedding_dim=embedding_dim,
                   embedding_matrix=embedding_matrix,
                   hidden_dim=hidden_dim,
                   layer_num=layers,
                   drop_pb=drop_pb,
                   bidirect=bidirect)
    opt = torch.optim.AdamW(model.parameters(), lr=lrate)
    criteria = torch.nn.BCELoss()
    model.to(DEVICE)
    # -----------------------Model configuration----------------------------

    # -----------------------Tensorboard configuration----------------------
    tf_path = os.path.join(CWD, 'test_experiment')
    if not os.path.exists(tf_path):
        os.mkdir(tf_path)
    writer = SummaryWriter(os.path.join(tf_path, expname))
    # -----------------------Tensorboard configuration----------------------
    history = {'train_F1':[], 'train_loss':[], 'valid_F1':[], 'valid_loss':[]}
    for epoch in range(max_epoch):
        print(f'Epoch:{epoch}')
        Run_Epoch(epoch, 'train', model, criteria, opt, trainset, batch, writer, history)
        Run_Epoch(epoch, 'valid', model, criteria, opt, validset, batch, writer, history)
        Save(epoch, model, history, expname)
    # TODO : find best model epoch and run predicttion
    # TODO : Submit Result
    hparams = {
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'learning_rate': lrate,
        'epoch': max_epoch,
        'batch_size': batch,
        'drop': drop_pb,
        'GRU_Layer': layers,
        'bidirect' : bidirect
    }
    for key, value in history.items():
        history[key] = max(value)
        # print(history[key])
        # input('Break')
    writer.add_hparams(hparams, history)
    writer.close()
