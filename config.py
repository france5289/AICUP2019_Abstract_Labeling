
import json
import os

from baseconfig import BaseConfig


class Config(BaseConfig):
    '''
    Use this object to load hyperparameters from json files
    '''
    def __init__(self):
        self.expname = 'Default'
        self.seed = 20
        self.embedding_dim = 0
        self.hidden_dim = 0
        self.learning_rate = 0.0
        self.max_epoch = 0
        self.batch_size = 0
        self.pretrained_embedding_path = ''
        self.removeOOV = False
        self.removepunct = False
    
    #======================
    # Getter
    @property
    def expname(self):
        return self._expname
    
    @property
    def seed(self):
        return self._seed

    @property
    def embedding_dim(self):
        return self._embedding_dim
    
    @property
    def hidden_dim(self):
        return self._hidden_dim
    
    @property
    def learning_rate(self):
        return self._learning_rate
    
    @property
    def max_epoch(self):
        return self._max_epoch
    
    @property
    def batch_size(self):
        return self._batch_size

    @property
    def pretrained_embedding_path(self):
        return self._pretrained_embedding_path
    
    @property 
    def removeOOV(self):
        return self._removeOOV
    
    @property
    def removepunct(self):
        return self._removepunct

    #======================
    # Setter
    @expname.setter
    def expname(self, value):
        if type(value) is not str:
            raise ValueError('expname should be a string!')
        self._expname = value

    @seed.setter
    def seed(self, value):
        if type(value) is not int:
            raise ValueError('Random seed should be an integer')
        self._seed = value

    @embedding_dim.setter
    def embedding_dim(self, value):
        if type(value) is not int:
            raise ValueError('embedding_dim should be an integer')
        self._embedding_dim = value

    @hidden_dim.setter
    def hidden_dim(self, value):
        if type(value) is not int:
            raise ValueError('hidden_dim should be an integer')
        self._hidden_dim = value
    
    @learning_rate.setter
    def learning_rate(self, value):
        if type(value) is not float:
            raise ValueError('learning_rate should be a float')
        self._learning_rate = value
    
    @max_epoch.setter
    def max_epoch(self, value):
        if type(value) is not int:
            raise ValueError('max_epoch should be an integer')
        self._max_epoch = value
    
    @batch_size.setter
    def batch_size(self, value):
        if type(value) is not int:
            raise ValueError('batch_size should be an integer')
        self._batch_size = value
    
    @pretrained_embedding_path.setter
    def pretrained_embedding_path(self, value):
        if type(value) is not str:
            raise ValueError('pretrained_embedding_path should be a string')
        self._pretrained_embedding_path = value
    
    @removeOOV.setter
    def removeOOV(self, value):
        if type(value) is not bool:
            raise ValueError('removeOOV should be a boolean')
        self._removeOOV = value
    
    @removepunct.setter
    def removepunct(self, value):
        if type(value) is not bool:
            raise ValueError('removepunct should be a boolean')
        self._removepunct = value

    #======================

    @classmethod
    def load_from_json(cls, jsonpath):
        '''
        Factory method to read hyperparameter settings from json files
        and return a new class
        
        Args:
        --------
            jsonpath(string): json file path
        
        Return:
        --------
            self(Config obeject)
        '''
        self = cls()
        
        if jsonpath is None or type(jsonpath) != str:
            raise ValueError('Arg `jsonpath` should be a string!')
        if not os.path.exists(jsonpath):
            raise FileNotFoundError(f'File {jsonpath} does not exits')

        print('Read Hyperparameter setting')
        with open(jsonpath, 'r') as f:
            hyper_params = f.read()
        obj = json.loads(hyper_params)
        # ============ set hyperparameters ===============
        self.expname = obj['expname']
        self.seed = obj['seed']
        self.embedding_dim = obj['embedding_dim']
        self.hidden_dim = obj['hidden_dim']
        self.learning_rate = obj['learning_rate']
        self.max_epoch = obj['max_epoch']
        self.batch_size = obj['batch_size']
        self.pretrained_embedding_path = obj['pretrained_path']
        self.removeOOV = obj['removeOOV']
        self.removepunct = obj['removepunct']
        print('All hyperparameters have been read in')
        return self