'''
this module provide a config writer to log model hyperparameters to a config file
'''
import configparser
import os
import json

def write_config(filename, **kwargs):
    '''
    Pass in hyperparameter value and write to config file \n
    User must give a specified filename! \n
    Args:
        filename(str)
        **kwargs : a set of hyperparameters 
    '''
    assert filename is not '', 'Please give this set of hyperparameters an identifier!' 
    if not os.path.exists('model_config'):
        os.mkdir('model_config')
    assert not os.path.exists(f'model_config/{filename}.json'), 'Please use another file name! This config has already existed!'
    model_config = {'Hyperparameters':{}}
    
    for para, value in kwargs.items():
        model_config['Hyperparameters'].update({para:value})
    
    
    with open(f'model_config/{filename}.json', 'w') as f:
        json.dump(model_config, f, indent=4)

if __name__ == '__main__':
    write_config('test', drop_pb=0.3, batch_size=16)