# AICUP2019
Repository for AICUP2019 Abstract Labeling test
# How to use
Type the following command to train and evaluate model
``` bash
python3 main.py -cfname={your config_filename}
```
Note that you should provide different config filename between every experiment.  
It can help you to log different model hyperparameter and model status etc.  
This program also provide the following options to help you fine tune your model:
``` bash
python3 main.py -cfname={config_filename} -ebd={embedding_dimension} 
-hid={hidden_dimension} -lrate={learning rate} -mepoch={max epoch} -bsize={batch size} 
-drop={dropout probability} -lnum={num of GRU layer}
```
# Tensorboard
## Local tensorboard server
Type the following command to start tensorboard  
``` bash
tensorboard --logdir=test_experiment
```
## Use tensorboard remotely
#### Launch Tensorboard on your remote machine
```
tensorboard --logdir=path/to/log/dir --port={which port you want to use}
```
#### set up ssh port forwarding
set up ssh port forwarding to one of your unused local ports  
``` bash
ssh -Nfl localhost:xxxx:localhost:{remote tfboard port} user@remote
```
# Model Ver 4 info
重新對資料做 pre processing 以及採用新的模型架構。

### TO-DO
- [ ] Data Pre Processing
- [ ] Re Contruct Model ( Prefer GRU + Linear )
- [ ] Fine tuned Model
- [ ] Use *Transformers ( self attention )*
- [ ] Fine tuned Model  
