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

### Data Preprocessing TO-DO
- [x] NLTK 是否 support 過濾掉 '(' ')' ',' '.' '\' 等符號？
  - [x] 嘗試 NLTK 中的 RegexpTokenizer
    - RegexpTokenizer(r'\w+') 可以過濾所有非字母與非數字字元 但會過濾掉 $ 
- [x] 已知 $$$ 斷句符號會被 NLTK tokenizer 斷成 $ $ $ 是否有其他替代的斷句符號(e.x 只使用 '$' ?)
  - 可將斷句符號設定為 'E_O_S' 
- [ ] Remove StopWords
- [ ] 紀錄斷句符號位置
- [ ] 建立 word vocabulary
