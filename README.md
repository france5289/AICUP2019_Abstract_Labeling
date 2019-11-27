# AICUP2019
Repository for AICUP2019 Abstract Labeling test
# How to use
Type the following command to train and evaluate model
``` bash
python3 train.py
```
# Release Note
11/26 新增以下功能:
- 將hyperparameters 寫於 `hparams_setting.json` training 時自動讀取
- 將hyperparameters 紀錄於 `tensorboard`
# Note
**現在暫時不強迫config檔名不能重複 做實驗時請注意**   
**目前還沒有實做 Prediction function**

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
- [x] ~~Data Pre Processing~~
- [x] ~~使用 tensorboard 紀錄 loss / f1 並作圖~~
- [x] ~~Re Contruct Model ( Prefer GRU + Linear )~~
- [x] ~~分析 train valid dataset 的詞頻~~
  - ~~另外開一個 jupyter notebook ~~
- [ ] Use 300d Glove word embedding
- [ ] 完成 Prediction function 與 Submit result
- [x] ~~Use Glove Pre trained word embedding~~  
- [x] ~~使用 tensorboard 紀錄 hyperparameter~~
- [ ] Fine tuned Model
- [ ] Use *Transformers ( self attention )*
- [ ] Fine tuned Model  

## 開發計畫
1. 先處理 Glove word embedding
2. 重新整理 code
   1. 簡化 code
   2. ~~讓tensorboard紀錄更多資訊~~  
`以上預計都在 modelVer4.0 branch 中實驗`
3. ~~Merge master branch and modelVer4 branch~~ 
4. Fine tuned model
5. Use *Transformers layer* 