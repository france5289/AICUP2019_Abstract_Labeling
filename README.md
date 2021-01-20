# AICUP2019_Abstract_Labeling
This repository stores the code of **2019 AICUP Artificial Intelligence Analysis and Classification of Thesis (Tagging of Thesis)**
## Competition description
- Develop a deep learning model that can label sentences within the abstract of papers from arXiv
- A sentence could be labeled as Background, Objectives, Methods, Results, Conclusions and Others
- A single sentence can belong to multiple labels

## Insight of training datasets
- There is **unbalance** problem of each label
- So we give **different weights of each label**

![](https://i.imgur.com/wLHEAVz.png)


| Label | #Sentence |
| ----- | --------- |
| Background  | 13,353 |
| Objectives | 9,329 |
| Methods | 13,655 |
| Results | 11,722 |
| Conclusions | 5,313 |
| Others | 901 |

## Data preprocessing
In order to solve OOV problems, we use following methods to preprocess our data:
- Remove URL
- Remove math equations
- Remove punctuation marks
- Replace math number with `[NUM]` token
    - Let each math number map to same embedding
- Replace out-of-vocabulary words with `[UNK]` token

## Pre-trained word embedding
- We fetch about 540,000 papers from arXiv to train word embedding
- We use `fastText` cause it is more robust to OOV words

## Model architecture
![](https://i.imgur.com/qFyFrZn.png)

## Final results



| Place | Team          | Val. F1  |
| ----- | ------------- |:--------:|
| 1     | 公鹿總冠軍    | 0.743458 |
| **15(ours)**    | **NCKUDM_ikmlab** | **0.705703** |




### This Branch stores the last model for submission of the private dataset.
Best model weight download link <br />
https://drive.google.com/file/d/1Wj9wE_u-U114FEW8QJBdigpMgihMRd9_/view?usp=sharing

Best model weight (state) download link <br />
https://drive.google.com/file/d/1h-vac-877Z_fNUNCLvC-pFJh465Ue92Q/view?usp=sharing

Public and private dataset concatenation <br />
https://drive.google.com/file/d/1d1whZ2IafmEcITGR2UooNRInS1cFdOQ1/view?usp=sharing
