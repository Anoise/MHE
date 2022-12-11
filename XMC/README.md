# MHE for XMC

## Introduction
MHE-XMC makes no assumptions about the label space and does not use techniques such as HLT and label clustering for preprocessing. This suggests that additional, complex preprocessing and training tricks are not critical for the XMC task, and using simple label partitioning techniques are sufficient to process the XMC task.

## Preparation
The used datasets are download from 
* [EUR-Lex](https://drive.google.com/open?id=1iPGbr5-z2LogtMFG1rwwekV_aTubvAb2)
* [Wiki10-31K](https://drive.google.com/open?id=1Tv4MHQzDWTUC9hRFihRhG8_jt1h0VhnR)
* [AmazonCat-13K](https://drive.google.com/open?id=1VwHAbri6y6oh8lkpZ6sSY_b1FRNnCLFL)
* [Wiki-500K](https://drive.google.com/open?id=1bGEcCagh8zaDV0ZNGsgF0QtwjcAm0Afk)
* [Amazon-670K](https://drive.google.com/open?id=1Xd4BPFy1RPmE7MEXMu77E2_xWOhR1pHW)
* [Amazon3M](https://drive.google.com/open?id=187vt5vAkGI2mS2WOMZ2Qv48YKSjNbQv4) 

The pretrained model, including bert, roberta and xlnet, can be download from [Huggingface](https://huggingface.co).

## Quickly Start
When the dataset and the pretrained model are download, you can quickly run MHE-XMC by
```shell script
data_name = **
data_path = **
model_path = **
python src/main.py 
    --dataset $data_name 
    --data_path $data_path 
    --bert_path $model_path  
    --lr 1e-4 --epoch 20  
    --swa --swa_warmup 2
    --swa_step 100 
    --batch 16
    --num_group 172 
```
Note that when 'num_group' greater than 0, MHE-XMC use MHE for the XMC task. Otherwise, MHE-XMC is the simple multi-label classification method. See script 'run.sh' for detail setting.


## Training and Testing
Clone the code repository
```git
git clone git@github.com:Anoise/MHE.git
```

and go to the directory "MHE/XMC", run
```bash
bash run.sh [eurlex4k|wiki31k|amazon13k|amazon670k|wiki500k]
```

Note that:
- Model was trained with Python 3.7 with CUDA 10.X.
- Model should work as expected with pytorch >= 1.7 support was recently included.
- The hyperparameter "num_group" is the factorization of the total number of categories, which can be greater than the number of categories.
- The code partly refer to [LightXML](https://github.com/kongds/LightXML).

- Please refer to our [XMC-mGPUs](https://github.com/liaingdaojun/MHE/XMC-mGPUs) for multi-GPUs version.

## Performance

<img src="https://github.com/Anoise/MHE/blob/main/Images/MHE-XMC1.jpg">

<img src="https://github.com/Anoise/MHE/blob/main/Images/MHE-XMC2.jpg">
