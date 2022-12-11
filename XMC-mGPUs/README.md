# MHE for XMC (multi-GPUs version)

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
data_name = eurlex4k
data_path = **
model_path = **
CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7,8 python -m torch.distributed.launch 
    --nproc_per_node=5 --use_env src/main.py
    --dataset $data_name 
    --data_path $data_path 
    --model_path $model_path 
    --lr 1e-4 
    --epoch 5 
    --use_swa 
    --swa_warmup_epoch 1 
    --swa_step 10000 
    --batch 16 
    --eval_step 10000
    --num_group 172 
```
Note that this version has slightly reduced performance compared to the single GPU [XMC](https://github.com/Anoise/MHE/blob/main/XMC) version, and we will continue to update this version to bridge this gap.

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
- Please refer to our [XMC](https://github.com/Anoise/MHE/blob/main/XMC) for single-GPU version.
