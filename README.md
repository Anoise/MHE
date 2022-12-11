# Multi-Head Encoding (MHE) for Extreme Label Classification

<img src="https://github.com/Anoise/MHE/blob/main/Images/intro.png">

## Introduction

  - An MHE mechanism is proposed to solve the parameter overweight problem in XLC tasks, and its parameters are geometrically reduced while the representation ability is theoretically analyzed.
  - The low-rank approximation problem is generalized from the Frobenius-norm metric to the CE metric, and it is found that nonlinear operations can greatly reduce the classifier's dependence on the rank of its weights.
  - Three MHE-based methods are designed to apply different XLC tasks from a unified perspective, and experiment results reveal that these three methods achieve SOTA performance and provide strong benchmarks.
  - MHE can arbitrarily partition the label space, making it flexibly applicable to any XLC task, including image classification, face recognition, XMC and neural machine translation (NMT), etc.
  - MHC has no restriction on the label space and abandons techniques such as HLT and label clustering, thus greatly simplifies the training and inference process of the model on XMC tasks.

<img src="https://github.com/Anoise/MHE/blob/main/Images/arch.png">

## Classification

## MHE for CIFAR

## Training
Clone the code repository
```git
git clone git@github.com:Anoise/MHE.git
```

### Multi-Head Product (MHP)
Go to the directory "MHE/Classification", and run
```python
python MHP-CIFAR/run_mhp_cifar.py 
    --dataset c100 
    --data-path ../../Data/cifar100  
    --epochs 200
    --batch-size 256  
    --num-classes 10 10 
    --save-path checkpoint_mhp
```

### Multi-Head Cascade (MHE)
Go to the directory "MHE/Classification", and run
```python
python MHC-CIFAR/run_mhc_h2.py 
    --dataset c100 
    --data-path ../../Data/cifar100  
    --epochs 200
    --batch-size 256  
    --num-classes 10 10 
    --save-path checkpoint_mhc
```
For head=3, run
```python
python MHC-CIFAR/run_mhc_h3.py 
    --dataset c100 
    --data-path ../../Data/cifar100  
    --epochs 200
    --batch-size 256  
    --num-classes 4 5 5
    --save-path checkpoint_mhc
```

### Multi-Head Sampling (MHS)
Go to the directory "MHE/Classification", and run
```python
python MHS-CIFAR/run_mhs_cifar.py 
    --dataset c100 
    --data-path ../../Data/cifar100  
    --epochs 200
    --batch-size 256  
    --num-classes 10 10 
    --save-path checkpoint_mhs
```

Note that:
- Model was trained with Python 3.7 with CUDA 10.X.
- Model should work as expected with pytorch >= 1.7 support was recently included.
- The hyperparameter "num-classes" is the factorization of the total number of categories, which can be greater than the number of categories.



---

## MHE for ImageNet
The code repository for training ImageNet refers to [Pytorch](https://pytorch.org).

Go to the directory "MHE/Classification", and run
```python
python MHE-ImageNet/[main_mhp.py or main_mhc.py or main_mhs.py]
    -a resnet50 
    --data [your ImageNet data path]
    --dist-url 'tcp://127.0.0.1:6006' 
    --dist-backend 'nccl' 
    --multiprocessing-distributed 
    --world-size 1 
    --rank 0 [imagenet-folder with train and val folders]
    --epochs 100
    --batch-size 256  
    --num-classes 40 25 
```

## Testing

<img src="https://github.com/Anoise/MHE/blob/main/Images/MHE_Classification.jpg">

Please refer to [Classification](https://github.com/Anoise/MHE/tree/main/Classification) for MHE on ImageNet and CIFAR datasets for details.

---
## MHE for XMC

### Preparation
The used datasets are download from 
* [EUR-Lex](https://drive.google.com/open?id=1iPGbr5-z2LogtMFG1rwwekV_aTubvAb2)
* [Wiki10-31K](https://drive.google.com/open?id=1Tv4MHQzDWTUC9hRFihRhG8_jt1h0VhnR)
* [AmazonCat-13K](https://drive.google.com/open?id=1VwHAbri6y6oh8lkpZ6sSY_b1FRNnCLFL)
* [Wiki-500K](https://drive.google.com/open?id=1bGEcCagh8zaDV0ZNGsgF0QtwjcAm0Afk)
* [Amazon-670K](https://drive.google.com/open?id=1Xd4BPFy1RPmE7MEXMu77E2_xWOhR1pHW)
* [Amazon3M](https://drive.google.com/open?id=187vt5vAkGI2mS2WOMZ2Qv48YKSjNbQv4) 

The pretrained model, including bert, roberta and xlnet, can be download from [Huggingface](https://huggingface.co).

### Quickly Start
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


### Training and Testing
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

## MHE for XMC (multi-GPUs version)

### Quickly Start
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

### Training and Testing
Clone the code repository
```git
git clone git@github.com:Anoise/MHE.git
```

and go to the directory "MHE/XMC", run
```bash
bash run.sh [eurlex4k|wiki31k|amazon13k|amazon670k|wiki500k]
```

### Performance

<img src="https://github.com/Anoise/MHE/blob/main/Images/MHE-XMC1.jpg">

<img src="https://github.com/Anoise/MHE/blob/main/Images/MHE-XMC2.jpg">

Note that:
- Model was trained with Python 3.7 with CUDA 10.X.
- Model should work as expected with pytorch >= 1.7 support was recently included.
- The hyperparameter "num_group" is the factorization of the total number of categories, which can be greater than the number of categories.

- Please refer to [XMC](https://github.com/Anoise/MHE/tree/main/XMC) for MHC on EUR-Lex, Wiki10-31K,AmazonCat-13K, Wiki-500K, Amazon-670K, Amazon3M datasets.

- Please refer to [XMC-mGPUs](https://github.com/Anoise/MHE/tree/main/XMC-mGPUs) for MHC on multi-GPUs.

---
## Face Recognition (MHS-Arcface)

### Declare and Requirements
The code repository is based on [insightface](https://github.com/deepinsight/insightface), please refer to it to complete the whole configuration. Here, the minimal configuration can be done via
```shell script
 pip install -r requirement.txt
```

### Datasets

- [WebFace42M](https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view) (2M IDs, 42.5M images)
- [MS1MV2](https://drive.google.com/file/d/1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR/view) (85K ids/5.8M images) 
- [MS1MV3](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-retinaface) (93k IDs, 5.2M images)

### Pretrained Model
The pretrained model has refer to [insightface](https://github.com/deepinsight/insightface), and can be found at [Baidu Yun Pan](https://pan.baidu.com/s/1CL-l4zWqsI1oDuEEYVhj-g): e8pw and [OneDrive](https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d).


### Training

To train a model, run `train.py` with the path to the configs.   

#### 1. To run on a machine with 8 GPUs:

```shell
python -m torch.distributed.launch 
    --nproc_per_node=8 
    --nnodes=1 
    --node_rank=0 
    --master_addr="127.0.0.1" 
    --master_port=12581 
    train.py configs/test_webface_r18_lr02
```

#### 2. To run on 2 machines with 8 GPUs each:

Node 0:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="ip1" --master_port=12581 train.py configs/test_webface_r18_lr02
```

Node 1:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="ip1" --master_port=12581 train.py configs/test_webface_r18_lr02
```

### Testing  

Testing on [IJB-B](https://drive.google.com/file/d/1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o/view) 
```shell
CUDA_VISIBLE_DEVICES=0, python eval_ijbc.py 
    --model-prefix work_dirs/test_webface_r18_lr02_fc01/model.pt 
    --image-path /home/user/Data/ijb/IJBB 
    --result-dir work_dirs/ijb_test_results 
    --network r18 
    --target IJBB
```

Testing on [IJB-C](https://drive.google.com/file/d/1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o/view)

```shell
CUDA_VISIBLE_DEVICES=0, python eval_ijbc.py 
    --model-prefix work_dirs/test_webface_r18_lr02/model.pt 
    --image-path /home/user/Data/ijb/IJBC 
    --result-dir work_dirs/ijb_test_results 
    --network r18
```

### Performance

<img src="https://github.com/Anoise/MHE/blob/main/Images/MHE-Face.jpg">

Please refer to [FaceRecognition](https://github.com/Anoise/MHE/tree/main/FaceRecognition) for MHS pretrained on WebFace and MS1MV datasets.

---
## Citations
come soon!
<!--
```
@inproceedings{deng2019arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4690--4699},
  year={2019}
}

```
-->
