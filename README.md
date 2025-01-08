# [Multi-Head Encoding (MHE) for Extreme Label Classification (TPAMI 2025 Accepted)](https://doi.org/10.1109/TPAMI.2024.3522298)

Paper: [Online](https://doi.org/10.1109/TPAMI.2024.3522298), [Arxiv](https://arxiv.org/pdf/2412.10182)

Intro: [WeiChat Info](https://mp.weixin.qq.com/s/v6SxHjUVENuyAbG0hETESg), [Zhihu](https://zhuanlan.zhihu.com/p/16432360258), [CSDN](https://blog.csdn.net/liangdaojun/article/details/144925237)

## Introductioin

A [Multi-Head Encoding (MHE)](https://arxiv.org/abs/2412.10182) mechanism is proposed to address the parameter overweight problem in Extreme Label Classification (XLC) tasks, which replaces the original classifier with multi-head classifier. During training, the extreme labels are decomposed into multiple short local labels, and each classification head is trained with the local labels. While during testing, the predicted labels are combined based on the local predictions of each classification head. This reduces the computational load geometrically. 

Based on this, three MHE-based training and testing methods are proposed in this paper to cope with the parameter overweight problem in different XLC tasks.
Experimental results, e.g., *extreme sigle-label and multi-label image recognition, extreme  multi-label text classification face recognition, model pre -training, and neural machine translation*, show that the proposed  methods achieve SOTA performance while significantly streamlining the training and inference processes of XLC tasks.

## Important Discovery

- MHE is equivalent to OHE in the single label classification.
<img src="https://github.com/Anoise/MHE/blob/main/Images/intro.png">

- Training a low-rank networks using Cross-Entropy with softmax as the loss function
 can recover the same accuracy as the vanilla classifier, as long as the rank of weight $R([W,B])>1$ is satisfied.
<img src="https://github.com/Anoise/MHE/blob/main/Images/Low_Rank.png">

- The model generalization becomes irrelevant to the semantics of the labels when they overfit the  data.
<img src="https://github.com/Anoise/MHE/blob/main/Images/Converage.png">

- Label preprocessing techniques, e.g., HLT and label clustering, are not necessary since the low rank approximation remains independent of label positioning. This can not only significantly improve the training-inference speed, but also achieve multi-GPU parallel acceleration.  
<img src="https://github.com/Anoise/MHE/blob/main/Images/LC_LRD.png">

## Contributions
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

### Multi-Head Cascade (MHC)
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

Please refer to [Classification](https://github.com/Anoise/MHE/tree/main/Classification) of MHE on ImageNet and CIFAR datasets for more details.

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

The pretrained model, including bert, roberta and xlnet, which can be download from [Huggingface](https://huggingface.co).

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

- Please refer to [XMC](https://github.com/Anoise/MHE/tree/main/XMC) of MHC on EUR-Lex, Wiki10-31K,AmazonCat-13K, Wiki-500K, Amazon-670K, Amazon3M datasets for more details.

- Please refer to [XMC-mGPUs](https://github.com/Anoise/MHE/tree/main/XMC-mGPUs) of MHC on multi-GPUs for more details.

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

Please refer to [FaceRecognition](https://github.com/Anoise/MHE/tree/main/FaceRecognition) of MHS pretrained on WebFace and MS1MV datasets for more details.

---
## Citations

Daojun Liang, Haixia Zhang, Dongfeng Yuan and Minggao Zhang. "[Multi-Head Encoding for Extreme Label Classification](https://arxiv.org/pdf/2412.10182)", IEEE Transactions on Pattern Analysis and Machine Intelligence, TPAMI, 2024.

```
@ARTICLE{liang2024MHE,
  author={Liang, Daojun and Zhang, Haixia and Yuan, Dongfeng and Zhang, Minggao},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Multi-Head Encoding for Extreme Label Classification}, 
  year={2024},
  volume={},
  number={},
  pages={1-14},
  keywords={Training;Encoding;Testing;Computational modeling;Magnetic heads;Tensors;Face recognition;Vocabulary;Vectors;Semantics;EXtreme Multi-label Classification;extreme label classification;multi-head encoding},
  doi={10.1109/TPAMI.2024.3522298}
}
% or ArXiv
@inproceedings{liang2024MHE,
  title={Multi-Head Encoding for Extreme Label Classification},
  author={Daojun, Liang and Haixia, Zhang and Dongfeng, Yuan and Minggao, Zhang},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2024}
}

```
