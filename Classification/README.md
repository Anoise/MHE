# MHE for CIFAR

## Training
Clone the code repository
```git
git clone git@github.com:liangdaojun/MHE.git
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
python MHC-CIFAR/run_mhe_h3.py 
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

# MHE for ImageNet
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

---
## Testing

<img src="https://github.com/Anoise/MHE/blob/main/Images/MHE_Classification.jpg">
