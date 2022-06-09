# MHCE for CIFAR

## Training
Clone the code repository
```git
git clone git@github.com:liangdaojun/MHCE.git
```

### Multi-Head Product (MHP)
Go to the directory "MHCE/Classification", and run
```python
python MHP-CIFAR/run_mhp_cifar.py 
    --dataset c100 
    --data-path ../../Data/cifar100  
    --epochs 200
    --batch-size 256  
    --num-classes 10 10 
    --save-path checkpoint_mhp
```

### Multi-Head Embedding (MHE)
Go to the directory "MHCE/Classification", and run
```python
python MHE-CIFAR/run_mhe_h2.py 
    --dataset c100 
    --data-path ../../Data/cifar100  
    --epochs 200
    --batch-size 256  
    --num-classes 10 10 
    --save-path checkpoint_mhe
```
For head=3, run
```python
python MHE-CIFAR/run_mhe_h3.py 
    --dataset c100 
    --data-path ../../Data/cifar100  
    --epochs 200
    --batch-size 256  
    --num-classes 4 5 5
    --save-path checkpoint_mhe
```

### Multi-Head Sampling (MHS)
Go to the directory "MHCE/Classification", and run
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

## Testing

<img src="https://github.com/liangdaojun/MHCE/blob/main/Images/MHCE_Classification.jpg">

---

# MHCE for ImageNet
The code repository for training ImageNet refers to Pytorch[https://pytorch.org].

