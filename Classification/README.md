# MHCE for CIFAR

## Training
Clone the code repository
```git
git clone git@github.com:liangdaojun/MHCE.git
```

### Multi-Head Product (MHP)
Go to the directory "MHCE/MHP-CIFAR", and run
```python
python run_mhp_cifar.py 
    --dataset c100 
    --data-path ../../Data/cifar100  
    --epochs 200
    --batch-size 256  
    --num-classes 10 10 
    --save-path checkpoint_mhp
```
Go to the directory "MHCE/MHE-CIFAR", and run
### Multi-Head Embedding (MHE)
```python
python run_mhs_cifar.py 
    --dataset c100 
    --data-path ../../Data/cifar100  
    --epochs 200
    --batch-size 256  
    --num-classes 10 10 
    --save-path checkpoint_mhe
```

Go to the directory "MHCE/MHS-CIFAR", and run
### Multi-Head Sampling (MHS)
```python
python run_mhs_cifar.py 
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

