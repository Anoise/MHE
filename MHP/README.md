# MHP for CIFAR

## Training
Clone the code repository
```git
git clone git@github.com:liangdaojun/MHCE.git
```
Go to directory "MHCE/MHP", and run
```python
python run_mhp_cifar.py 
    --dataset c100 
    --data-path ../../Data/cifar100  
    --epochs 200
    --batch-size 256  
    --num-classes 10 10 
    --save-path checkpoint
```

Note that:
- Model was trained with Python 3.7 with CUDA 10.X.
- Model should work as expected with pytorch >= 1.7 support was recently included.
- The hyperparameter "num-classes" is the factorization of the total number of categories, which can be greater than the number of categories.

# MHP for ImageNet
The code repository for training ImageNet refers to Pytorch[https://pytorch.org].

## Testing

<img src="https://github.com/liangdaojun/MHCE/blob/main/Images/MHCE_Classification.jpg">