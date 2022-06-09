# MHP for CIFAR

## Training

```python
python run_mhp_cifar.py 
    --dataset c100 
    --data-path ../../Data/cifar100  
    --epochs 200 --batch-size 256  
    --num-classes 10 10 
    --save-path checkpoint
```
- Model was trained with Python 3.7 with CUDA 10.X.
- Model should work as expected with pytorch >= 1.7 support was recently included.

Note that the hyperparameter "num-classes" is the factorization of the total number of categories.
