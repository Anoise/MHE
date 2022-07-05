# Multi-Head Combination Encoding (MHCE) for Extreme Label Classification


## Introduction
Aiming at solving the eXtreme Label Classification (XLC) problem, three kinds of Multi-Head Combination Encoding (MHCE) methods are proposed, including Multi-Head Product (MHP), Multi-Head Embedding (MHE) and Multi-Head Sampling (MHS). MHP uses the Kronecker product to approximate the original classifier. MHE utilizes multiple classifier heads in series to obtain candidate labels from coarse to fine. MHS samples part of the classifier heads for training to reduce the complexity of the computation. Since there are no assumptions made on the label space in obtaining these methods, they can be flexibly applied to different XLC tasks such as image classification, face recognition, eXtreme Multi-label Classification (XMC) and Neural Machine Translation (NMT). Because the original classifier is approximated by multi-head classifiers, the computational complexity is decreased by a polynomial order, which in turn greatly increases the representation ability of XLC. To validate the performance of the proposed methods, experiments are done on multiple tasks such as image classification, face recognition, XMC and NMT. The results show that these methods outperform or approach the performance of existing state-of-the-art methods while greatly simplifying the training and inference process of the model, which validates that the proposed methods can effectively solve the XLC problem in machine learning.

<img src="https://github.com/liangdaojun/MHCE/blob/main/Images/MHCE.jpg">

## Classification

Please refer to [Classification](https://github.com/liangdaojun/MHCE/tree/main/Classification) for MHCE on ImageNet and CIFAR datasets.

## XMC

- Please refer to [XMC](https://github.com/liangdaojun/MHCE/tree/main/XMC) for MHE on EUR-Lex, Wiki10-31K,AmazonCat-13K, Wiki-500K, Amazon-670K, Amazon3M datasets.

- Please refer to [XMC-mGPUs](https://github.com/liangdaojun/MHCE/tree/main/XMC-mGPUs) for MHE on multi-GPUs.

## Face Recognition

Please refer to [FaceRecognition](https://github.com/liangdaojun/MHCE/tree/main/FaceRecognition) for MHS pretrained on WebFace and MS1MV datasets.


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
