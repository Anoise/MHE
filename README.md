# Multi-Head Encoding (MHE) for Extreme Label Classification



## Introduction

  - An MHE mechanism is proposed to solve the parameter overweight problem in XLC tasks, and its parameters are geometrically reduced while the representation ability is theoretically analyzed.
  - The low-rank approximation problem is generalized from the Frobenius-norm metric to the CE metric, and it is found that nonlinear operations can greatly reduce the classifier's dependence on the rank of its weights.
  - Three MHE-based methods are designed to apply different XLC tasks from a unified perspective, and experiment results reveal that these three methods achieve SOTA performance and provide strong benchmarks.
  - MHE can arbitrarily partition the label space, making it flexibly applicable to any XLC task, including image classification, face recognition, XMC and neural machine translation (NMT), etc.
  - MHC has no restriction on the label space and abandons techniques such as HLT and label clustering, thus greatly simplifies the training and inference process of the model on XMC tasks.

<img src="https://github.com/Anoise/MHCE/blob/main/Images/MHCE.jpg">

## Classification

Please refer to [Classification](https://github.com/Anoise/MHCE/tree/main/Classification) for MHE on ImageNet and CIFAR datasets.

## XMC

- Please refer to [XMC](https://github.com/Anoise/MHCE/tree/main/XMC) for MHC on EUR-Lex, Wiki10-31K,AmazonCat-13K, Wiki-500K, Amazon-670K, Amazon3M datasets.

- Please refer to [XMC-mGPUs](https://github.com/Anoise/MHCE/tree/main/XMC-mGPUs) for MHC on multi-GPUs.

## Face Recognition

Please refer to [FaceRecognition](https://github.com/Anoise/MHCE/tree/main/FaceRecognition) for MHS pretrained on WebFace and MS1MV datasets.


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
