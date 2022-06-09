# MHCE
Multi-Head Combination Encoding for Extreme Label Classification

The code will coming soon!

## Abstract
Aiming at solving the eXtreme Label Classification (XLC) problem, in this work, we first analyze the impact of the low-rank weight on the performance of the classifier and then demonstrate that nonlinear operations in the classifier make it less dependent on the rank of the weights. And then, three kinds of Multi-Head Combination Encoding (MHCE) methods are proposed. These methods decompose the original classifier into multi-head classifiers, and the contents and positions of all multi-head classifiers are combined to represent the contents of the original classifier. There are various combinations of multi-head classifiers, including Multi-Head Product (MHP), Multi-Head Embedding (MHE) and Multi-Head Sampling (MHS). MHP uses the Kronecker product to approximate the original classifier. MHE utilizes multiple classifier heads in series to obtain candidate labels from coarse to fine. MHS samples part of the classifier heads for training to reduce the complexity of the computation. Since there are no assumptions made on the label space in obtaining these methods, they can be flexibly applied to different XLC tasks such as image classification, face recognition, eXtreme Multi-label Classification (XMC) and Neural Machine Translation (NMT). Because the original classifier is approximated by multi-head classifiers, the computational complexity is decreased by a polynomial order, which in turn greatly increases the representation ability of XLC. To validate the performance of the proposed methods, experiments are done on multiple tasks such as image classification, face recognition, XMC and NMT. The results show that these methods outperform or approach the performance of existing state-of-the-art methods while greatly simplifying the training and inference process of the model, which validates that the proposed methods can effectively solve the XLC problem in machine learning.

<img src="https://github.com/liangdaojun/MHCE/blob/master/images/MHCE.jpg">

## 

