# MHS-Arcface

we consider the model pretraining method with XLC tasks such as image classification, face recognition, Natural Language Generation (NLG), etc. On these tasks, the classifier head of the pretrained model is discarded when the training is completed, and only the features extracted by the model are used for fine-tuning on downstream tasks. For the pretraining model, all parameters of the weights in the classifier can be trained to extract more discriminative features, but training all parameters of classifier in the XLC tasks is computationally expensive. Therefore, MHS is proposed to update the model parameters by selecting the classifier head where ground truth labels are located.

## Declare and Requirements
The code repository is based on [insightface](https://github.com/deepinsight/insightface), please refer to it to complete the whole configuration. Here, the minimal configuration can be done via
```shell script
 pip install -r requirement.txt
```

## Datasets

- [WebFace42M](https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view) (2M IDs, 42.5M images)
- [MS1MV2](https://drive.google.com/file/d/1SXS4-Am3bsKSK615qbYdbA_FMVh3sAvR/view) (85K ids/5.8M images) 
- [MS1MV3](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-retinaface) (93k IDs, 5.2M images)

## Pretrained Model
The pretrained model has refer to [insightface](https://github.com/deepinsight/insightface), and can be found at [Baidu Yun Pan](https://pan.baidu.com/s/1CL-l4zWqsI1oDuEEYVhj-g): e8pw and [OneDrive](https://1drv.ms/u/s!AswpsDO2toNKq0lWY69vN58GR6mw?e=p9Ov5d).


## Training

To train a model, run `train.py` with the path to the configs.   

### 1. To run on a machine with 8 GPUs:

```shell
python -m torch.distributed.launch 
    --nproc_per_node=8 
    --nnodes=1 
    --node_rank=0 
    --master_addr="127.0.0.1" 
    --master_port=12581 
    train.py configs/test_webface_r18_lr02
```

### 2. To run on 2 machines with 8 GPUs each:

Node 0:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="ip1" --master_port=12581 train.py configs/test_webface_r18_lr02
```

Node 1:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="ip1" --master_port=12581 train.py configs/test_webface_r18_lr02
```

## Testing  

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

## Performance

<img src="https://github.com/liangdaojun/MHE/blob/main/Images/MHE-Face.jpg">
