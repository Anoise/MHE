#!/bin/bash

CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7,8 python -m torch.distributed.launch --nproc_per_node=5 --use_env src_v6_group_copy/main_amp_v3.py --lr 1e-4 --epoch 5 --dataset $1 --use_swa --swa_warmup_epoch 1 --swa_step 10000 --batch 16 --eval_step 10000

