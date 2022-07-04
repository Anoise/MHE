#!/bin/bash
set -e



# python ./src/cluster.py --dataset amazon670k --id $1
# CUDA_VISIBLE_DEVICES=0, python src_v6_group/main_amp_v3.py --lr 1e-4 --epoch 15 --dataset amazon670k --batch 16 --max_len 128 --eval_step 3000  --candidates_topk 75 --hidden_dim 512 --num_group 8200
        
CUDA_VISIBLE_DEVICES=0,2,3,4,5, python -m torch.distributed.launch --nproc_per_node=5 --use_env src_v6_group_copy/main_amp_v3.py --lr 1e-4 --epoch 5 --dataset amazoncat13k --use_swa --swa_warmup_epoch 1 --swa_step 10000 --batch 16 --eval_step 10000
#python src_v6_group_copy/main_amp_v3.py --lr 1e-4 --epoch 5 --dataset amazoncat13k --swa --swa_warmup 1 --swa_step 10000 --batch 16 --eval_step 10000 --bert roberta
#python src_v6_group_copy/main_amp_v3.py --lr 1e-4 --epoch 5 --dataset amazoncat13k --swa --swa_warmup 1 --swa_step 10000 --batch 32 --eval_step 10000 --bert xlnet --max_len 128

#CUDA_VISIBLE_DEVICES=6, python src_v6_group_copy/main_amp_v3.py --lr 1e-4 --epoch 5 --dataset amazoncat13k --batch 16 --eval_step 3000  --candidates_topk 40 --valid  --hidden_dim 512 --num_group 8200 --eval_model --dropout 0.1 --model roberta  #--resume

# python src/main.py --lr 1e-4 --epoch 15 --dataset amazon670k --swa --swa_warmup 4 --swa_step 3000 --batch 16 --max_len 128 --eval_step 3000 --group_y_candidate_num 2000 --group_y_candidate_topk 75 --valid  --hidden_dim 400 --group_y_group $i --eval_model

# python src/ensemble_direct.py --model1 amazon670k_t0 --model2 amazon670k_t1 --model3 amazon670k_t2 --dataset amazon670k
