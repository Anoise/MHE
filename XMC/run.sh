#!/bin/bash 

arr=(bert roberta xlnet)

if [ "$1" = "wiki500k" ]; then
    echo start $1
    for str in ${arr[@]};do
        python src/main.py --lr 1e-4 --epoch 10 --dataset wiki500k --swa --swa_warmup 4 --swa_step 3000  --batch 32 --max_len 128 --eval_step 3000  --group_y_candidate_num 2000 --group_y_candidate_topk 32 --valid --hidden_dim 500 --bert $str --num_group 5630

        python src/main.py --lr 1e-4 --epoch 10 --dataset wiki500k --swa --swa_warmup 4 --swa_step 3000  --batch 32 --max_len 128 --eval_step 3000  --group_y_candidate_num 2000 --group_y_candidate_topk 32 --valid --hidden_dim 500 --bert $str --num_group 5630 --eval_model
    done
    python src/ensemble_direct.py --model1 wiki500k_${arr[0]}_5630 --model2 wiki500k_${arr[0]}_5630 --model3 wiki500k_${arr[0]}_5630 --dataset wiki500k

elif [ "$1" = "amazon670k" ]; then
    echo start $1
    for str in ${arr[@]};do
        python src/main.py --lr 1e-4 --epoch 15 --dataset amazon670k --swa --swa_warmup 1 --swa_step 1000 --batch 32 --max_len 128 --eval_step 1000 --group_y_candidate_topk 75 --valid  --hidden_dim 400  --bert $str --num_group 8400
        python src/main.py --lr 1e-4 --epoch 15 --dataset amazon670k --swa --swa_warmup 1 --swa_step 1000 --batch 32 --max_len 128 --eval_step 1000 --group_y_candidate_topk 75 --valid  --hidden_dim 400 --bert $str --num_group 8400  --eval_model
    done
    python src/ensemble_direct.py --model1 amazon670k_${arr[0]}_8400 --model2 amazon670k_${arr[1]}_8400 --model3 amazon670k_${arr[2]}_8400 --dataset amazon670k

elif [ "$1" = "amazon3m" ]; then
    echo start $1
    for str in ${arr[@]};do
        python src/main.py --lr 1e-4 --epoch 5 --dataset amazon3m --swa --swa_warmup 0 --swa_step 10000  --batch 16 --max_len 128 --eval_step 40000  --group_y_candidate_num 2000 --group_y_candidate_topk 30 --hidden_dim 400  --bert $str  --num_group 8761

        python src/main.py --lr 1e-4 --epoch 5 --dataset amazon3m --swa --swa_warmup 0 --swa_step 10000  --batch 16 --max_len 128 --eval_step 40000  --group_y_candidate_num 2000 --group_y_candidate_topk 30 --hidden_dim 400  --bert $str  --num_group 8761 --eval_model
    done
    python src/ensemble_direct.py --model1 amazon3m_${arr[0]}_8761 --model2 amazon3m_${arr[0]}_8761 -model3 amazon3m_${arr[0]}_8761 --dataset amazon3m

elif [ "$1" = "amazoncat13k" ]; then
    echo start $1
    python src/main.py --lr 1e-4 --epoch 5 --dataset amazoncat13k --swa --swa_warmup 2 --swa_step 5000 --batch 16 --eval_step 5000
    python src/main.py --lr 1e-4 --epoch 5 --dataset amazoncat13k --swa --swa_warmup 2 --swa_step 10000 --batch 16 --eval_step 5000 --bert roberta
    python src/main.py --lr 1e-4 --epoch 5 --dataset amazoncat13k --swa --swa_warmup 2 --swa_step 10000 --batch 32 --eval_step 5000 --bert xlnet --max_len 128

    python src/ensemble.py --dataset amazoncat13k
elif [ "$1" = "wiki31k" ]; then
    echo start $1
    python src/main.py --lr 1e-4 --epoch 30 --dataset wiki31k --swa --swa_warmup 4 --swa_step 200 --batch 16
    python src/main.py --lr 1e-4 --epoch 30 --dataset wiki31k --swa --swa_warmup 4 --swa_step 200 --batch 8  --bert xlnet
    python src/main.py --lr 1e-4 --epoch 30 --dataset wiki31k --swa --swa_warmup 4 --swa_step 400 --batch 16 --bert roberta

    python src/ensemble.py --dataset wiki31k
elif [ "$1" = "eurlex4k" ]; then
    echo start $1
    python src/main.py --lr 1e-4 --epoch 20 --dataset eurlex4k --swa --swa_warmup 2 --swa_step 100 --batch 16  
    python src/main.py --lr 1e-4 --epoch 20 --dataset eurlex4k --swa --swa_warmup 2 --swa_step 100 --batch 16  --bert roberta
    python src/main.py --lr 1e-4 --epoch 20 --dataset eurlex4k --swa --swa_warmup 2 --swa_step 200 --batch 8 --update_count 2 --bert xlnet

    python src/ensemble.py --dataset eurlex4k
fi   
