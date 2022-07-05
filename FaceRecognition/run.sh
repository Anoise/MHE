
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12581 train.py configs/test_webface_r18_lr02

CUDA_VISIBLE_DEVICES=0, python eval_ijbc.py --model-prefix work_dirs/test_webface_r18_lr02_fc01/model.pt --image-path /home/user/Data/ijb/IJBB --result-dir work_dirs/ijb_test_results --network r18 --target IJBB

CUDA_VISIBLE_DEVICES=0, python eval_ijbc.py --model-prefix work_dirs/test_webface_r18_lr02/model.pt --image-path /home/user/Data/ijb/IJBC --result-dir work_dirs/ijb_test_results --network r18



