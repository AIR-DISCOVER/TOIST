CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --master_port=23456 --nproc_per_node=6 --use_env main.py \
--dataset_config configs/tdod.json \
--train_batch_size 3  \
--valid_batch_size 8  \
--load /path/to/pronoun/detection/checkpoint  \
--load_noun /path/to/noun/detection/checkpoint \
--ema --text_encoder_lr 1e-5 --lr 5e-5 \
--num_workers 5 \
--output-dir 'logs/test' \
--eval_skip 1 \
--softkd_loss \
--softkd_coef 50 \
--distillation \
--cluster \
--cluster_memory_size 1024 \
--cluster_num 3 \
--cluster_feature_loss 1e4
# --without_pretrain
