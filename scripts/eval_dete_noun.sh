CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=23456 --nproc_per_node=1 --use_env main.py \
--dataset_config configs/tdod.json \
--valid_batch_size 8  \
--num_workers 5 \
--resume /path/to/checkpoint  \
--ema --eval \
--output-dir 'logs/test' \
--no_contrastive_align_loss \
--verb_noun_input
