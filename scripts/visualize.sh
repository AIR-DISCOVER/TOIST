CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=23467 --nproc_per_node=1 --use_env visualize.py \
--dataset_config configs/tdod.json --valid_batch_size 1  \
--num_workers 0 \
--output-dir 'logs/test' \
--resume /path/to/checkpoint  \
--ema --eval \
--no_contrastive_align_loss \
--mask_model smallconv
