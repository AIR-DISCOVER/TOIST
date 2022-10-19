CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port=23469 --nproc_per_node=1 --use_env visualize.py \
--dataset_config configs/tdod.json --valid_batch_size 1  \
--num_workers 0 \
--output-dir '/DATA2/lpf/tdod/mdetr/eval_logs/test' \
--resume /DATA2/lpf/tdod/mdetr/logs/test0421_0/BEST_checkpoint.pth  \
--ema --eval \
--no_contrastive_align_loss \
--no_contrastive_inbatch_loss \
--no_contrastive_outbatch_loss \
--mask_model smallconv \
--no_nsthl2_loss
# --cluster \


