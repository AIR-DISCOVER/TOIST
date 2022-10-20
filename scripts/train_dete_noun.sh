CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --master_port=23456 --nproc_per_node=6 --use_env main.py \
--dataset_config configs/tdod.json \
--train_batch_size 6  \
--valid_batch_size 8  \
--load /path/to/pretrained_resnet101_checkpoint.pth  \
--ema --text_encoder_lr 1e-5 --lr 5e-5 \
--num_workers 5 \
--output-dir 'logs/test' \
--eval_skip 1 \
--verb_noun_input
