# Copyright (c) Pengfei Li. All Rights Reserved
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import os
import random
import time
from collections import namedtuple
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.utils
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from transformers import RobertaTokenizerFast

from torch.utils.tensorboard import SummaryWriter
import util.dist as dist
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.coco_eval import TDODCocoEvaluator
from engine import evaluate, train_one_epoch_plain, train_one_epoch_distillation
from models import build_model
from models.postprocessors import build_postprocessors
from IPython import embed

def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--run_name", default="", type=str)

    # Dataset specific
    parser.add_argument("--dataset_config", default=None, required=True)

    parser.add_argument(
        "--combine_datasets", nargs="+", help="List of datasets to combine for training", default=["flickr"]
    )
    parser.add_argument(
        "--combine_datasets_val", nargs="+", help="List of datasets to combine for eval", default=["flickr"]
    )

    # Training hyper-parameters
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--text_encoder_lr", default=5e-5, type=float)
    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--valid_batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--lr_drop", default=35, type=int)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm")
    parser.add_argument(
        "--eval_skip",
        default=1,
        type=int,
        help='do evaluation every "eval_skip" frames',
    )

    parser.add_argument(
        "--schedule",
        default="linear_with_warmup",
        type=str,
        choices=("step", "multistep", "linear_with_warmup", "all_linear_with_warmup"),
    )
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9998)
    parser.add_argument("--fraction_warmup_steps", default=0.01, type=float, help="Fraction of total number of steps")

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )

    parser.add_argument(
        "--freeze_text_encoder", action="store_true", help="Whether to freeze the weights of the text encoder"
    )

    parser.add_argument(
        "--text_encoder_type",
        default="roberta-base",
        choices=("roberta-base", "distilroberta-base", "roberta-large"),
    )

    # Backbone
    parser.add_argument(
        "--backbone",
        default="resnet101",
        type=str,
        help="Name of the convolutional backbone to use such as resnet50 resnet101 timm_tf_efficientnet_b3_ns",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument("--num_queries", default=100, type=int, help="Number of query slots")
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument(
        "--no_pass_pos_and_query",
        dest="pass_pos_and_query",
        action="store_false",
        help="Disables passing the positional encodings to each attention layers",
    )

    # Segmentation
    parser.add_argument(
        "--mask_model",
        default="none",
        type=str,
        choices=("none", "smallconv", "v2"),
        help="Segmentation head to be used (if None, segmentation will not be trained)",
    )
    parser.add_argument("--masks", action="store_true")

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    parser.add_argument(
        "--set_loss",
        default="hungarian",
        type=str,
        choices=("sequential", "hungarian", "lexicographical"),
        help="Type of matching to perform in the loss",
    )

    parser.add_argument("--contrastive_loss", action="store_true", help="Whether to add contrastive loss")
    parser.add_argument(
        "--no_contrastive_align_loss",
        dest="contrastive_align_loss",
        action="store_false",
        help="Whether to add contrastive alignment loss",
    )

    parser.add_argument(
        "--contrastive_loss_hdim",
        type=int,
        default=64,
        help="Projection head output size before computing normalized temperature-scaled cross entropy loss",
    )

    parser.add_argument(
        "--temperature_NCE", type=float, default=0.07, help="Temperature in the  temperature-scaled cross entropy loss"
    )

    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=1,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )
    # Loss coefficients
    parser.add_argument("--ce_loss_coef", default=1, type=float)
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--qa_loss_coef", default=1, type=float)
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )
    parser.add_argument("--contrastive_loss_coef", default=0.1, type=float)
    parser.add_argument("--contrastive_align_loss_coef", default=1, type=float)

    parser.add_argument(
        "--nsthl2_loss",
        action="store_true",
        help="Whether to add noun&sth text l2 loss",
    )
    parser.add_argument("--nsthl2_coef", default=1e4, type=float)

    parser.add_argument(
        "--softkd_loss",
        action="store_true",
        help="Whether to use softkd loss",
    )
    parser.add_argument("--softkd_coef", default=1, type=float)

    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--cluster_choice_loss", default=0, type=float) # 1 
    parser.add_argument("--cluster_feature_loss", default=1e4, type=float) # 1e4
    parser.add_argument("--cluster_memory_size", default=1024, type=int)
    parser.add_argument("--cluster_num", default=3, type=int) # 2

    parser.add_argument("--distillation", action="store_true")
    parser.add_argument("--fifo_memory", action="store_true")
    parser.add_argument("--without_pretrain", action="store_true")

    # Run specific
    parser.add_argument("--test", action="store_true", help="Whether to run evaluation on val or test set")
    parser.add_argument("--test_type", type=str, default="test", choices=("testA", "testB", "test"))
    parser.add_argument("--output-dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--load", default="", help="resume from checkpoint")
    parser.add_argument("--load_sth", default="", help="resume from checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true", help="Only run evaluation")
    parser.add_argument("--num_workers", default=10, type=int)

    # Distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    return parser


def main(args):
    # Init distributed mode
    dist.init_distributed_mode(args)

    if dist.is_main_process():
        if os.path.exists(Path(args.output_dir)):
            raise RuntimeError('The model directory already exists: %s' % args.output_dir)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Update dataset specific configs
    if args.dataset_config is not None:
        # https://stackoverflow.com/a/16878364
        d = vars(args)
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        d.update(cfg)

    print("git:\n  {}\n".format(utils.get_sha()))

    # Segmentation related
    if args.mask_model != "none":
        args.masks = True
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    print(args)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    # fix the seed for reproducibility
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(False)

    # Build the model
    model, criterion, cluster_criterion, weight_dict = build_model(args)
    model.to(device)
    criterion.to(device)
    assert criterion is not None
    if args.distillation:
        model_noun = deepcopy(model)
        model_noun.to(device)
        if cluster_criterion:
            cluster_criterion.to(device)
            cluster_criterion.syn_memory()
    else:
        model_noun = None
        model_noun_ema = None
        model_noun_without_ddp = None

    # Get a copy of the model for exponential moving averaged version of the model
    model_ema = deepcopy(model) if args.ema else None
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    if args.distillation:
        model_noun_ema = deepcopy(model_noun) if args.ema else None
        model_noun_without_ddp = model_noun
        if args.distributed:
            model_noun = torch.nn.parallel.DistributedDataParallel(model_noun, device_ids=[args.gpu], find_unused_parameters=True)
            model_noun_without_ddp = model_noun.module

    # Set up optimizers
    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and "text_encoder" not in n and p.requires_grad
            ]
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "text_encoder" in n and p.requires_grad],
            "lr": args.text_encoder_lr,
        },
    ]
    if args.distillation:
        param_dicts += [
            {
                "params": [
                    p
                    for n, p in model_noun_without_ddp.named_parameters()
                    if "backbone" not in n and "text_encoder" not in n and p.requires_grad
                ]
            },
            {
                "params": [p for n, p in model_noun_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
            {
                "params": [p for n, p in model_noun_without_ddp.named_parameters() if "text_encoder" in n and p.requires_grad],
                "lr": args.text_encoder_lr,
            },
        ]

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer in ["adam", "adamw"]:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Unsupported optimizer {args.optimizer}")

    # Train dataset
    if len(args.combine_datasets) == 0 and not args.eval:
        raise RuntimeError("Please provide at least one training dataset")

    dataset_train, sampler_train, data_loader_train = None, None, None

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)

    if not args.eval:
        dataset_train = ConcatDataset(
            [build_dataset(name, image_set="train", args=args, tokenizer=tokenizer) for name in args.combine_datasets]
        )
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.train_batch_size, drop_last=True)
        if args.distillation:
            data_loader_train = DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                collate_fn=partial(utils.collate_fn, False),
                num_workers=args.num_workers,
            )
        else:
            data_loader_train = DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                collate_fn=partial(utils.collate_fn_plain, False),
                num_workers=args.num_workers,
            )

    # Val dataset
    if len(args.combine_datasets_val) == 0:
        raise RuntimeError("Please provide at leas one validation dataset")

    Val_all = namedtuple(typename="val_data", field_names=["dataset_name", "dataloader", "base_ds", "evaluator_list"])

    val_tuples = []
    for dset_name in args.combine_datasets_val:
        dset = build_dataset(dset_name, image_set="val", args=args, tokenizer=tokenizer)
        sampler = (
            DistributedSampler(dset, shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(dset)
        )
        dataloader = DataLoader(
            dset,
            args.valid_batch_size,
            sampler=sampler,
            drop_last=False,
            collate_fn=partial(utils.collate_fn_plain, False),
            num_workers=args.num_workers,
        )
        base_ds = get_coco_api_from_dataset(dset)
        val_tuples.append(Val_all(dataset_name=dset_name, dataloader=dataloader, base_ds=base_ds, evaluator_list=None))

    if args.frozen_weights is not None:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        if "model_ema" in checkpoint and checkpoint["model_ema"] is not None:
            model_without_ddp.detr.load_state_dict(checkpoint["model_ema"], strict=False)
        else:
            model_without_ddp.detr.load_state_dict(checkpoint["model"], strict=False)

        if "criterion" in checkpoint:
            criterion.load_state_dict(checkpoint["criterion"])

        if args.ema:
            model_ema = deepcopy(model_without_ddp)

    # Used for loading weights from another model and starting a training from scratch. Especially useful if
    # loading into a model with different functionality.
    if args.load:
        if args.distillation:
            print("loading from", args.load)
            checkpoint = torch.load(args.load, map_location="cpu")
            if "model_ema" in checkpoint:
                model_noun_without_ddp.load_state_dict(checkpoint["model_ema"], strict=False)
            else:
                model_noun_without_ddp.load_state_dict(checkpoint["model"], strict=False)
            if args.ema:
                model_noun_ema = deepcopy(model_noun_without_ddp)

            # if "criterion" in checkpoint:
            #     criterion.load_state_dict(checkpoint["criterion"])

            checkpoint_base = torch.load(args.load_sth, map_location="cpu")
            if "model_ema" in checkpoint_base:
                model_without_ddp.load_state_dict(checkpoint_base["model_ema"], strict=False)
            else:
                model_without_ddp.load_state_dict(checkpoint_base["model"], strict=False)
            if args.ema:
                model_ema = deepcopy(model_without_ddp)
            if "cluster_criterion" in checkpoint_base and cluster_criterion:
                cluster_criterion.load_state_dict(checkpoint_base["cluster_criterion"])

    # Used for resuming training from the checkpoint of a model. Used when training times-out or is pre-empted.
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        if not args.eval and "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1

        if "criterion" in checkpoint:
            criterion.load_state_dict(checkpoint["criterion"], strict=False)
        if "cluster_criterion" in checkpoint and cluster_criterion:
            cluster_criterion.load_state_dict(checkpoint["cluster_criterion"], strict=False)

        if args.ema:
            if "model_ema" not in checkpoint:
                print("WARNING: ema model not found in checkpoint, resetting to current model")
                model_ema = deepcopy(model_without_ddp)
            else:
                model_ema.load_state_dict(checkpoint["model_ema"], strict=False)

    def build_evaluator_list(base_ds, dataset_name):
        """Helper function to build the list of evaluators for a given dataset"""
        evaluator_list = []
        iou_types = ["bbox"]
        if args.masks:
            iou_types.append("segm")

        if dataset_name[:4] == "tdod":
            evaluator_list = [TDODCocoEvaluator(base_ds, tuple(iou_types), useCats=True)]
        return evaluator_list

    if args.eval:
        ap_bbox_0p5_list = []
        ap_mask_0p5_list = []
        test_stats = {}
        test_model = model_ema if model_ema is not None else model
        test_model_noun = model_noun_ema if model_noun_ema is not None else model_noun
        for i, item in enumerate(val_tuples):
            evaluator_list = build_evaluator_list(item.base_ds, item.dataset_name)
            postprocessors = build_postprocessors(args, item.dataset_name)
            item = item._replace(evaluator_list=evaluator_list)
            print(f"Evaluating {item.dataset_name}")
            curr_test_stats = evaluate(
                model=test_model,
                model_noun=test_model_noun,
                criterion=criterion,
                cluster_criterion=cluster_criterion,
                postprocessors=postprocessors,
                weight_dict=weight_dict,
                data_loader=item.dataloader,
                evaluator_list=item.evaluator_list,
                device=device,
                args=args,
            )
            test_stats.update({item.dataset_name + "_" + k: v for k, v in curr_test_stats.items()})
            ap_bbox_0p5_list.append(curr_test_stats['coco_eval_bbox'][1])
            if args.masks:
                ap_mask_0p5_list.append(curr_test_stats['coco_eval_masks'][1])
        log_stats = {
            **{f"test_{k}": v for k, v in test_stats.items()},
            "n_parameters": n_parameters,
        }
        print('===============')
        print(log_stats)
        print('===============')

        final_map = np.mean(ap_bbox_0p5_list)
        for i in range(len(ap_bbox_0p5_list)):
            print("bbox AP@0.5 Task %d: %s" % (i+1, str(ap_bbox_0p5_list[i])))
        print("***bbox mAP@0.5: %s" % str(final_map))

        if args.masks:
            final_map = np.mean(ap_mask_0p5_list)
            for i in range(len(ap_mask_0p5_list)):
                print("mask AP@0.5 Task %d: %s" % (i+1, str(ap_mask_0p5_list[i])))
            print("***mask mAP@0.5: %s" % str(final_map))
        return

    writer = SummaryWriter(output_dir)

    # Runs training and evaluates after every --eval_skip epochs
    print("Start training")
    start_time = time.time()
    best_metric = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        print(f"Starting epoch {epoch}")
        if args.distributed:
            sampler_train.set_epoch(epoch)
        if args.distillation:
            train_stats = train_one_epoch_distillation(
                writer=writer,
                model=model,
                model_noun=model_noun,
                criterion=criterion,
                cluster_criterion=cluster_criterion,
                data_loader=data_loader_train,
                weight_dict=weight_dict,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                args=args,
                max_norm=args.clip_max_norm,
                model_ema=model_ema,
                model_noun_ema=model_noun_ema,
            )
        else:
            train_stats = train_one_epoch_plain(
                writer=writer,
                model=model,
                criterion=criterion,
                data_loader=data_loader_train,
                weight_dict=weight_dict,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                args=args,
                max_norm=args.clip_max_norm,
                model_ema=model_ema,
            )
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 2 epochs
            # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
            #     checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                dist.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "model_ema": model_ema.state_dict() if args.ema else None,
                        "model_noun": model_noun_without_ddp.state_dict() if args.distillation else None,
                        "model_noun_ema": model_noun_ema.state_dict() if args.distillation and args.ema else None,
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "args": args,
                        "criterion": criterion.state_dict(),
                        "cluster_criterion": cluster_criterion.state_dict() if args.cluster else None,
                    },
                    checkpoint_path,
                )

        if epoch % args.eval_skip == 0:
            test_stats = {}
            test_model = model_ema if model_ema is not None else model
            test_model_noun = model_noun_ema if model_noun_ema is not None else model_noun
            for i, item in enumerate(val_tuples):
                evaluator_list = build_evaluator_list(item.base_ds, item.dataset_name)
                item = item._replace(evaluator_list=evaluator_list)
                postprocessors = build_postprocessors(args, item.dataset_name)
                print(f"Evaluating {item.dataset_name}")
                curr_test_stats = evaluate(
                    model=test_model,
                    model_noun=test_model_noun,
                    criterion=criterion,
                    cluster_criterion=cluster_criterion,
                    postprocessors=postprocessors,
                    weight_dict=weight_dict,
                    data_loader=item.dataloader,
                    evaluator_list=item.evaluator_list,
                    device=device,
                    args=args,
                )
                test_stats.update({item.dataset_name + "_" + k: v for k, v in curr_test_stats.items()})
        else:
            test_stats = {}

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and dist.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            
        if epoch % args.eval_skip == 0:
            metric_bbox = np.mean([v[1] for k, v in test_stats.items() if "coco_eval_bbox" in k])    # mAP@0.5
            if args.mask:
                metric_masks = np.mean([v[1] for k, v in test_stats.items() if "coco_eval_masks" in k])    # mAP@0.5

            if dist.is_main_process():
                writer.add_scalar('map@0.5_bbox', metric_bbox, epoch)
                writer.add_scalar('map@0.5_masks', metric_masks, epoch)

                for i in range(1,15):
                    writer.add_scalar('%02d_ap@0.5_bbox' % i, test_stats['tdod_%d_coco_eval_bbox' % i][1], epoch)
                if args.mask:
                    for i in range(1,15):
                        writer.add_scalar('%02d_ap@0.5_masks' % i, test_stats['tdod_%d_coco_eval_masks' % i][1], epoch)

            if args.output_dir:
                save_best = False
                if not args.mask and metric_bbox > best_metric:
                    save_best = True
                    best_metric = metric_bbox
                elif args.mask and metric_masks > best_metric:
                    save_best = True
                    best_metric = metric_masks

                if save_best:
                    checkpoint_paths = [output_dir / "BEST_checkpoint.pth"]
                    # extra checkpoint before LR drop and every 100 epochs
                    for checkpoint_path in checkpoint_paths:
                        dist.save_on_master(
                            {
                                "model": model_without_ddp.state_dict(),
                                "model_ema": model_ema.state_dict() if args.ema else None,
                                "model_noun": model_noun_without_ddp.state_dict() if args.distillation else None,
                                "model_noun_ema": model_noun_ema.state_dict() if args.distillation and args.ema else None,
                                "optimizer": optimizer.state_dict(),
                                "epoch": epoch,
                                "args": args,
                                "criterion": criterion.state_dict(),
                                "cluster_criterion": cluster_criterion.state_dict() if args.cluster else None,
                            },
                            checkpoint_path,
                        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TOIST training and evaluation.", parents=[get_args_parser()])
    args = parser.parse_args()
    if not args.output_dir:
        args.output_dir = './logs/test'

    main(args)
