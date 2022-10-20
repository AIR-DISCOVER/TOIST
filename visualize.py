# Copyright (c) Pengfei Li. All Rights Reserved
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import json
import os
import random
import time
from collections import namedtuple
from copy import deepcopy
from functools import partial
from pathlib import Path
import math
import sys

import numpy as np
import torch
import torch.nn
import torch.utils
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from transformers import RobertaTokenizerFast

from util.misc import interpolate
from torch.utils.tensorboard import SummaryWriter
import util.dist as dist
import util.misc as utils
from util import box_ops
from util.misc import targets_to
from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model
from models.postprocessors import build_postprocessors
from IPython import embed
import cv2



def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--run_name", default="", type=str)

    # Dataset specific
    parser.add_argument("--dataset_config", default=None, required=True)

    parser.add_argument("--no_detection", action="store_true", help="Whether to train the detector")

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
    parser.add_argument("--epochs", default=40, type=int)
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
    parser.add_argument("--masks", action="store_true") # lpf: for segmentation mask

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
    parser.add_argument("--nsthl2_coef", default=1, type=float)

    # Run specific

    parser.add_argument("--test", action="store_true", help="Whether to run evaluation on val or test set")
    parser.add_argument("--test_type", type=str, default="test", choices=("testA", "testB", "test"))
    parser.add_argument("--output-dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--load", default="", help="resume from checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true", help="Only run evaluation")
    parser.add_argument("--num_workers", default=10, type=int)

    # Distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    return parser

def draw_box(img,box,img_name,arm=None,gt_box=None,box1=None):
    pt1 = (int(box[0]),int(box[1]))
    pt2 = (int(box[2]),int(box[3]))
    cv2.rectangle(img,pt1,pt2,(0,255,0),4) # green
    if arm is not None:
        pt3 = (int(arm[0]),int(arm[1]))
        pt4 = (int(arm[2]),int(arm[3]))
        cv2.line(img,pt3,pt4,(0,0,255),5) #red
    if gt_box is not None:
        pt5 = (int(gt_box[0]),int(gt_box[1]))
        pt6 = (int(gt_box[2]),int(gt_box[3]))
        cv2.rectangle(img,pt5,pt6,(255,0,0),4) #blue
    if box1 is not None: # second best box
        pt7 = (int(box1[0]),int(box1[1]))
        pt8 = (int(box1[2]),int(box1[3]))
        cv2.rectangle(img,pt7,pt8,(255,255,0),4) #light blue
    cv2.imwrite(img_name,img)


def draw_box_mask(img_ori, targets, res, out_path_i):
    score_thresh = 0.95
    SAVE_BBOX = False
    SAVE_MASK = False
    SAVE_BBOX_MASK = True
    alpha = 0.5

    for i in range(len(img_ori)):
        img = img_ori[i]
        rgb_img_array = np.array(img)
        img_array = np.zeros_like(rgb_img_array)
        img_array[:,:,0] = rgb_img_array[:,:,2]
        img_array[:,:,1] = rgb_img_array[:,:,1]
        img_array[:,:,2] = rgb_img_array[:,:,0] # bgr
        target = targets[i]
        image_id = target["image_id"].item()
        res_item = res[image_id]

        # ori
        ori_out_file_path = out_path_i+'/'+str(image_id)+'_ori.png'
        cv2.imwrite(ori_out_file_path, img_array)

        # bbox
        if SAVE_BBOX:
            ## both
            img_array_both = deepcopy(img_array)
            box_out_file_path = out_path_i+'/'+str(image_id)+'_bbox_both.png'
            gt_bboxes = target['boxes']
            gt_bboxes_show = box_ops.box_cxcywh_to_xyxy(gt_bboxes)
            target_sizes = target['orig_size']
            img_h, img_w = target_sizes
            scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).cuda()
            gt_bboxes_show = gt_bboxes_show * scale_fct

            for gt_bbox in gt_bboxes_show:
                pt5 = (int(gt_bbox[0].item()),int(gt_bbox[1].item()))
                pt6 = (int(gt_bbox[2].item()),int(gt_bbox[3].item()))
                cv2.rectangle(img_array_both,pt5,pt6,(255,0,0),2) # blue

            keep = res_item['scores'] > score_thresh
            pred_boxes = res_item['boxes'][keep].view(-1, 4)
            pred_scores = res_item['scores'][keep]
            for pred_bbox in pred_boxes:
                pt3 = (int(pred_bbox[0].item()),int(pred_bbox[1].item()))
                pt4 = (int(pred_bbox[2].item()),int(pred_bbox[3].item()))
                cv2.rectangle(img_array_both,pt3,pt4,(0,0,255),2) # red

            cv2.imwrite(box_out_file_path, img_array_both)

            ## gt
            img_array_gt = deepcopy(img_array)
            gtbox_out_file_path = out_path_i+'/'+str(image_id)+'_bbox_gt_num'+str(len(gt_bboxes))+'.png'
            for gt_bbox in gt_bboxes_show:
                pt5 = (int(gt_bbox[0].item()),int(gt_bbox[1].item()))
                pt6 = (int(gt_bbox[2].item()),int(gt_bbox[3].item()))
                cv2.rectangle(img_array_gt,pt5,pt6,(255,0,0),2) # blue
            cv2.imwrite(gtbox_out_file_path, img_array_gt)

            ## pred
            img_array_pred = deepcopy(img_array)
            predbox_out_file_path = out_path_i+'/'+str(image_id)+'_bbox_pred_num'+str(len(pred_boxes))+'.png'
            for i in range(len(pred_boxes)):
                pred_bbox = pred_boxes[i]
                pt3 = (int(pred_bbox[0].item()),int(pred_bbox[1].item()))
                pt4 = (int(pred_bbox[2].item()),int(pred_bbox[3].item()))
                cv2.rectangle(img_array_pred,pt3,pt4,(0,0,255),2) # red

                bbox_text = str(pred_scores[i].item())[:5]
                (w, h), _ = cv2.getTextSize(bbox_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img_array_pred, (pt3[0], pt3[1] - h - 2), (pt3[0] + w, pt3[1]), (0,0,255), -1)
                cv2.putText(img_array_pred, bbox_text, (pt3[0], pt3[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 1, lineType=cv2.LINE_AA)

            cv2.imwrite(predbox_out_file_path, img_array_pred)

        # mask
        if SAVE_MASK:
            ## gt
            img_array_gt_mask = deepcopy(img_array)
            gtmask_out_file_path = out_path_i+'/'+str(image_id)+'_mask_gt.png'

            target_sizes = target['orig_size']
            h = target_sizes[0].item()
            w = target_sizes[1].item()
            ori_size = (h,w)
            ori_mask = interpolate(target["masks"][:, None].float(), ori_size, mode="nearest")[:, 0] > 0.5
            ori_mask = ori_mask.any(0)
            ori_mask = np.array(ori_mask.cpu())

            gt_mask_color = deepcopy(img_array)
            # gt_mask_color[ori_mask] = np.array((0,255,0)) # green
            gt_mask_color[ori_mask] = np.array((255,0,0)) # blue

            out_img_mask_gt = cv2.addWeighted(gt_mask_color, alpha, img_array_gt_mask, 1-alpha, 0)
            cv2.imwrite(gtmask_out_file_path, out_img_mask_gt)

            ## pred
            img_array_pred_mask = deepcopy(img_array)
            predmask_out_file_path = out_path_i+'/'+str(image_id)+'_mask_pred.png'

            keep = res_item['scores'] > score_thresh
            pred_mask = res_item['masks'][keep].squeeze(1)
            pred_mask = pred_mask.any(0)
            pred_mask = np.array(pred_mask.cpu())

            pred_mask_color = deepcopy(img_array)
            # pred_mask_color[pred_mask] = np.array((0,255,0)) # green
            pred_mask_color[pred_mask] = np.array((0,0,255)) # red

            out_img_mask_pred = cv2.addWeighted(pred_mask_color, alpha, img_array_pred_mask, 1-alpha, 0)
            cv2.imwrite(predmask_out_file_path, out_img_mask_pred)

        if SAVE_BBOX_MASK:
            ## gt
            gt_bboxes = target['boxes']
            gt_bboxes_show = box_ops.box_cxcywh_to_xyxy(gt_bboxes)
            target_sizes = target['orig_size']
            img_h, img_w = target_sizes
            scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).cuda()
            gt_bboxes_show = gt_bboxes_show * scale_fct

            img_array_gt = deepcopy(img_array)
            gtbox_out_file_path = out_path_i+'/'+str(image_id)+'_bbox_mask_gt'+str(len(gt_bboxes))+'.png'

            ### mask
            target_sizes = target['orig_size']
            h = target_sizes[0].item()
            w = target_sizes[1].item()
            ori_size = (h,w)
            ori_mask = interpolate(target["masks"][:, None].float(), ori_size, mode="nearest")[:, 0] > 0.5
            ori_mask = ori_mask.any(0)
            ori_mask = np.array(ori_mask.cpu())

            gt_mask_color = deepcopy(img_array)
            gt_mask_color[ori_mask] = np.array((255,0,0)) # blue

            out_img_mask_gt = cv2.addWeighted(gt_mask_color, alpha, img_array_gt, 1-alpha, 0)

            ### bbox
            for gt_bbox in gt_bboxes_show:
                pt5 = (int(gt_bbox[0].item()),int(gt_bbox[1].item()))
                pt6 = (int(gt_bbox[2].item()),int(gt_bbox[3].item()))
                cv2.rectangle(out_img_mask_gt,pt5,pt6,(255,0,0),2) # blue
            cv2.imwrite(gtbox_out_file_path, out_img_mask_gt)


            ## pred
            img_array_pred = deepcopy(img_array)

            ### mask
            keep = res_item['scores'] > score_thresh
            pred_mask = res_item['masks'][keep].squeeze(1)
            pred_mask = pred_mask.any(0)
            pred_mask = np.array(pred_mask.cpu())

            pred_mask_color = deepcopy(img_array)
            pred_mask_color[pred_mask] = np.array((0,0,255)) # red

            out_img_mask_pred = cv2.addWeighted(pred_mask_color, alpha, img_array_pred, 1-alpha, 0)

            ### bbox
            pred_boxes = res_item['boxes'][keep].view(-1, 4)
            pred_scores = res_item['scores'][keep]

            predbox_out_file_path = out_path_i+'/'+str(image_id)+'_bbox_mask_pred'+str(len(pred_boxes))+'.png'
            for i in range(len(pred_boxes)):
                pred_bbox = pred_boxes[i]
                pt3 = (int(pred_bbox[0].item()),int(pred_bbox[1].item()))
                pt4 = (int(pred_bbox[2].item()),int(pred_bbox[3].item()))
                cv2.rectangle(out_img_mask_pred,pt3,pt4,(0,0,255),2) # red

                bbox_text = str(pred_scores[i].item())[:8]
                (w, h), _ = cv2.getTextSize(bbox_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(out_img_mask_pred, (pt3[0], pt3[1] - h - 2), (pt3[0] + w, pt3[1]), (0,0,255), -1)
                cv2.putText(out_img_mask_pred, bbox_text, (pt3[0], pt3[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 1, lineType=cv2.LINE_AA)

            cv2.imwrite(predbox_out_file_path, out_img_mask_pred)


def main(args):
    # Init distributed mode
    dist.init_distributed_mode(args)

    if dist.is_main_process():
        # if os.path.exists(Path(args.output_dir)):
        #     raise RuntimeError('The model directory already exists: %s' % args.output_dir)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Update dataset specific configs
    if args.dataset_config is not None:
        # https://stackoverflow.com/a/16878364
        d = vars(args)
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        d.update(cfg)

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
    model, criterion, contrastive_criterion, qa_criterion, weight_dict = build_model(args)
    model.to(device)
    criterion.to(device)
    assert (
        criterion is not None or qa_criterion is not None
    ), "Error: should train either detection or question answering (or both)"

    # Get a copy of the model for exponential moving averaged version of the model
    model_ema = deepcopy(model) if args.ema else None
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)

    # Val dataset
    if len(args.combine_datasets_val) == 0:
        raise RuntimeError("Please provide at leas one validation dataset")

    Val_all = namedtuple(typename="val_data", field_names=["dataset_name", "dataloader", "base_ds", "evaluator_list"])

    val_tuples = []
    out_path = []
    for dset_name in args.combine_datasets_val:
        dset = build_dataset(dset_name, image_set="val", args=args, tokenizer=tokenizer, visualize=True)
        sampler = (
            DistributedSampler(dset, shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(dset)
        )
        dataloader = DataLoader(
            dset,
            args.valid_batch_size,
            sampler=sampler,
            drop_last=False,
            collate_fn=partial(utils.collate_fn_visualize, False),
            num_workers=args.num_workers,
        )
        base_ds = get_coco_api_from_dataset(dset)
        val_tuples.append(Val_all(dataset_name=dset_name, dataloader=dataloader, base_ds=base_ds, evaluator_list=None))

        if dist.is_main_process():
            Path(args.output_dir+'/'+dset_name).mkdir(parents=True, exist_ok=True)
        
        out_path.append(args.output_dir+'/'+dset_name)

    # Used for resuming training from the checkpoint of a model. Used when training times-out or is pre-empted.
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])

        if "criterion" in checkpoint:
            criterion.load_state_dict(checkpoint["criterion"])

        if args.ema:
            if "model_ema" not in checkpoint:
                print("WARNING: ema model not found in checkpoint, resetting to current model")
                model_ema = deepcopy(model_without_ddp)
            else:
                model_ema.load_state_dict(checkpoint["model_ema"])

    test_model = model_ema if model_ema is not None else model

    for i, item in enumerate(val_tuples):
        postprocessors = build_postprocessors(args, item.dataset_name)

        out_path_i = out_path[i]

        test_model.eval()

        for j, batch_dict in enumerate(item.dataloader):
            example_rel = 0
            samples = batch_dict["samples"].to(device)
            positive_map = batch_dict["positive_map"].to(device) if "positive_map" in batch_dict else None
            targets = batch_dict["targets"]
            answers = {k: v.to(device) for k, v in batch_dict["answers"].items()} if "answers" in batch_dict else None
            captions = [t["caption"] for t in targets]

            targets = targets_to(targets, device)

            image_id = targets[0]['image_id'].item() # batch size need to be 1

            memory_cache = None
            if args.masks:
                outputs = test_model(samples, captions)
            else:
                memory_cache = test_model(samples, captions, encode_and_save=True)
                outputs = test_model(samples, captions, encode_and_save=False, memory_cache=memory_cache)

            if not args.no_detection:
                orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
                results = postprocessors["bbox"](outputs, orig_target_sizes)
                if "segm" in postprocessors.keys():
                    target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                    results = postprocessors["segm"](results, outputs, orig_target_sizes, target_sizes)

                res = {target["image_id"].item(): output for target, output in zip(targets, results)}

            draw_box_mask(batch_dict["img_ori"], targets, res, out_path_i)
            print(j, image_id)

            torch.cuda.empty_cache()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TOIST training and evaluation.", parents=[get_args_parser()])
    args = parser.parse_args()
    if not args.output_dir:
        args.output_dir = 'logs/test'

    main(args)
