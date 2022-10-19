# Copyright (c) Pengfei Li. All Rights Reserved
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Dict, Iterable, Optional

import torch
import torch.nn
import torch.optim

import util.dist as dist
from datasets.coco_eval import TDODCocoEvaluator
from util.metrics import MetricLogger, SmoothedValue
from util.misc import targets_to
from util.optim import my_adjust_learning_rate, update_ema

from IPython import embed

def train_one_epoch_plain(
    writer,
    model: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    weight_dict: Dict[str, float],
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    max_norm: float = 0,
    model_ema: Optional[torch.nn.Module] = None,
):
    model.train()
    if criterion is not None:
        criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_backbone", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_text_encoder", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    num_training_steps = int(len(data_loader) * args.epochs)
    for i, batch_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        curr_step = epoch * len(data_loader) + i
        # example_rel = batch_dict["example_rel"]
        example_rel = 0
        samples = batch_dict["samples"].to(device)
        positive_map = batch_dict["positive_map"].to(device) if "positive_map" in batch_dict else None
        targets = batch_dict["targets"]
        answers = {k: v.to(device) for k, v in batch_dict["answers"].items()} if "answers" in batch_dict else None
        captions = [t["caption"] for t in targets]

        targets = targets_to(targets, device)
        # print(targets[0]['dataset_name'])

        memory_cache = None
        if args.masks:
            outputs = model(samples, captions)
        else:
            memory_cache = model(samples, captions, encode_and_save=True)
            outputs = model(samples, captions, encode_and_save=False, memory_cache=memory_cache)

        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(memory_cache, outputs, targets, positive_map, example_rel))

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        adjust_learning_rate(
            optimizer,
            epoch,
            curr_step,
            num_training_steps=num_training_steps,
            args=args,
        )
        if model_ema is not None:
            update_ema(model, model_ema, args.ema_decay)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_backbone=optimizer.param_groups[1]["lr"])
        metric_logger.update(lr_text_encoder=optimizer.param_groups[2]["lr"])
        
        if dist.is_main_process():
            writer.add_scalar('training_loss', loss_value, curr_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(k, loss_dict_reduced[k].item(), curr_step)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_distillation(
    writer,
    model: torch.nn.Module,
    model_noun: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    cluster_criterion: Optional[torch.nn.Module],
    weight_dict: Dict[str, float],
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    max_norm: float = 0,
    model_ema: Optional[torch.nn.Module] = None,
    model_noun_ema: Optional[torch.nn.Module] = None,
):
    model.train()
    model_noun.train()
    if criterion is not None:
        criterion.train()
    if cluster_criterion is not None:
        cluster_criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_backbone", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_text_encoder", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    num_training_steps = int(len(data_loader) * args.epochs)
    for i, batch_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        curr_step = epoch * len(data_loader) + i
        example_rel = batch_dict["example_rel"]

        samples = batch_dict["samples"]
        samples_noun = samples[0].to(device)
        samples_sth = samples[1].to(device)
        samples = [samples_noun, samples_sth]

        positive_map = batch_dict["positive_map"]
        positive_map_noun = positive_map[0].to(device)
        positive_map_sth = positive_map[1].to(device)
        positive_map = [positive_map_noun, positive_map_sth]

        targets = batch_dict["targets"]
        targets_noun = targets[0]
        targets_sth = targets[1]
        captions_noun = [t["caption"] for t in targets_noun]
        captions_sth = [t["caption"] for t in targets_sth]

        targets_noun = targets_to(targets_noun, device)
        targets_sth = targets_to(targets_sth, device)
        targets = [targets_noun, targets_sth]

        memory_cache = None
        if args.masks:
            outputs = model(samples_sth, captions_sth)
        else:
            memory_cache_noun = model_noun(samples_noun, captions_noun, encode_and_save=True)
            if args.cluster:
                memory_cache_noun = cluster_criterion.update_memory(memory_cache_noun, targets_noun, captions_noun)
            outputs_noun = model_noun(samples_noun, captions_noun, encode_and_save=False, memory_cache=memory_cache_noun)

            memory_cache_sth = model(samples_sth, captions_sth, encode_and_save=True)
            if args.cluster:
                memory_cache_sth, loss_cluster = cluster_criterion(memory_cache_sth, targets_sth, captions_sth)
            outputs_sth = model(samples_sth, captions_sth, encode_and_save=False, memory_cache=memory_cache_sth)

            memory_cache = [memory_cache_noun, memory_cache_sth]
            outputs = [outputs_noun, outputs_sth]

        if dist.is_main_process():
            for j in range(14):
                writer.add_scalar('full_label_'+str(j+1), memory_cache_noun['full_label'][j].item(), curr_step)
                writer.add_scalar('update_count_'+str(j+1), memory_cache_noun['update_count'][j].item(), curr_step)

        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(memory_cache, outputs, targets, positive_map, example_rel))

        if args.cluster:
            loss_dict.update(loss_cluster)

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(model_noun.parameters(), max_norm)
        optimizer.step()

        my_adjust_learning_rate(
            optimizer,
            epoch,
            curr_step,
            num_training_steps=num_training_steps,
            args=args,
        )
        if model_ema is not None:
            update_ema(model, model_ema, args.ema_decay)
        if model_noun_ema is not None:
            update_ema(model_noun, model_noun_ema, args.ema_decay)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_backbone=optimizer.param_groups[1]["lr"])
        metric_logger.update(lr_text_encoder=optimizer.param_groups[2]["lr"])
        
        if dist.is_main_process():
            writer.add_scalar('training_loss', loss_value, curr_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(k, loss_dict_reduced[k].item(), curr_step)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    model_noun: torch.nn.Module,
    criterion: Optional[torch.nn.Module],
    cluster_criterion: Optional[torch.nn.Module],
    postprocessors: Dict[str, torch.nn.Module],
    weight_dict: Dict[str, float],
    data_loader,
    evaluator_list,
    device: torch.device,
    args,
):
    model.eval()
    model_noun.eval()
    if criterion is not None:
        criterion.eval()
    if cluster_criterion is not None:
        cluster_criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "*****Test:"

    for batch_dict in metric_logger.log_every(data_loader, 10, header):
        example_rel = batch_dict["example_rel"]
        samples = batch_dict["samples"].to(device)
        positive_map = batch_dict["positive_map"].to(device) if "positive_map" in batch_dict else None
        targets = batch_dict["targets"]
        answers = {k: v.to(device) for k, v in batch_dict["answers"].items()} if "answers" in batch_dict else None
        captions = [t["caption"] for t in targets]
        dataset_name_list = [t["dataset_name"] for t in targets]
        noun_tokens_positive_list = [t["noun_tokens_positive"] for t in targets]

        targets = targets_to(targets, device)

        memory_cache = None
        if args.masks:
            outputs = model(samples, captions)
        else:
            memory_cache = model(samples, captions, encode_and_save=True)
            if args.distillation and args.cluster:
                memory_cache = cluster_criterion.infer_choice(memory_cache, dataset_name_list, captions)
            outputs = model(samples, captions, encode_and_save=False, memory_cache=memory_cache)

        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(memory_cache, outputs, targets, positive_map, example_rel))

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        if "segm" in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors["segm"](results, outputs, orig_target_sizes, target_sizes)

        res = {target["image_id"].item(): output for target, output in zip(targets, results)}

        for evaluator in evaluator_list:
            evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('===============')
    print("Averaged stats:", metric_logger)
    print('===============')

    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()

    for evaluator in evaluator_list:
        if isinstance(evaluator, TDODCocoEvaluator):
            evaluator.accumulate()
            evaluator.summarize()

    # accumulate predictions from all images

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    for evaluator in evaluator_list:
        if isinstance(evaluator, TDODCocoEvaluator):
            if "bbox" in postprocessors.keys():
                stats["coco_eval_bbox"] = evaluator.coco_eval["bbox"].stats.tolist()
            if "segm" in postprocessors.keys():
                stats["coco_eval_masks"] = evaluator.coco_eval["segm"].stats.tolist()

    return stats
