# Copyright (c) Pengfei Li. All Rights Reserved
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""Postprocessors class to transform MDETR output according to the downstream task"""
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from IPython import embed


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        labels = torch.ones_like(labels)

        scores = 1 - prob[:, :, -1]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        assert len(scores) == len(labels) == len(boxes)
        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        if "pred_isfinal" in outputs:
            is_final = outputs["pred_isfinal"].sigmoid()
            scores_refexp = scores * is_final.view_as(scores)
            assert len(results) == len(scores_refexp)
            for i in range(len(results)):
                results[i]["scores_refexp"] = scores_refexp[i]

        return results


class PostProcessSegm(nn.Module):
    """Similar to PostProcess but for segmentation masks.

    This processor is to be called sequentially after PostProcess.

    Args:
        threshold: threshold that will be applied to binarize the segmentation masks.
    """

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    @torch.no_grad()
    def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
        """Perform the computation
        Parameters:
            results: already pre-processed boxes (output of PostProcess)
            outputs: raw outputs of the model
            orig_target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            max_target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                              after data augmentation.
        """
        assert len(orig_target_sizes) == len(max_target_sizes)
        max_h, max_w = max_target_sizes.max(0)[0].tolist()
        outputs_masks = outputs["pred_masks"].squeeze(2)
        outputs_masks = F.interpolate(outputs_masks, size=(max_h, max_w), mode="bilinear", align_corners=False)

        # Check if all sizes are the same, in which case we can do the interpolation more efficiently
        min_h, min_w = max_target_sizes.min(0)[0].tolist()
        min_orig_h, min_orig_w = orig_target_sizes.min(0)[0].tolist()
        max_orig_h, max_orig_w = orig_target_sizes.max(0)[0].tolist()
        if min_h == max_h and min_w == max_w and min_orig_h == max_orig_h and min_orig_w == max_orig_w:
            outputs_masks = (
                F.interpolate(outputs_masks, size=(min_orig_h, min_orig_w), mode="bilinear").sigmoid() > self.threshold
            ).cpu()
            for i, cur_mask in enumerate(outputs_masks):
                results[i]["masks"] = cur_mask.unsqueeze(1)
            return results

        for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
            img_h, img_w = t[0], t[1]
            results[i]["masks"] = cur_mask[:, :img_h, :img_w].unsqueeze(1)
            results[i]["masks"] = (
                F.interpolate(results[i]["masks"].float(), size=tuple(tt.tolist()), mode="bilinear").sigmoid()
                > self.threshold
            ).cpu()

        return results


def build_postprocessors(args, dataset_name) -> Dict[str, nn.Module]:
    postprocessors: Dict[str, nn.Module] = {"bbox": PostProcess()}
    if args.masks:
        postprocessors["segm"] = PostProcessSegm()

    return postprocessors
