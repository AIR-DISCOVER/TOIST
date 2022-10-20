# Copyright (c) Pengfei Li. All Rights Reserved
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import subprocess
from typing import Any, Dict, List, Optional

import torch
import torchvision
from torch import Tensor
from IPython import embed


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(do_round, batch):
    batch = list(zip(*batch))

    batch_0_noun = []
    batch_0_sth = []
    for i in range(len(batch[0])):
        batch_0_noun.append(batch[0][i][0])
        batch_0_sth.append(batch[0][i][1])

    batch_1_noun = []
    batch_1_sth = []
    for i in range(len(batch[1])):
        batch_1_noun.append(batch[1][i][0])
        batch_1_sth.append(batch[1][i][1])

    example_rel = []
    for i in range(len(batch[0])):
        example_rel += [i] * len(batch[0][i])

    final_batch = {}
    final_batch["samples"] = [NestedTensor.from_tensor_list(batch_0_noun, do_round),
                              NestedTensor.from_tensor_list(batch_0_sth, do_round)]

    final_batch['example_rel'] = example_rel
    final_batch["targets"] = [batch_1_noun, batch_1_sth]

    if "positive_map" in batch_1_noun[0]:
        # noun
        max_len_noun = max([v["positive_map"].shape[1] for v in batch_1_noun])
        nb_boxes_noun = sum([v["positive_map"].shape[0] for v in batch_1_noun])
        batched_pos_map_noun = torch.zeros((nb_boxes_noun, max_len_noun), dtype=torch.bool)
        cur_count_noun = 0
        for v in batch_1_noun:
            cur_pos_noun = v["positive_map"]
            batched_pos_map_noun[cur_count_noun : cur_count_noun + len(cur_pos_noun), : cur_pos_noun.shape[1]] = cur_pos_noun
            cur_count_noun += len(cur_pos_noun)

        assert cur_count_noun == len(batched_pos_map_noun)
        # sth
        max_len_sth = max([v["positive_map"].shape[1] for v in batch_1_sth])
        nb_boxes_sth = sum([v["positive_map"].shape[0] for v in batch_1_sth])
        batched_pos_map_sth = torch.zeros((nb_boxes_sth, max_len_sth), dtype=torch.bool)
        cur_count_sth = 0
        for v in batch_1_sth:
            cur_pos_sth = v["positive_map"]
            batched_pos_map_sth[cur_count_sth : cur_count_sth + len(cur_pos_sth), : cur_pos_sth.shape[1]] = cur_pos_sth
            cur_count_sth += len(cur_pos_sth)

        assert cur_count_sth == len(batched_pos_map_sth)

        final_batch["positive_map"] = [batched_pos_map_noun.float(), batched_pos_map_sth.float()]

    return final_batch

def collate_fn_plain(do_round, batch):
    batch = list(zip(*batch))

    batch_0 = []
    for i in range(len(batch[0])):
        for item in batch[0][i]:
            batch_0.append(item)
    batch_1 = []
    for i in range(len(batch[1])):
        for item in batch[1][i]:
            batch_1.append(item)
    example_rel = []
    for i in range(len(batch[0])):
        example_rel += [i] * len(batch[0][i])
    batch = [batch_0, batch_1]

    final_batch = {}
    final_batch['example_rel'] = example_rel
    final_batch["samples"] = NestedTensor.from_tensor_list(batch[0], do_round)
    final_batch["targets"] = batch[1]
    if "positive_map" in batch[1][0]:
        max_len = max([v["positive_map"].shape[1] for v in batch[1]])
        nb_boxes = sum([v["positive_map"].shape[0] for v in batch[1]])
        batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
        cur_count = 0
        for v in batch[1]:
            cur_pos = v["positive_map"]
            batched_pos_map[cur_count : cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
            cur_count += len(cur_pos)

        assert cur_count == len(batched_pos_map)
        final_batch["positive_map"] = batched_pos_map.float()

    return final_batch

def collate_fn_visualize(do_round, batch):
    batch = list(zip(*batch))
    final_batch = {}
    final_batch["samples"] = NestedTensor.from_tensor_list(batch[0], do_round)
    final_batch["targets"] = batch[1]
    final_batch["img_ori"] = batch[2]
    if "positive_map" in batch[1][0]:
        # we batch the positive maps here
        # Since in general each batch element will have a different number of boxes,
        # we collapse a single batch dimension to avoid padding. This is sufficient for our purposes.
        max_len = max([v["positive_map"].shape[1] for v in batch[1]])
        nb_boxes = sum([v["positive_map"].shape[0] for v in batch[1]])
        batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
        cur_count = 0
        for v in batch[1]:
            cur_pos = v["positive_map"]
            batched_pos_map[cur_count : cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
            cur_count += len(cur_pos)

        assert cur_count == len(batched_pos_map)
        # assert batched_pos_map.sum().item() == sum([v["positive_map"].sum().item() for v in batch[1]])
        final_batch["positive_map"] = batched_pos_map.float()
    if "positive_map_eval" in batch[1][0]:
        # we batch the positive maps here
        # Since in general each batch element will have a different number of boxes,
        # we collapse a single batch dimension to avoid padding. This is sufficient for our purposes.
        max_len = max([v["positive_map_eval"].shape[1] for v in batch[1]])
        nb_boxes = sum([v["positive_map_eval"].shape[0] for v in batch[1]])
        batched_pos_map = torch.zeros((nb_boxes, max_len), dtype=torch.bool)
        cur_count = 0
        for v in batch[1]:
            cur_pos = v["positive_map_eval"]
            batched_pos_map[cur_count : cur_count + len(cur_pos), : cur_pos.shape[1]] = cur_pos
            cur_count += len(cur_pos)

        assert cur_count == len(batched_pos_map)
        # assert batched_pos_map.sum().item() == sum([v["positive_map"].sum().item() for v in batch[1]])
        final_batch["positive_map_eval"] = batched_pos_map.float()

    return final_batch


class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        cast_mask = self.mask.to(*args, **kwargs) if self.mask is not None else None
        return type(self)(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    @classmethod
    def from_tensor_list(cls, tensor_list, do_round=False):
        # TODO make this more general
        if tensor_list[0].ndim == 3:
            # TODO make it support different-sized images
            max_size = tuple(max(s) for s in zip(*[img.shape for img in tensor_list]))
            # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
            batch_shape = (len(tensor_list),) + max_size
            b, c, h, w = batch_shape
            if do_round:
                # Round to an even size to avoid rounding issues in fpn
                p = 128
                h = h if h % p == 0 else (h // p + 1) * p
                w = w if w % p == 0 else (w // p + 1) * p
                batch_shape = b, c, h, w

            dtype = tensor_list[0].dtype
            device = tensor_list[0].device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
            for img, pad_img, m in zip(tensor_list, tensor, mask):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
                m[: img.shape[1], : img.shape[2]] = False
        else:
            raise ValueError("not supported")
        return cls(tensor, mask)

    def __repr__(self):
        return repr(self.tensors)


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty channel sizes.
    """
    if input.numel() > 0:
        return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)

    assert input.shape[0] != 0 or input.shape[1] != 0, "At least one of the two first dimensions must be non zero"

    if input.shape[1] == 0:
        # Pytorch doesn't support null dimension on the channel dimension, so we transpose to fake a null batch dim
        return torch.nn.functional.interpolate(input.transpose(0, 1), size, scale_factor, mode, align_corners).transpose(0, 1)

    # empty batch dimension is now supported in pytorch
    return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)



def targets_to(targets: List[Dict[str, Any]], device):
    """Moves the target dicts to the given device."""
    excluded_keys = [
        "questionId",
        "tokens_positive",
        "noun_tokens_positive",
        "tokens",
        "dataset_name",
        "sentence_id",
        "original_img_id",
        "nb_eval",
        "task_id",
        "original_id",
        "idx",
        "cat_name",
    ]
    return [{k: v.to(device) if k not in excluded_keys else v for k, v in t.items() if k != "caption"} for t in targets]
