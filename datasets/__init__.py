# Copyright (c) Pengfei Li. All Rights Reserved
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .tdod import build as build_tdod
from .tdod_visualize import build as build_tdod_visualize


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(dataset_file: str, image_set: str, args, tokenizer, visualize=False):
    if dataset_file[:4] == "tdod":
        if visualize:
            return build_tdod_visualize(dataset_file, image_set, args, tokenizer)
        else:
            return build_tdod(dataset_file, image_set, args, tokenizer)

    raise ValueError(f"dataset {dataset_file} not supported")
