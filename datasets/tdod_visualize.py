# Copyright (c) Pengfei Li. All Rights Reserved
import copy
from pathlib import Path

import torch
import torch.utils.data
from transformers import RobertaTokenizerFast
import torchvision
from pycocotools import mask as coco_mask

import util.dist as dist
from util.box_ops import generalized_box_iou
import datasets.transforms as T

import numpy as np

from .coco import make_coco_transforms
from IPython import embed

from PIL import Image
import json

TasksAbbreviation={
    1: 'step on something',
    2: 'sit comfortably on something',
    3: 'place flowers in something',
    4: 'get potatoes out of fire with something',
    5: 'water plant with something',
    6: 'get lemon out of tea with something',
    7: 'dig hole with something',
    8: 'open bottle of beer with something',
    9: 'open parcel with something',
    10: 'serve wine with something',
    11: 'pour sugar with something',
    12: 'smear butter with something',
    13: 'extinguish fire with something',
    14: 'pound carpet with something',
}

with open('/DATA1/lpf/tdod/mdetr/id2name.json', 'r') as id2name_f:
    catid2name = json.load(id2name_f)


class TdodDetection(torchvision.datasets.CocoDetection):
    def __init__(self, args, img_folder, ann_file, image_set, transforms, return_masks, return_tokens, tokenizer, is_train=False):
        self.args = args
        self.ann_file = ann_file
        self.image_set = image_set
        super(TdodDetection, self).__init__(img_folder, ann_file)

        with open(args.catid2name_path, 'r') as id2name_f:
            self.catid2name = json.load(id2name_f)

        self._transforms = transforms
        self.prepare = TOISTConvertCocoPolysToMask(self.catid2name, return_masks, return_tokens, tokenizer=tokenizer)
        self.is_train = is_train

        self.tasks_list = [TasksAbbreviation[i] for i in range(1,15)]
        self.total_caption = ' '.join(self.tasks_list)

        self.tokens_positive = {}
        for task_idx in TasksAbbreviation:
            task = TasksAbbreviation[task_idx]
            begin_idx = self.total_caption.find(task)
            end_idx = begin_idx + len(task)

            self.tokens_positive[task_idx] = [begin_idx, end_idx]


    def __getitem__(self, idx):
        img, target_ori = super(TdodDetection, self).__getitem__(idx)
        img_ori = copy.deepcopy(img)

        image_id = self.ids[idx]
        coco_img = self.coco.loadImgs(image_id)[0]

        dataset_name = self.ann_file.name

        caption = TasksAbbreviation[int(dataset_name.split('_')[1])]
        target = {"image_id": image_id, "annotations": target_ori, "caption": caption, 
                  "tokens_positive": [0,len(caption)]}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        target["dataset_name"] = dataset_name
        for extra_key in ["sentence_id", "original_img_id", "original_id", "task_id", "COCO_category_id"]:
            if extra_key in coco_img:
                target[extra_key] = coco_img[extra_key]

        if "tokens_positive_eval" in coco_img and not self.is_train:
            tokenized = self.prepare.tokenizer(caption, return_tensors="pt")
            target["positive_map_eval"] = create_positive_map(tokenized, coco_img["tokens_positive_eval"])
            target["nb_eval"] = len(target["positive_map_eval"])

        target['cat_name'] = [catid2name[str(item['COCO_category_id'])] for item in target_ori if item['category_id']==1]
        return img, target, img_ori


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


class TOISTConvertCocoPolysToMask(object):
    def __init__(self, catid2name=None, return_masks=False, return_tokens=False, tokenizer=None):
        self.catid2name = catid2name
        self.return_masks = return_masks
        self.return_tokens = return_tokens
        self.tokenizer = tokenizer

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        caption = target["caption"] if "caption" in target else None

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]
        anno = [obj for obj in anno if obj["category_id"] == 1] # prefered obj

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        isfinal = None
        if anno and "isfinal" in anno[0]:
            isfinal = torch.as_tensor([obj["isfinal"] for obj in anno], dtype=torch.float)

        tokens_positive = [] if self.return_tokens else None
        tokens_positive = [[target["tokens_positive"]] for obj in anno]

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if caption is not None:
            target["caption"] = caption
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        if tokens_positive is not None:
            target["tokens_positive"] = []

            for i, k in enumerate(keep):
                if k:
                    target["tokens_positive"].append(tokens_positive[i])

        if isfinal is not None:
            target["isfinal"] = isfinal

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self.return_tokens and self.tokenizer is not None:
            assert len(target["boxes"]) == len(target["tokens_positive"])
            tokenized = self.tokenizer(caption, return_tensors="pt")
            target["positive_map"] = create_positive_map(tokenized, target["tokens_positive"])
        return image, target


def make_coco_transforms(image_set, cautious):

    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    max_size = 1333
    if image_set == "train":
        horizontal = [] if cautious else [T.RandomHorizontalFlip()]
        return T.Compose(
            horizontal
            + [
                T.RandomSelect(
                    T.RandomResize(scales, max_size=max_size),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, max_size, respect_boxes=cautious),
                            T.RandomResize(scales, max_size=max_size),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose(
            [
                T.RandomResize([800], max_size=max_size),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")



def build(dataset_file, image_set, args, tokenizer):
    if image_set == 'train':
        img_dir = Path(args.coco_path) / "train2014"
        ann_file = Path(args.refexp_ann_path) / ('task_%s_train.json' % dataset_file.split('_')[-1])
        
    else:
        img_dir = Path(args.coco_path) / "val2014"
        ann_file = Path(args.refexp_ann_path) / ('task_%s_test.json' % dataset_file.split('_')[-1])

    dataset = TdodDetection(
        args,
        img_dir,
        ann_file,
        image_set,
        transforms=make_coco_transforms(image_set, cautious=True),
        return_masks=args.masks,
        return_tokens=True,
        tokenizer=tokenizer,
    )
    return dataset
