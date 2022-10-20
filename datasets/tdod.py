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
    1: 'step on ',
    2: 'sit comfortably on ', 
    3: 'place flowers in ', 
    4: 'get potatoes out of fire with ', 
    5: 'water plant with ', 
    6: 'get lemon out of tea with ', 
    7: 'dig hole with ', 
    8: 'open bottle of beer with ', 
    9: 'open parcel with ', 
    10: 'serve wine with ', 
    11: 'pour sugar with ', 
    12: 'smear butter with ', 
    13: 'extinguish fire with ', 
    14: 'pound carpet with ', 
}

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
        self.return_masks = return_masks
        self.is_train = is_train

    def __getitem__(self, idx):
        img_ori, target_ori = super(TdodDetection, self).__getitem__(idx)

        image_id = self.ids[idx]
        coco_img = self.coco.loadImgs(image_id)[0]

        dataset_name = self.ann_file.name

        task_caption = TasksAbbreviation[int(dataset_name.split('_')[1])]

        if not self.args.distillation: # plain TOIST
            if self.args.verb_noun_input:
                # task + gt object name
                obj_caption = [task_caption+self.catid2name[str(item['COCO_category_id'])] for item in target_ori if item['category_id']==1]
                obj_caption = list(set(obj_caption))
                caption = ' '.join(obj_caption)

                target_obj = {"image_id": image_id, "annotations": target_ori, "caption": caption}
                target_obj["dataset_name"] = dataset_name
                img_obj = copy.deepcopy(img_ori)
                img_obj, target_obj = self.prepare(img_obj, target_obj, gt_obj=1)
                if self._transforms is not None:
                    img_obj, target_obj = self._transforms(img_obj, target_obj)

                return [img_obj], [target_obj]
            else:
                # task+'something'
                caption_sth = task_caption + 'something'
                target_sth = {"image_id": image_id, "annotations": target_ori, "caption": caption_sth}
                target_sth["dataset_name"] = dataset_name
                img_sth, target_sth = self.prepare(img_ori, target_sth, gt_obj=0)
                if self._transforms is not None:
                    img_sth, target_sth = self._transforms(img_sth, target_sth)

                return [img_sth], [target_sth]
        else: # noun-pronoun distillation
            if self.image_set == 'train':
                # task + gt object name
                obj_caption = [task_caption+self.catid2name[str(item['COCO_category_id'])] for item in target_ori if item['category_id']==1]
                obj_caption = list(set(obj_caption))
                caption = ' '.join(obj_caption)

                target_obj = {"image_id": image_id, "annotations": target_ori, "caption": caption}
                target_obj["dataset_name"] = dataset_name
                img_obj = copy.deepcopy(img_ori)
                img_obj, target_obj = self.prepare(img_obj, target_obj, gt_obj=1)
                if self._transforms is not None:
                    img_obj, target_obj = self._transforms(img_obj, target_obj)

                # task+'something'
                caption_sth = task_caption + 'something'
                target_sth = {"image_id": image_id, "annotations": target_ori, "caption": caption_sth}
                target_sth["dataset_name"] = dataset_name
                img_sth, target_sth = self.prepare(img_ori, target_sth, gt_obj=0)
                img_sth = copy.deepcopy(img_obj)
                target_sth['boxes'] = target_obj['boxes']
                if self.return_masks:
                    target_sth['masks'] = target_obj['masks']
                target_sth['labels'] = target_obj['labels']
                target_sth['area'] = target_obj['area']
                target_sth['size'] = target_obj['size']

                target_obj["idx"] = idx
                target_sth["idx"] = idx

                return [img_obj, img_sth], [target_obj, target_sth]
            else:
                # task+'something'
                caption_sth = task_caption + 'something'
                target_sth = {"image_id": image_id, "annotations": target_ori, "caption": caption_sth}
                target_sth["dataset_name"] = dataset_name
                img_sth, target_sth = self.prepare(img_ori, target_sth, gt_obj=0)
                if self._transforms is not None:
                    img_sth, target_sth = self._transforms(img_sth, target_sth)

                return [img_sth], [target_sth]


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

    def __call__(self, image, target, gt_obj):
        w, h = image.size

        dataset_name = target["dataset_name"]
        task_caption = TasksAbbreviation[int(dataset_name.split('_')[1])]

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
        noun_tokens_positive = [] if self.return_tokens else None
        if gt_obj == 1: # noun
            tokens_positive = []
            for obj in anno:
                anno_name = task_caption+self.catid2name[str(obj['COCO_category_id'])]
                begin_idx = caption.find(anno_name)
                end_idx = begin_idx + len(anno_name)
                tokens_positive.append([[begin_idx, end_idx]])
            for obj in anno:
                anno_name = self.catid2name[str(obj['COCO_category_id'])]
                begin_idx = caption.find(anno_name)
                end_idx = begin_idx + len(anno_name)
                noun_tokens_positive.append([[begin_idx, end_idx]])
        else: # pronoun
            tokens_positive = []
            for obj in anno:
                tokens_positive.append([[0, len(caption)]])
            for obj in anno:
                anno_name = 'something'
                begin_idx = caption.find(anno_name)
                end_idx = begin_idx + len(anno_name)
                noun_tokens_positive.append([[begin_idx, end_idx]])

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
        target["dataset_name"] = dataset_name
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

            target["noun_tokens_positive"] = []
            for i, k in enumerate(keep):
                if k:
                    target["noun_tokens_positive"].append(noun_tokens_positive[i])

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
            assert len(target["boxes"]) == len(target["tokens_positive"]) == len(target["noun_tokens_positive"])
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
