# Copyright (c) Pengfei Li. All Rights Reserved
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MDETR model and criterion classes.
"""
from typing import Dict, Optional

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn

import util.dist as dist
from util import box_ops
from util.misc import NestedTensor, interpolate

from scipy.optimize import linear_sum_assignment

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import DETRsegm, dice_loss, sigmoid_focal_loss
from .transformer import build_transformer
from .kmeans import kmeans, kmeans_predict

from IPython import embed


class ClusterCriterion(nn.Module):
    def __init__(self, feature_dim, memory_size, cluster_num, task_count, args):
        super().__init__()
        self.args = args

        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.cluster_num = cluster_num
        self.task_count = task_count

        self.temp_feature_idx_list = [torch.zeros([args.train_batch_size, feature_dim+1]).cuda()
                                        for _ in range(torch.distributed.get_world_size())]

        feature_bank = torch.randn([task_count, memory_size, feature_dim])
        self.register_buffer("feature_bank", feature_bank)

        cluster_centers = torch.randn([task_count, cluster_num, feature_dim])
        self.register_buffer("cluster_centers", cluster_centers)

        update_count = torch.zeros([task_count])
        self.register_buffer("update_count", update_count)
        full_label = torch.zeros([task_count])
        self.register_buffer("full_label", full_label)

    def syn_memory(self):
        world_size = torch.distributed.get_world_size()

        torch.distributed.all_reduce(self.feature_bank)
        self.feature_bank /= world_size

        torch.distributed.all_reduce(self.cluster_centers)
        self.cluster_centers /= world_size

    def update_memory_queue(self, feature_idx_list):
        for feature_data in self.temp_feature_idx_list:
            feature_data *= 0
        torch.distributed.all_gather(self.temp_feature_idx_list, feature_idx_list)
        temp_feature_idx_tensor = torch.cat(self.temp_feature_idx_list, dim=0)

        new_feature_list = {}
        for i in range(self.task_count):
            new_feature_list[i] = []

        for i in range(len(temp_feature_idx_tensor)):
            new_task_idx = temp_feature_idx_tensor[i][-1]
            if new_task_idx == -1:  # empty
                continue
            new_feature = temp_feature_idx_tensor[i][:-1]
            new_feature_list[int(new_task_idx)].append(new_feature)

        for i in range(self.task_count):
            feature_length = len(new_feature_list[i])
            if feature_length == 0:
                continue
            feature_to_update = torch.stack(new_feature_list[i])

            if self.full_label[i] == 0:
                feature_to_remain = self.feature_bank[i][feature_length:].clone()
                self.feature_bank[i][:-feature_length] = feature_to_remain
                self.feature_bank[i][-feature_length:] = feature_to_update

                if self.update_count[i] > self.memory_size:
                    self.full_label[i] = 1
                self.update_count[i] += feature_length
            else:
                if self.args.fifo_memory:
                    feature_to_remain = self.feature_bank[i][feature_length:].clone()
                    self.feature_bank[i][:-feature_length] = feature_to_remain
                    self.feature_bank[i][-feature_length:] = feature_to_update
                else:   # replace nearest one
                    l1_dist = torch.cdist(feature_to_update, self.feature_bank[i], p=1).cpu()
                    indices = linear_sum_assignment(l1_dist)

                    for j in range(len(indices[1])):
                        self.feature_bank[i][indices[1][j]] = feature_to_update[indices[0][j]]

    def update_memory(self, memory_cache_noun, targets_noun, captions_noun):
        text_feature = torch.permute(memory_cache_noun['text_memory'], (1,0,2)) # BS x (num_tokens) x feature_dim
        normalized_text_emb = text_feature
        bs = normalized_text_emb.shape[0]

        tokenized = memory_cache_noun["tokenized"]

        # text token feature average
        token_feature_all_noun = torch.zeros([bs, normalized_text_emb.shape[-1]]).to(normalized_text_emb.device) # BS x hdim
        for i, tgt in enumerate(targets_noun):   # batchsize
            feature_i = []
            cur_tokens = [tgt["noun_tokens_positive"][j] for j in range(len(tgt["noun_tokens_positive"]))]
            for j, tok_list in enumerate(cur_tokens):   # bboxes in a sample
                pos_true = torch.zeros(normalized_text_emb.shape[1])
                for (beg, end) in tok_list:
                    beg_pos = tokenized.char_to_token(i, beg)
                    end_pos = tokenized.char_to_token(i, end - 1)
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

                    pos_true[beg_pos : end_pos + 1] = 1
                temp_token_feature = normalized_text_emb[i][pos_true.nonzero().reshape(-1)].mean(0)
                feature_i.append(temp_token_feature)
            if len(feature_i) > 0:
                feature_i = torch.stack(feature_i, dim=0)
                token_feature_all_noun[i] = feature_i.mean(0)

        all_mask = torch.zeros(bs, dtype=torch.bool)
        for i in range(len(targets_noun)):
            if len(targets_noun[i]['boxes']) == 0:
                all_mask[i] = True
        all_mask = all_mask.to(normalized_text_emb.device)

        task_idx_list = torch.ones(bs) * -1
        feature_list = torch.zeros([bs, normalized_text_emb.shape[-1]])
        task_idx_list = task_idx_list.to(normalized_text_emb.device)
        feature_list = feature_list.to(normalized_text_emb.device)
        for i in range(bs):
            if all_mask[i]:
                continue
            task_idx = int(targets_noun[i]['dataset_name'].split('_')[1]) - 1 # count from 0
            task_idx_list[i] = task_idx
            feature_list[i] = token_feature_all_noun[i].clone().detach()

        # update
        feature_idx_list = torch.cat([feature_list, task_idx_list.reshape(-1,1)], dim=-1)
        self.update_memory_queue(feature_idx_list)

        # reture
        memory_cache_noun['img_memory_mod'] = memory_cache_noun['img_memory'].clone()
        for i in range(bs):
            if all_mask[i]:
                continue

            task_idx = int(targets_noun[i]['dataset_name'].split('_')[1]) - 1 # count from 0
            cluster_center_choice, cluster_center_feature = self.memory_cluster(token_feature_all_noun[i].clone().detach(), task_idx)

            select_feature = self.cluster_centers[task_idx, cluster_center_choice]

            cur_tokens = [targets_noun[i]["noun_tokens_positive"][j] for j in range(len(targets_noun[i]["noun_tokens_positive"]))]
            pos_true = torch.zeros(normalized_text_emb.shape[1])
            for j, tok_list in enumerate(cur_tokens):   # bboxes in a sample
                for (beg, end) in tok_list:
                    beg_pos = tokenized.char_to_token(i, beg)
                    end_pos = tokenized.char_to_token(i, end - 1)
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

                    pos_true[beg_pos : end_pos + 1] = 1

            memory_cache_noun['img_memory_mod'][-len(memory_cache_noun['text_memory']):,i,:][pos_true.nonzero().reshape(-1)] = select_feature

        memory_cache_noun['full_label'] = self.full_label
        memory_cache_noun['update_count'] = self.update_count
        return memory_cache_noun

    def memory_cluster(self, feature_to_cluster, task_idx):
        memory_feature = self.feature_bank[task_idx]
        device = memory_feature.device

        cluster_ids_x, new_cluster_centers = kmeans(
            X=memory_feature, 
            init_cluster_centers=self.cluster_centers[task_idx].clone(),
            num_clusters=self.cluster_num, 
            distance='euclidean', 
            device=device,
            full_label=self.full_label[task_idx].item()
        )
        self.cluster_centers[task_idx] = new_cluster_centers

        cluster_ids_y = kmeans_predict(
            feature_to_cluster.reshape(1,-1), new_cluster_centers, 'euclidean', device=device, full_label=self.full_label[task_idx].item()
        )

        cluster_center_choice = cluster_ids_y[0]
        cluster_center_feature = new_cluster_centers[cluster_center_choice]

        return cluster_center_choice, cluster_center_feature

    def forward(self, memory_cache_sth, targets_sth, captions_sth):
        text_feature = torch.permute(memory_cache_sth['text_memory'], (1,0,2)) # BS x (num_tokens) x feature_dim

        normalized_text_emb = text_feature
        bs = normalized_text_emb.shape[0]

        tokenized = memory_cache_sth["tokenized"]

        memory_cache_sth['img_memory_mod'] = memory_cache_sth['img_memory'].clone()    # transformer decoder input

        loss_cluster_choice = torch.tensor(0.).to(normalized_text_emb.device)
        loss_cluster_feature = torch.tensor(0.).to(normalized_text_emb.device)
        loss_count = 0
        for i in range(bs):
            pos_true = torch.zeros(normalized_text_emb.shape[1])

            anno_name = 'something'
            begin_idx = captions_sth[i].find(anno_name)
            end_idx = begin_idx + len(anno_name)

            beg_pos = tokenized.char_to_token(i, begin_idx)
            end_pos = tokenized.char_to_token(i, end_idx - 1)
            pos_true[beg_pos : end_pos + 1] = 1

            temp_token_feature = normalized_text_emb[i][pos_true.nonzero().reshape(-1)].mean(0)

            task_idx = int(targets_sth[i]['dataset_name'].split('_')[1]) - 1 # count from 0
            cluster_center_choice, cluster_center_feature = self.memory_cluster(temp_token_feature.clone().detach(), task_idx)

            select_feature = self.cluster_centers[task_idx, cluster_center_choice]

            memory_cache_sth['img_memory_mod'][-len(memory_cache_sth['text_memory']):,i,:][pos_true.nonzero().reshape(-1)] = select_feature

            # loss
            loss_cluster_feature_i = F.mse_loss(temp_token_feature, \
                                        cluster_center_feature)
            loss_cluster_feature += loss_cluster_feature_i

            loss_count += 1

        if loss_count:
            loss_cluster_choice /= loss_count
            loss_cluster_feature /= loss_count

        return memory_cache_sth, {"loss_cluster_choice": loss_cluster_choice, "loss_cluster_feature": loss_cluster_feature}

    def infer_choice(self, memory_cache_sth, dataset_name_list, captions):
        text_feature = torch.permute(memory_cache_sth['text_memory'], (1,0,2)) # BS x (num_tokens) x feature_dim

        normalized_text_emb = text_feature
        bs = normalized_text_emb.shape[0]

        tokenized = memory_cache_sth["tokenized"]
        
        memory_cache_sth['img_memory_mod'] = memory_cache_sth['img_memory'].clone()    # transformer decoder input

        for i in range(bs):
            pos_true = torch.zeros(normalized_text_emb.shape[1])

            anno_name = 'something'
            begin_idx = captions[i].find(anno_name)
            end_idx = begin_idx + len(anno_name)

            beg_pos = tokenized.char_to_token(i, begin_idx)
            end_pos = tokenized.char_to_token(i, end_idx - 1)
            pos_true[beg_pos : end_pos + 1] = 1

            temp_token_feature = normalized_text_emb[i][pos_true.nonzero().reshape(-1)].mean(0)

            task_idx = int(dataset_name_list[i].split('_')[1]) - 1 # count from 0
            cluster_center_choice, cluster_center_feature = self.memory_cluster(temp_token_feature.clone().detach(), task_idx)

            select_feature = self.cluster_centers[task_idx, cluster_center_choice]

            memory_cache_sth['img_memory_mod'][-len(memory_cache_sth['text_memory']):,i,:][pos_true.nonzero().reshape(-1)] = select_feature

        return memory_cache_sth


class MDETR(nn.Module):
    """ This is the MDETR module that performs modulated object detection """

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        aux_loss=False,
        contrastive_hdim=64,
        contrastive_align_loss=False,
        cluster_num=16,
        args=None,
    ):
        """Initializes the model.

        Args:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         MDETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            contrastive_align_loss: If true, perform box - token contrastive learning
        """
        super().__init__()
        self.args = args
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_image = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_text = nn.Linear(hidden_dim, contrastive_hdim)

    def forward(self, samples: NestedTensor, captions, encode_and_save=True, memory_cache=None):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)

        if encode_and_save:
            assert memory_cache is None
            features, pos = self.backbone(samples)
            src, mask = features[-1].decompose()
            query_embed = self.query_embed.weight
            memory_cache = self.transformer(
                self.input_proj(src),
                mask,
                query_embed,
                pos[-1],
                captions,
                encode_and_save=True,
                text_memory=None,
                img_memory=None,
                text_attention_mask=None,
            )

            return memory_cache

        else:
            assert memory_cache is not None

            if self.args.cluster:
                hs = self.transformer(
                    mask=memory_cache["mask"],
                    query_embed=memory_cache["query_embed"],
                    pos_embed=memory_cache["pos_embed"],
                    encode_and_save=False,
                    text_memory=memory_cache["text_memory_resized"],
                    img_memory=memory_cache["img_memory_mod"],  # if args.cluster
                    text_attention_mask=memory_cache["text_attention_mask"],
                )
            else:
                hs = self.transformer(
                    mask=memory_cache["mask"],
                    query_embed=memory_cache["query_embed"],
                    pos_embed=memory_cache["pos_embed"],
                    encode_and_save=False,
                    text_memory=memory_cache["text_memory_resized"],
                    img_memory=memory_cache["img_memory"],
                    text_attention_mask=memory_cache["text_attention_mask"],
                )
            out = {}
            outputs_class = self.class_embed(hs)
            outputs_coord = self.bbox_embed(hs).sigmoid()
            out.update(
                {
                    "pred_logits": outputs_class[-1],
                    "pred_boxes": outputs_coord[-1],
                }
            )
            proj_queries, proj_tokens = None, None
            if self.contrastive_align_loss:
                proj_queries = F.normalize(self.contrastive_align_projection_image(hs), p=2, dim=-1)
                proj_tokens = F.normalize(
                    self.contrastive_align_projection_text(memory_cache["text_memory"]).transpose(0, 1), p=2, dim=-1
                )
                out.update(
                    {
                        "proj_queries": proj_queries[-1],
                        "proj_tokens": proj_tokens,
                        "tokenized": memory_cache["tokenized"],
                    }
                )
            if self.aux_loss:
                if self.contrastive_align_loss:
                    assert proj_tokens is not None and proj_queries is not None
                    out["aux_outputs"] = [
                        {
                            "pred_logits": a,
                            "pred_boxes": b,
                            "proj_queries": c,
                            "proj_tokens": proj_tokens,
                            "tokenized": memory_cache["tokenized"],
                        }
                        for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], proj_queries[:-1])
                    ]
                else:
                    out["aux_outputs"] = [
                        {
                            "pred_logits": a,
                            "pred_boxes": b,
                        }
                        for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                    ]
            return out


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, args, num_classes, matcher, eos_coef, losses, temperature, contrastive_hdim, task_count=14):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.temperature = temperature

    def loss_labels(self, memory_cache, outputs, targets, positive_map, indices, num_boxes, **kwargs):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        logits = outputs["pred_logits"].log_softmax(-1)  # BS x (num_queries) x (num_tokens)

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        tgt_pos = positive_map[tgt_idx]
        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1
        target_sim[src_idx] = tgt_pos

        loss_ce = -(logits * target_sim).sum(-1)

        eos_coef = torch.full(loss_ce.shape, self.eos_coef, device=target_sim.device)
        eos_coef[src_idx] = 1

        loss_ce = loss_ce * eos_coef
        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce}

        return losses

    @torch.no_grad()
    def softkd_matcher(self, prob, box):
        target_prob, source_prob = prob
        target_box, source_box = box

        # KLDIVLOSS 
        cost_class = (target_prob * (target_prob.unsqueeze(0).log() - source_prob.log().unsqueeze(1))).sum(-1)

        cost_bbox = torch.cdist(source_box, target_box, p=1)
        assert cost_class.shape == cost_bbox.shape

        cost_giou = -box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(source_box), box_ops.box_cxcywh_to_xyxy(target_box))

        # Final cost matrix
        # C = self.args.set_cost_bbox * cost_bbox + \
        #     self.args.set_cost_class * cost_class + \
        #     self.args.set_cost_giou * cost_giou
        C = (cost_bbox + cost_class + cost_giou).cpu()

        indices = linear_sum_assignment(C)

        return (torch.as_tensor(indices[0], dtype=torch.int64), torch.as_tensor(indices[1], dtype=torch.int64)) # src_idx, tgt_idx

    def loss_softkd(self, memory_cache, outputs, targets, positive_map, indices, num_boxes, **kwargs):
        if not isinstance(outputs, list): # [outputs_noun, outputs_sth]
            return {"loss_softkd": torch.tensor(0.).to(outputs["proj_tokens"].device)}

        memory_cache_noun, memory_cache_sth = memory_cache
        outputs_noun, outputs_sth = outputs
        targets_noun, targets_sth = targets
        indices_noun, indices_sth = indices

        prob_noun = outputs_noun["pred_logits"].clone().detach().softmax(-1)
        prob_sth = outputs_sth["pred_logits"].softmax(-1)

        bi_prob_noun = torch.cat([prob_noun[...,:-1].sum(-1, keepdim=True), prob_noun[...,-1:]], dim=-1)
        bi_prob_sth = torch.cat([prob_sth[...,:-1].sum(-1, keepdim=True), prob_sth[...,-1:]], dim=-1)

        loss_softkd = torch.tensor(0.).to(prob_noun.device)
        for i in range(len(indices_noun)):
            tp_bbox_noun = indices_noun[i]
            tp_bbox_sth = indices_sth[i]
            bi_prob_noun_i = bi_prob_noun[i]
            bi_prob_sth_i = bi_prob_sth[i]
            pred_boxes_noun_i = outputs_noun['pred_boxes'][i]
            pred_boxes_sth_i = outputs_sth['pred_boxes'][i]

            # tp
            tp_bi_prob_noun_i = torch.zeros([tp_bbox_noun[0].shape[0], 2]).to(prob_noun.device)
            tp_bi_prob_sth_i = torch.zeros([tp_bbox_sth[0].shape[0], 2]).to(prob_sth.device)
            
            tp_bi_prob_noun_i[tp_bbox_noun[1]] = bi_prob_noun_i[tp_bbox_noun[0]]
            tp_bi_prob_sth_i[tp_bbox_sth[1]] = bi_prob_sth_i[tp_bbox_sth[0]]

            # fp
            fp_indices_noun = torch.ones(self.args.num_queries, dtype=torch.bool).to(prob_noun.device)
            fp_indices_noun[tp_bbox_noun[0]] = False
            fp_bi_prob_noun_i = bi_prob_noun_i[fp_indices_noun]
            fp_pred_boxes_noun_i = pred_boxes_noun_i[fp_indices_noun]

            fp_indices_sth = torch.ones(self.args.num_queries, dtype=torch.bool).to(prob_sth.device)
            fp_indices_sth[tp_bbox_sth[0]] = False
            fp_bi_prob_sth_i = bi_prob_sth_i[fp_indices_sth]
            fp_pred_boxes_sth_i = pred_boxes_sth_i[fp_indices_sth]

            fp_indices = self.softkd_matcher([fp_bi_prob_noun_i, fp_bi_prob_sth_i], [fp_pred_boxes_noun_i, fp_pred_boxes_sth_i])

            fp_bi_prob_noun_i_loss = fp_bi_prob_noun_i[fp_indices[1]]
            fp_bi_prob_sth_i_loss = fp_bi_prob_sth_i[fp_indices[0]]

            # loss
            loss_bi_prob_noun_i = torch.cat([tp_bi_prob_noun_i, fp_bi_prob_noun_i_loss], dim=0)
            loss_bi_prob_sth_i = torch.cat([tp_bi_prob_sth_i, fp_bi_prob_sth_i_loss], dim=0)

            loss_softkd += F.kl_div(loss_bi_prob_sth_i.log(), loss_bi_prob_noun_i, reduction='batchmean')

        loss_softkd /= len(indices_noun)

        losses = {"loss_softkd": loss_softkd}
        return losses

    def loss_contrastive_align(self, memory_cache, outputs, targets, positive_map, indices, num_boxes, **kwargs):
        bs = outputs["proj_queries"].shape[0]
        tokenized = outputs["tokenized"]

        normalized_text_emb = outputs["proj_tokens"]  # BS x (num_tokens) x hdim
        normalized_img_emb = outputs["proj_queries"]  # BS x (num_queries) x hdim

        logits = (
            torch.matmul(normalized_img_emb, normalized_text_emb.transpose(-1, -2)) / self.temperature
        )  # BS x (num_queries) x (num_tokens)

        # construct a map such that positive_map[k, i,j] = True iff query i is associated to token j in batch item k
        # For efficency, the construction happens on CPU, then the whole matrix is transferred to GPU in one go.
        positive_map = torch.zeros(logits.shape, dtype=torch.bool)
        for i, ((idx_src, idx_tgt), tgt) in enumerate(zip(indices, targets)):
            if "tokens_positive" in tgt:
                cur_tokens = [tgt["tokens_positive"][j] for j in idx_tgt]
            else:
                cur_tokens = [tgt["tokens"][j] for j in idx_tgt]

            for j, tok_list in enumerate(cur_tokens):
                for (beg, end) in tok_list:
                    beg_pos = tokenized.char_to_token(i, beg)
                    end_pos = tokenized.char_to_token(i, end - 1)
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
                    positive_map[i, idx_src[j], beg_pos : end_pos + 1].fill_(True)

        positive_map = positive_map.to(logits.device)
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits  # .masked_fill(positive_map, -1000000)

        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        neg_term = negative_logits.logsumexp(2)

        nb_pos = positive_map.sum(2) + 1e-6

        box_to_token_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~boxes_with_pos, 0).sum()

        tokens_with_pos = positive_map.any(1)
        pos_term = positive_logits.sum(1)
        neg_term = negative_logits.logsumexp(1)

        nb_pos = positive_map.sum(1) + 1e-6

        tokens_to_boxes_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~tokens_with_pos, 0).sum()
        tot_loss = (box_to_token_loss + tokens_to_boxes_loss) / 2

        return {"loss_contrastive_align": tot_loss / num_boxes}

    def loss_nsthl2(self, memory_cache, outputs, targets, positive_map, indices, num_boxes, example_rel):
        if not isinstance(outputs, list): # [outputs_noun, outputs_sth]
            return {"loss_nsthl2": torch.tensor(0.).to(outputs["proj_tokens"].device)}

        memory_cache_noun, memory_cache_sth = memory_cache
        outputs_noun, outputs_sth = outputs
        targets_noun, targets_sth = targets
        indices_noun, indices_sth = indices

        ###### noun
        bs = outputs_noun["proj_queries"].shape[0]
        tokenized = outputs_noun["tokenized"]

        text_feature = torch.permute(memory_cache_noun['text_memory'], (1,0,2)) # BS x (num_tokens) x feature_dim
        normalized_text_emb = text_feature

        # text token feature average
        token_feature_all_noun = torch.zeros([bs, normalized_text_emb.shape[-1]]).to(normalized_text_emb.device) # BS x hdim
        for i, ((idx_src, idx_tgt), tgt) in enumerate(zip(indices_noun, targets_noun)):   # batchsize
            feature_i = []
            cur_tokens = [tgt["noun_tokens_positive"][j] for j in range(len(tgt["noun_tokens_positive"]))]
            for j, tok_list in enumerate(cur_tokens):   # bboxes in a sample
                pos_true = torch.zeros(normalized_text_emb.shape[1])
                for (beg, end) in tok_list:
                    beg_pos = tokenized.char_to_token(i, beg)
                    end_pos = tokenized.char_to_token(i, end - 1)
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

                    pos_true[beg_pos : end_pos + 1] = 1
                temp_token_feature = normalized_text_emb[i][pos_true.nonzero().reshape(-1)].mean(0)
                feature_i.append(temp_token_feature)
            if len(feature_i) > 0:
                feature_i = torch.stack(feature_i, dim=0)
                token_feature_all_noun[i] = feature_i.mean(0)

        ###### sth
        bs = outputs_sth["proj_queries"].shape[0]
        tokenized = outputs_sth["tokenized"]

        text_feature = torch.permute(memory_cache_sth['text_memory'], (1,0,2)) # BS x (num_tokens) x feature_dim
        normalized_text_emb = text_feature

        # calculate loss
        token_feature_all_sth = torch.zeros([bs, normalized_text_emb.shape[-1]]).to(normalized_text_emb.device) # BS x hdim
        for i, ((idx_src, idx_tgt), tgt) in enumerate(zip(indices_sth, targets_sth)):   # batchsize
            feature_i = []
            cur_tokens = [tgt["noun_tokens_positive"][j] for j in range(len(tgt["noun_tokens_positive"]))]
            for j, tok_list in enumerate(cur_tokens):   # bboxes in a sample
                pos_true = torch.zeros(normalized_text_emb.shape[1])
                for (beg, end) in tok_list:
                    beg_pos = tokenized.char_to_token(i, beg)
                    end_pos = tokenized.char_to_token(i, end - 1)
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

                    pos_true[beg_pos : end_pos + 1] = 1
                temp_token_feature = normalized_text_emb[i][pos_true.nonzero().reshape(-1)].mean(0)
                feature_i.append(temp_token_feature)
            if len(feature_i) > 0:
                feature_i = torch.stack(feature_i, dim=0)
                token_feature_all_sth[i] = feature_i.mean(0)

        all_mask = torch.zeros(bs, dtype=torch.bool)
        for i in range(len(indices_sth)):
            if len(indices_sth[i][0]) == 0:
                all_mask[i] = True
        all_mask = all_mask.to(normalized_text_emb.device)
        if (~all_mask).sum() == 0:
            return {"loss_nsthl2": torch.tensor(0.).to(normalized_text_emb.device)}

        ## text
        text_loss = torch.tensor(0.).to(normalized_text_emb.device)
        text_loss_count = 0
        for i in range(bs):
            if not all_mask[i]:
                text_loss += F.mse_loss(token_feature_all_sth[i], token_feature_all_noun[i].clone().detach())

                text_loss_count += 1
        text_loss = text_loss / text_loss_count

        return {"loss_nsthl2": text_loss}

    @torch.no_grad()
    def loss_cardinality(self, memory_cache, outputs, targets, positive_map, indices, num_boxes, **kwargs):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        ## Count the number of predictions that are NOT "no-object" (which is the last class)
        # normalized_text_emb = outputs["proj_tokens"]  # BS x (num_tokens) x hdim
        # normalized_img_emb = outputs["proj_queries"]  # BS x (num_queries) x hdim

        # logits = torch.matmul(
        #    normalized_img_emb, normalized_text_emb.transpose(-1, -2)
        # )  # BS x (num_queries) x (num_tokens)
        # card_pred = (logits[:, :, 0] > 0.5).sum(1)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}

        return losses

    def loss_boxes(self, memory_cache, outputs, targets, positive_map, indices, num_boxes, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def loss_masks(self, memory_cache, outputs, targets, positive_map, indices, num_boxes, **kwargs):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = NestedTensor.from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, memory_cache, outputs, targets, positive_map, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "contrastive_align": self.loss_contrastive_align,
            "nsthl2": self.loss_nsthl2,
            "softkd": self.loss_softkd
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](memory_cache, outputs, targets, positive_map, indices, num_boxes, **kwargs)

    def forward(self, memory_cache, outputs, targets, positive_map, example_rel):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        if isinstance(outputs, list): # [outputs_noun, outputs_sth]
            outputs_noun, outputs_sth = outputs
            targets_noun, targets_sth = targets
            positive_map_noun, positive_map_sth = positive_map

            losses = {}

            ############### loss noun
            loss_prefix = 'noun'
            outputs_without_aux = {k: v for k, v in outputs_noun.items() if k != "aux_outputs"}

            # Retrieve the matching between the outputs of the last layer and the targets
            indices_noun_main = self.matcher(outputs_without_aux, targets_noun, positive_map_noun)

            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes_noun = sum(len(t["labels"]) for t in targets_noun)
            num_boxes_noun = torch.as_tensor([num_boxes_noun], dtype=torch.float, device=next(iter(outputs_noun.values())).device)
            if dist.is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes_noun)
            num_boxes_noun = torch.clamp(num_boxes_noun / dist.get_world_size(), min=1).item()

            # Compute all the requested losses
            for loss in self.losses:
                if loss != "nsthl2" and loss != "softkd":
                    l_dict = self.get_loss(loss, memory_cache, outputs_noun, targets_noun, positive_map_noun, indices_noun_main, num_boxes_noun, example_rel=example_rel)
                    l_dict = {loss_prefix+'_'+k: v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if "aux_outputs" in outputs_noun:
                for i, aux_outputs in enumerate(outputs_noun["aux_outputs"]):
                    indices_noun = self.matcher(aux_outputs, targets_noun, positive_map_noun)
                    for loss in self.losses:
                        if loss == "masks" or loss == "nsthl2" or loss == "softkd":
                            # Intermediate masks losses are too costly to compute, we ignore them.
                            continue
                        kwargs = {"example_rel": example_rel}
                        l_dict = self.get_loss(loss, memory_cache, aux_outputs, targets_noun, positive_map_noun, indices_noun, num_boxes_noun, **kwargs)
                        l_dict = {loss_prefix+'_'+k+f"_{i}": v for k, v in l_dict.items()}
                        losses.update(l_dict)

            ############### loss sth
            loss_prefix = 'sth'
            outputs_without_aux = {k: v for k, v in outputs_sth.items() if k != "aux_outputs"}

            # Retrieve the matching between the outputs of the last layer and the targets
            indices_sth_main = self.matcher(outputs_without_aux, targets_sth, positive_map_sth)

            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes_sth = sum(len(t["labels"]) for t in targets_sth)
            num_boxes_sth = torch.as_tensor([num_boxes_sth], dtype=torch.float, device=next(iter(outputs_sth.values())).device)
            if dist.is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes_sth)
            num_boxes_sth = torch.clamp(num_boxes_sth / dist.get_world_size(), min=1).item()

            # Compute all the requested losses
            for loss in self.losses:
                if loss != "nsthl2" and loss != "softkd":
                    l_dict = self.get_loss(loss, memory_cache, outputs_sth, targets_sth, positive_map_sth, indices_sth_main, num_boxes_sth, example_rel=example_rel)
                    l_dict = {loss_prefix+'_'+k: v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if "aux_outputs" in outputs_sth:
                for i, aux_outputs in enumerate(outputs_sth["aux_outputs"]):
                    indices_sth = self.matcher(aux_outputs, targets_sth, positive_map_sth)
                    for loss in self.losses:
                        if loss == "masks" or loss == "nsthl2" or loss == "softkd":
                            # Intermediate masks losses are too costly to compute, we ignore them.
                            continue
                        kwargs = {"example_rel": example_rel}
                        l_dict = self.get_loss(loss, memory_cache, aux_outputs, targets_sth, positive_map_sth, indices_sth, num_boxes_sth, **kwargs)
                        l_dict = {loss_prefix+'_'+k+f"_{i}": v for k, v in l_dict.items()}
                        losses.update(l_dict)

            ############ noun sth l2 loss
            if self.args.nsthl2_loss:
                l_dict = self.get_loss('nsthl2', memory_cache, outputs, targets, positive_map, \
                                [indices_noun_main, indices_sth_main], [num_boxes_noun, num_boxes_sth], example_rel=example_rel)
                losses.update(l_dict)

            ############ softkd loss
            if self.args.softkd_loss:
                l_dict = self.get_loss('softkd', memory_cache, outputs, targets, positive_map, \
                                [indices_noun_main, indices_sth_main], [num_boxes_noun, num_boxes_sth], example_rel=example_rel)
                losses.update(l_dict)
                if "aux_outputs" in outputs_sth:
                    for i in range(len(outputs_noun["aux_outputs"])):
                        aux_outputs_noun = outputs_noun["aux_outputs"][i]
                        indices_noun = self.matcher(aux_outputs_noun, targets_noun, positive_map_noun)

                        aux_outputs_sth = outputs_sth["aux_outputs"][i]
                        indices_sth = self.matcher(aux_outputs_sth, targets_sth, positive_map_sth)

                        aux_outputs = [aux_outputs_noun, aux_outputs_sth]

                        l_dict = self.get_loss("softkd", memory_cache, aux_outputs, targets, positive_map, \
                                        [indices_noun, indices_sth], [num_boxes_noun, num_boxes_sth], example_rel=example_rel)
                        l_dict = {k+f"_{i}": v for k, v in l_dict.items()}
                        losses.update(l_dict)

            # return
            return losses
        else:
            outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

            # Retrieve the matching between the outputs of the last layer and the targets
            indices = self.matcher(outputs_without_aux, targets, positive_map)

            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
            if dist.is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

            # Compute all the requested losses
            losses = {}
            for loss in self.losses:
                losses.update(self.get_loss(loss, memory_cache, outputs, targets, positive_map, indices, num_boxes, example_rel=example_rel))

            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if "aux_outputs" in outputs:
                for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                    indices = self.matcher(aux_outputs, targets, positive_map)
                    for loss in self.losses:
                        if loss == "masks" or loss == "nsthl2" or loss == "softkd":
                            # Intermediate masks losses are too costly to compute, we ignore them.
                            continue
                        kwargs = {"example_rel": example_rel}
                        l_dict = self.get_loss(loss, memory_cache, aux_outputs, targets, positive_map, indices, num_boxes, **kwargs)
                        l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                        losses.update(l_dict)

            return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 255
    device = torch.device(args.device)

    assert not args.masks or args.mask_model != "none"

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = MDETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        contrastive_hdim=args.contrastive_loss_hdim,
        contrastive_align_loss=args.contrastive_align_loss,
        cluster_num=args.cluster_num,
        args=args,
    )
    if args.mask_model != "none":
        model = DETRsegm(
            model,
            mask_head=args.mask_model,
            freeze_detr=(args.frozen_weights is not None),
        )
    matcher = build_matcher(args)
    weight_dict = {"loss_ce": args.ce_loss_coef, "loss_bbox": args.bbox_loss_coef}
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    if args.nsthl2_loss:
        weight_dict["loss_nsthl2"] = args.nsthl2_coef
    if args.softkd_loss:
        weight_dict["loss_softkd"] = args.softkd_coef
    if args.cluster and args.distillation:
        weight_dict["loss_cluster_choice"] = args.cluster_choice_loss
        weight_dict["loss_cluster_feature"] = args.cluster_feature_loss

    weight_dict["loss_giou"] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef

    if args.distillation:
        cross_loss_list = ["loss_nsthl2", "loss_softkd", "loss_cluster_choice", "loss_cluster_feature"]
        prefix_weight_dict = {}
        for k,v in weight_dict.items():
            if k in cross_loss_list:
                prefix_weight_dict[k] = v
            else:
                prefix_weight_dict["noun_"+k] = v
                prefix_weight_dict["sth_"+k] = v

        if args.aux_loss:
            aux_weight_dict = {}
            for i in range(args.dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in prefix_weight_dict.items()})
            prefix_weight_dict.update(aux_weight_dict)
    else:
        if args.aux_loss:
            aux_weight_dict = {}
            for i in range(args.dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

    losses = ["labels", "boxes", "cardinality"]
    if args.masks:
        losses += ["masks"]
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]
    if args.nsthl2_loss:
        losses += ["nsthl2"]
    if args.softkd_loss:
        losses += ["softkd"]

    criterion = SetCriterion(
        args,
        num_classes,
        matcher=matcher,
        eos_coef=args.eos_coef,
        losses=losses,
        temperature=args.temperature_NCE,
        contrastive_hdim=args.contrastive_loss_hdim,
        task_count=14,
    )
    criterion.to(device)

    if args.cluster:
        cluster_criterion = ClusterCriterion(
            feature_dim=args.hidden_dim, 
            memory_size=args.cluster_memory_size, 
            cluster_num=args.cluster_num, 
            task_count=14,
            args=args,
        )
    else:
        cluster_criterion = None

    if args.distillation:
        return model, criterion, cluster_criterion, prefix_weight_dict
    else:
        return model, criterion, cluster_criterion, weight_dict
