# TOIST: Task Oriented Instance Segmentation Transformer with Noun-Pronoun Distillation

This repository is an official implementation of [TOIST](https://arxiv.org/abs/2210.10775):

TOIST: Task Oriented Instance Segmentation Transformer with Noun-Pronoun Distillation  
[Pengfei Li](https://scholar.google.com/citations?hl=en&user=hmii_L8AAAAJ), [Beiwen Tian](https://github.com/TB5zhh), [Yongliang Shi](https://github.com/liangyongshi), [Xiaoxue Chen](https://scholar.google.com/citations?user=_tz64W0AAAAJ&hl=en), [Hao Zhao](https://sites.google.com/view/fromandto), [Guyue Zhou](https://air.tsinghua.edu.cn/en/info/1046/1196.htm), [Ya-Qin Zhang](https://air.tsinghua.edu.cn/en/info/1046/1188.htm)

**In NeurIPS 2022**

## Introduction
Current referring expression comprehension algorithms can effectively detect or segment objects indicated by nouns, but how to understand verb reference is still under-explored. As such, we study the challenging problem of task oriented detection, which aims to find objects that best afford an action indicated by verbs like *sit comfortably on*. Towards a finer localization that better serves downstream applications like robot interaction, we extend the problem into task oriented instance segmentation. A unique requirement of this task is to select *preferred* candidates among possible alternatives. Thus we resort to the transformer architecture which naturally models pair-wise query relationships with attention, leading to the TOIST method. In order to leverage pre-trained noun referring expression comprehension models and the fact that we can access privileged noun ground truth during training, a novel noun-pronoun distillation framework is proposed. Noun prototypes are generated in an unsupervised manner and contextual pronoun features are trained to select prototypes. As such, the network remains noun-agnostic during inference. We evaluate TOIST on the large-scale task oriented dataset COCO-Tasks and achieve +10.9\% higher $\rm{mAP^{box}}$ than the best-reported results. The proposed noun-pronoun distillation can boost $\rm{mAP^{box}}$ and $\rm{mAP^{mask}}$ by +2.8\% and +3.8\%.

<p align="center"><img src="media/teaser.png" width="600" /></p>

<p align="center"><img src="media/main.png" width="700" /></p>


If you find our code or paper useful, please consider citing:
```bibtex
coming soon...
```

This repository is a PyTorch implementation.

## Datasets
Please follow the instructions in [the official website](https://github.com/coco-tasks/dataset) to download the COCO-Tasks dataset.

You can organize the 'data' folder as follows:
```
data/
  ├── id2name.json
  ├── images/
  │    ├── train2014/
  │    └── val2014/
  └── coco-tasks/
       └── annotations/
            ├── task_1_train.json
            ├── task_1_test.json
            ...
            ├── task_14_train.json
            └── task_14_test.json
```
Then set the arguments `coco_path`, `refexp_ann_path` and `catid2name_path` in file `configs/tdod.json` to be the path of `data/images/`, `data/coco-tasks/annotations/` and `data/id2name.json`, respectively.

## Installation
Make sure that you have all dependencies in place. The simplest way to do so is to use anaconda.

Make a new conda env and activate it:
```
conda create --name TOIST python=3.8
conda activate TOIST
```

Install the the packages in the requirements.txt:
```
pip install -r requirements.txt
```

## Running

### 1. Plain TOIST detection

#### Training

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --master_port=23456 --nproc_per_node=6 --use_env main.py \
--dataset_config configs/tdod.json \
--train_batch_size 6  \
--valid_batch_size 8  \
--load /path/to/pretrained_resnet101_checkpoint.pth  \
--ema --text_encoder_lr 1e-5 --lr 5e-5 \
--num_workers 5 \
--output-dir 'logs/test' \
--eval_skip 1
```

To leverage the pre-trained noun referring expression comprehension model, download the checkpoint from [here](https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth?download=1) (provided by [MDETR](https://github.com/ashkamath/mdetr/blob/49fe251a1e1410cc529585d0e875e7e3d1fba92a/.github/pretrain.md)) and change the value of `--load` to be the path of the checkpoint.

#### Evaluation
Please change `--resume` to the path of the trained model to be evaluated.
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=23456 --nproc_per_node=1 --use_env main.py \
--dataset_config configs/tdod.json \
--valid_batch_size 8  \
--num_workers 5 \
--resume /path/to/checkpoint  \
--ema --eval \
--output-dir 'logs/test' \
--no_contrastive_align_loss
```

#### Verb-noun input
To train or evaluate the teacher TOIST model which leverages the privileged ground truth knowledge by taking verb-noun expression as text input, just set `--verb_noun_input` like:
```
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
```

#### Running without pre-training
To train TOIST without using the pre-trained noun referring expression comprehension model, leave the parameter `--load` empty and set `--without_pretrain`.
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --master_port=23456 --nproc_per_node=6 --use_env main.py \
--dataset_config configs/tdod.json \
--train_batch_size 6  \
--valid_batch_size 8  \
--ema --text_encoder_lr 1e-5 --lr 5e-5 \
--num_workers 5 \
--output-dir 'logs/test' \
--eval_skip 1 \
--without_pretrain
```
For evaluation, just change `--resume` and set `--without_pretrain` in the aforementioned evaluation command.


### 2. Plain TOIST segmentation
After training the detection part of TOIST, using the following commands to train and evaluate the segment head of TOIST.

#### Training
Please change `--frozen_weights` to the path of the trained detection model.
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --master_port=23456 --nproc_per_node=6 --use_env main.py \
--dataset_config configs/tdod.json \
--train_batch_size 2  \
--valid_batch_size 4  \
--frozen_weights /path/to/trained/detection/checkpoint \
--mask_model smallconv \
--no_aux_loss \
--ema --text_encoder_lr 1e-5 --lr 5e-5 \
--num_workers 5 \
--output-dir 'logs/test' \
--eval_skip 1 \
--no_contrastive_align_loss
```

#### Evaluation
Please change `--resume` to the path of the trained model to be evaluated.
```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=23456 --nproc_per_node=1 --use_env main.py \
--dataset_config configs/tdod.json \
--valid_batch_size 4  \
--num_workers 5 \
--resume /path/to/checkpoint  \
--ema --eval \
--output-dir 'logs/test' \
--mask_model smallconv \
--no_contrastive_align_loss
```

### 3. TOIST detection with noun-pronoun distillation

#### Training
To train TOIST with distillation, change `--load` to the path of the trained student model (taking verb-pronoun as text input) and `--load_noun` to the path of the trained teacher model (taking verb-noun as text input).

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --master_port=23456 --nproc_per_node=6 --use_env main.py \
--dataset_config configs/tdod.json \
--train_batch_size 3  \
--valid_batch_size 8  \
--load /path/to/pronoun/detection/checkpoint  \
--load_noun /path/to/noun/detection/checkpoint \
--ema --text_encoder_lr 1e-5 --lr 5e-5 \
--num_workers 5 \
--output-dir 'logs/test' \
--eval_skip 1 \
--distillation \
--softkd_loss \
--softkd_coef 50 \
--cluster \
--cluster_memory_size 1024 \
--cluster_num 3 \
--cluster_feature_loss 1e4
```

The parameters `--cluster`, `--cluster_memory_size`, `--cluster_num` and `--cluster_feature_loss` are used for *Clustering Distillation*. The parameters `--softkd_loss` and `--softkd_coef` are used for *Preference Distillation*.

#### Evaluation
Please change `--resume` to the path of the trained model to be evaluated.

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=23456 --nproc_per_node=1 --use_env main.py \
--dataset_config configs/tdod.json \
--valid_batch_size 4  \
--num_workers 5 \
--resume /path/to/checkpoint  \
--ema --eval \
--output-dir 'logs/test' \
--cluster \
--cluster_memory_size 1024 \
--cluster_num 3 \
--no_contrastive_align_loss \
--distillation
```

The parameters `--cluster_memory_size` and `--cluster_num` should be consistent with training setting.

### 4. TOIST segmentation with noun-pronoun distillation

#### Training
Please change `--frozen_weights` to the path of the trained detection (with distillation) model.

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.launch --master_port=23456 --nproc_per_node=6 --use_env main.py \
--dataset_config configs/tdod.json \
--train_batch_size 2  \
--valid_batch_size 4  \
--frozen_weights /path/to/trained/detection/with/distillation/checkpoint \
--mask_model smallconv \
--no_aux_loss \
--ema --text_encoder_lr 1e-5 --lr 5e-5 \
--num_workers 5 \
--output-dir 'logs/test' \
--eval_skip 1 \
--cluster \
--cluster_memory_size 1024 \
--cluster_num 3 \
--no_contrastive_align_loss
```

#### Evaluation
Please change `--resume` to the path of the trained model to be evaluated.

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=23456 --nproc_per_node=1 --use_env main.py \
--dataset_config configs/tdod.json \
--valid_batch_size 4  \
--num_workers 5 \
--resume /path/to/checkpoint  \
--ema --eval \
--output-dir 'logs/test' \
--cluster \
--cluster_memory_size 1024 \
--cluster_num 3 \
--mask_model smallconv \
--no_contrastive_align_loss
```





## Pre-trained Models
We provide our pretrained models on [Google Drive](https://drive.google.com/drive/folders/1g-4adboRxwO3yuob9tTnq8BZvjbbeVO6?usp=sharing).

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-y02l{border-color:inherit;color:#00E;text-align:left;text-decoration:underline;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky">Table/Figure No.</th>
    <th class="tg-0pky">Row No.</th>
    <th class="tg-0pky">Model Name</th>
    <th class="tg-0pky">Checkpoint</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow" rowspan="3">Table 1</td>
    <td class="tg-c3ow">1</td>
    <td class="tg-0pky">verb-pronoun input</td>
    <td class="tg-0pky"><a href="https://drive.google.com/file/d/1ud7VahH9vfKoUtd3L3Hkk_iTbXBRrGsb/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">2</td>
    <td class="tg-0pky">verb-noun input</td>
    <td class="tg-0pky"><a href="https://drive.google.com/file/d/1_7GSlO4u-3bCnQq4IqWqzdCVGM9aUXp3/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">5</td>
    <td class="tg-0pky">noun-pronoun distillation</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1OdIbiqF5E6fxMYVagQBNnIiFj1epT-VA/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">Figure3 (a)</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-0pky">decoder w/o self attention</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1YZu-hRYqy--MujuQdVpwGeydveBExrP0/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="5">Figure3 (b)</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-0pky">cluster number K=1</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1cygbd6ausRctEP89OjO9wOL06OJ4rJqo/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">/</td>
    <td class="tg-0pky">cluster number K=2</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/191E5QXJUIBJjFCd1neqZjlgVKNoSl1yI/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">/</td>
    <td class="tg-0pky">cluster number K=5</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/10Y0GECxo_-BFA6vullBcrD-uzcZMQyhf/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">/</td>
    <td class="tg-0pky">cluster number K=7</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1Og1hV7ZkHCRs3Qsy_bKu_SKMhcqoLeep/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">/</td>
    <td class="tg-0pky">cluster number K=10</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1eMrcspX0QxefaBl-gryHZtqMeMHOPY8E/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="6">Table 3</td>
    <td class="tg-c3ow">2</td>
    <td class="tg-0pky">CCR/CL/SBTL=F/F/T</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1Ibg4xOQJyHT2mtJQ-9qKIMuyQzOYrB1M/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">3</td>
    <td class="tg-0pky">CCR/CL/SBTL=F/T/F</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1Sjbp8P1wFgNlKeVakQN3X9WSUqa0D36s/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">4</td>
    <td class="tg-0pky">CCR/CL/SBTL=F/T/T</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1MhJEeApyR5Cg60gM4waq7-dV8U8XeSU4/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">5</td>
    <td class="tg-0pky">CCR/CL/SBTL=T/F/F</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/18gMXj0cryvvYANjfDfyWqW7wy5iR7eyr/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">6</td>
    <td class="tg-0pky">CCR/CL/SBTL=T/F/T</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1JjFxYrBpkl1By6K3N13txbtbbAE44mT-/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">7</td>
    <td class="tg-0pky">CCR/CL/SBTL=T/T/F</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1IdZiFgq7YRi-mueenI_iPM3tp070wd5j/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="3">Table 5</td>
    <td class="tg-c3ow">1</td>
    <td class="tg-0pky">verb-pronoun input w/o pretraining</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1HDvXd2UNpzpTgWmu0caFExqFAbMoOWrT/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">2</td>
    <td class="tg-0pky">verb-noun input w/o pretraining</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1Q2xE3YrOjWl4JBFaEBLtA2e-mSjuBeEs/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">3</td>
    <td class="tg-0pky">noun-pronoun distillation w/o pretraining</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1QhaYl0lTihYJko5jyXKDyl2wYUUvBt6X/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="6">Table 6</td>
    <td class="tg-c3ow">2</td>
    <td class="tg-0pky">it</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1cdqrHtoFbXFDP7fWrF25zW9M92A2t8c1/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">3</td>
    <td class="tg-0pky">them</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1DjcpOPeU20SFzVX6dw_NEPTn9sdXfgKf/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">4</td>
    <td class="tg-0pky">abcd</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/13OfdqoHmgmWlUDr_sp-8601I7kMWis6_/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">6</td>
    <td class="tg-0pky">it w/ distillation</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1g93uqLJ5L3fPzBS5eOCi1VsWM-4DbUDD/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">7</td>
    <td class="tg-0pky">them w/ distillation</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1xyoWRXSeude5UebYrvIFRcKbdUNzaDwT/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">8</td>
    <td class="tg-0pky">abcd w/ distillation</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1NzPW09ih4grF8JihdVWh_q2JrFV51ByF/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">Table 8</td>
    <td class="tg-c3ow">2</td>
    <td class="tg-0pky">first-in-first-out memory update</td>
    <td class="tg-y02l"><a href="https://drive.google.com/file/d/1Fb6M_pLcR7AMPewpoAb9I-4BVCNN9KWG/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a></td>
  </tr>
</tbody>
</table>


## License

TOIST is released under the MIT License.


## Acknowledgment

We would like to thank the open-source data and code of [COCO-Tasks](https://coco-tasks.github.io/), [Microsoft COCO](https://cocodataset.org/#home), [GGNN](https://github.com/yassersouri/task-driven-object-detection), [MDETR](https://github.com/ashkamath/mdetr), [DETR](https://github.com/facebookresearch/detr) and [Detectron2](https://github.com/facebookresearch/detectron2).




