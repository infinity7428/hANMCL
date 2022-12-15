## Hierarchical-Attention-Network-for-Few-Shot-Object-Detection-Via-Meta-Contrastive-Learning
![image](https://user-images.githubusercontent.com/59869350/207762476-2005a476-431b-4e4f-9c43-4f3a14eb80b5.png)



## Introduction
This repo contains the official PyTorch implementation of our proposed methods Hierarchical Attention Network for Few-Shot Object Detection Via Meta-Contrastive Learning.
This repo is built upon DAnA.

## Updates
[2022/07/01] We release the official PyTorch implementation of hANMCL.  
[2022/09/21] Support Pascal VOC and MS-COCO datasets.  
[2022/11/30] Support Multi-GPU training and fine-tuning.  
[2022/12/15] Upload support images and new codes.

## Requirements
<pre><code>Linux with Python == 3.7.0
PyTorch == 1.8.0+cu111
Torchvision == 0.9.0+cu111
CUDA 11.3
GCC == 7.5.0</code></pre>

## Getting Started
<pre><code>git clone https://github.com/infinity7428/hANMCL.git</code></pre>

## Compile COCO API
<pre><code>$ cd lib
$ git clone https://github.com/pdollar/coco.git 
$ cd coco/PythonAPI
$ make && make install</code></pre>


## Build hANIMAL
<pre><code>cd lib
python setup.py build develop</code></pre>

## Datasets
<pre><code>Pascal VOC(07,12)
> data/voc/images/train2014
> data/voc/annotations/voc_train
https://drive.google.com/drive/folders/1K94ot7sBrChubmUA8HK0ym_7QT-ZG4su?usp=sharing

MS-COCO
> data/coco/images/train2014
> data/coco/annotations/coco60_train
60 base classes for training (https://drive.google.com/file/d/10mXvdpgSjFYML_9J-zMDLPuBYrSrG2ub/view?usp=sharing)
20 novel classes for testing (https://drive.google.com/file/d/1FZJhC-Ob-IXTKf5heNeNAN00V8OUJXi2/view?usp=sharing)
Reference : https://github.com/Tung-I/Dual-awareness-Attention-for-Few-shot-Object-Detection
</code></pre>

## Support images and JSON
<pre><code>Pascal VOC(07,12)
> supports/pascal
https://drive.google.com/file/d/13x06dwZn4focLMUvM73zv88f3FbIs4E_/view?usp=sharing

MS-COCO
> supports/coco
https://drive.google.com/file/d/1G3xoXEJu5E0HMcQtZoTvFAVb9YFQoadX/view?usp=sharing
</code></pre>

## Train
<pre><code>train COCO (60 base category)
python train.py --dataset coco_base --flip --net hanmcl --lr 0.001 --lr_decay_step 12 --bs 4 --epochs 12 --disp_interval 20 --save_dir models/hanmcl --way 2 --shot 3

train VOC (5 base category)
python train.py --dataset voc{SPLIT_ID (1, 2 or 3)} --flip --net hanmcl --lr 0.001 --lr_decay_step 12 --bs 4 --epochs 10 --disp_interval 20 --save_dir models/hANMCL --way 2 --shot 3
ex) python train.py --dataset voc1 --flip --net hanmcl --lr 0.001 --lr_decay_step 12 --bs 4 --epochs 10 --disp_interval 20 --save_dir models/hANMCL --way 2 --shot 3</code></pre>

## for Multi-GPU Train
<pre><code>python train.py --dataset {DATASET} --flip --net hanmcl --lr 0.001 --lr_decay_step 12 --bs 4 --epochs 12 --disp_interval 20 --save_dir models/hanmcl --way 2 --shot 3 --mGPUS
</code></pre>


## for fine-tune
<pre><code>train COCO
python train.py --dataset coco_ft --flip --net DAnA --lr 0.001 --lr_decay_step 12 --bs 2 --epochs 12 --disp_interval 20 --save_dir models/{SEED}_{SHOTS} --way 2 --shot {TRAIN SHOTS} --seed seed{SEED} --shots {SHOTS}shots --r

train VOC
python train.py --dataset coco_ft --flip --net DAnA --lr 0.001 --lr_decay_step 12 --bs 2 --epochs 12 --disp_interval 20 --save_dir models/{SEED}_{SHOTS} --way 2 --shot {TRAIN SHOTS} --seed seed{SEED} --shots {SHOTS}shots --r
</code></pre>


## Inference
<pre><code>test COCO (20 novel category)
python inference.py --eval --dataset val2014_novel --net hanmcl --r --load_dir models/hANMCL --checkepoch 12 --checkpoint 34467 --bs 1 --shot 3 --eval_dir hanmcl

test COCO (60 base category)
python inference.py --eval --dataset val2014_base --net hanmcl --r --load_dir models/hANMCL --checkepoch 12 --checkpoint 34467 --bs 1 --shot 3 --eval_dir hanmcl

test VOC (5 novel category)
python inference.py --eval --dataset voc_test{SPLIT_ID (1, 2 or 3)} --net hanmcl --r --load_dir models/hANMCL --checkepoch 10 --checkpoint ft --bs 1 --shot 3 --eval_dir hanmcl --sup_dir pascal/split{SPLIT_ID (1, 2 or 3)}/seed{SEED (1 to 10)}/{SHOTS (1 to 10)}shot_image_novel
ex) python inference.py --eval --dataset voc_test1 --net hanmcl --r --load_dir models/hANMCL --checkepoch 10 --checkpoint ft --bs 1 --shot 3 --eval_dir hanmcl --sup_dir pascal/split1/seed1/10shot_image_novel
</code></pre>

## Pretrained Weights
<pre><code>path: models/hanmcl/train/checkpoints/model_12_34467.pth  
link: https://drive.google.com/drive/folders/1sPiadJ-Aw5N9uFaR1lTS2qlx3ABUudXh?usp=sharing</code></pre>
