## Hierarchical-Attention-Network-for-Few-Shot-Object-Detection-Via-Meta-Contrastive-Learning
![Picture1](https://user-images.githubusercontent.com/59869350/179343518-7ae94313-66e6-45a7-b5e2-f57a2c069827.png)


## Introduction
This repo contains the official PyTorch implementation of our proposed methods Hierarchical Attention Network for Few-Shot Object Detection Via Meta-Contrastive Learning.
This repo is built upon DAnA.

## Updates
[2022/07/01] We release the official PyTorch implementation of hANMCL

## Requirements
<pre><code>Linux with Python == 3.7.0
PyTorch == 1.8.0+cu111
Torchvision == 0.9.0+cu111
CUDA 11.3
GCC == 7.5.0</code></pre>

## Getting Started
<pre><code>git clone https://github.com/infinity7428/hANMCL.git</code></pre>

## Build hANIMAL
<pre><code>cd lib
python setup.py build develop</code></pre>

## Datasets
<pre><code>Pascal VOC(07,12)
> data/voc/images/train2014
> data/voc/annotations/voc_train

MS-COCO
> data/coco/images/train2014
> data/coco/annotations/coco60_train
</code></pre>

### Datasets for training json

## Train
<pre><code>python train.py --dataset coco_base --flip --net hanmcl --lr 0.001 --lr_decay_step 12 --bs 4 --epochs 12 --disp_interval 20 --save_dir models/hanmcl --way 2 --shot 3</code></pre>


## Inference
<pre><code>python inference.py --eval --dataset val2014_novel --net hanmcl --r --load_dir models/hanmcl --checkepoch 12 --checkpoint 34467 --bs 1 --shot 3 --eval_dir result</code></pre>
> 

## Results on COCO dataset(nAP)
|Models|1shot|3shot|5shot|10shot|30shot|
|-----------|--------|--------|--------|--------|--------|
|Ours|12.9|14.4|14.5|22.4|25.0|
