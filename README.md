## Hierarchical-Attention-Network-for-Few-Shot-Object-Detection-Via-Meta-Contrastive-Learning

To be continued..

## Introduction
This repo contains the official PyTorch implementation of our proposed methods Hierarchical Attention Network for Few-Shot Object Detection Via Meta-Contrastive Learning.

## Updates
[2022/07/01] We release the official PyTorch implementation of hANIMAL. h is silent.

## Quick Start
### 1. Check Requirements

## Comparison
### (1)
Linux with Python >= 3.6\n
PyTorch >= 1.6 & torchvision that matches the PyTorch version.\n
CUDA 10.1, 10.2\n
GCC >= 4.9\n

## Build hANIMAL


## Running
<pre><code>python inference.py --eval --dataset val2014_novel --net hanmcl --r --load_dir models/result --checkepoch 16 --checkpoint 34467 --bs 1 --shot 3 --eval_dir result</code></pre>
> 
