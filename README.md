## Hierarchical-Attention-Network-for-Few-Shot-Object-Detection-Via-Meta-Contrastive-Learning

To be continued..

## Introduction
This repo contains the official PyTorch implementation of our proposed methods Hierarchical Attention Network for Few-Shot Object Detection Via Meta-Contrastive Learning.

## Updates
[2022/07/01] We release the official PyTorch implementation of hANIMAL. h is silent.

## Quick Start
### 1. Check Requirements

## Requirements
### (1)
<pre><code>Linux with Python == 3.7.0
PyTorch == 1.8.0+cu111
Torchvision == 0.9.0+cu111
CUDA 11.3
GCC == 7.5.0</code></pre>

## Getting Started
<pre><code>https://github.com/tommy/</code></pre>

## Build hANIMAL
<pre><code>cd lib
python setup.py build develop</code></pre>

## Running
<pre><code>python inference.py --eval --dataset val2014_novel --net hanmcl --r --load_dir models/result --checkepoch 16 --checkpoint 34467 --bs 1 --shot 3 --eval_dir result</code></pre>
> 
