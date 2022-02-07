# About the repository

This incomplete repository works on action recognition (excercise activity in particular) on the [MM-Fit dataset](https://mmfit.github.io/) that adapts:
+ [Vision Transformer (ViT)](https://arxiv.org/pdf/2010.11929.pdf)  
+ [MLP-Mixer](https://arxiv.org/pdf/2105.01601.pdf)  
+ [Video Vision Transformer (ViViT)](https://arxiv.org/pdf/2103.15691v1.pdf)  

Although the MM-Fit dataset contains various form of data, with 10 activity excercise classes (plus 1 class non activity). So far we have just been working with only 2D skeleton data for visual action recognition, but there are more to come soon...

Some resource has been used for this repository:
+ [MM-Fit paper](https://dl.acm.org/doi/10.1145/3432701)
+ The [baseline](https://github.com/KDMStromback/mm-fit) auto encoder-decoder from MM-Fit 
+ Implementation of [ViT](https://github.com/lucidrains/vit-pytorch) on Pytorch  
+ Implementation of [MLP-Mixer](https://github.com/lucidrains/mlp-mixer-pytorch) on Pytorch  
+ Implementation of ViVit on Pytorch [model 2](https://github.com/DylanTao94/ViViT-Model2) and [model 3](https://github.com/drv-agwl/ViViT-pytorch)


Detailed of the MM-Fit dataset is providedin `EDA.ipynb`.  
Please look at `training_scenarios.txt` for some training parameters suggestion.  
Two file `sampling_image.py` and `sampling_video.py` help provide the distribution of the MM-fit dataset over the train/var/test set on 11 classes.  

## Installation Guide

```
conda env create -f environment.yml
conda activate mm-fit
conda install -c conda-forge einops
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
```

## Result
Result so far (up to Sep 9th, 2021):  
  + ViT: 56.32% Acc
  + MLP-Mixer: 74.44% Acc
  + Vivit: 79.69% Acc
