
# Ensemble Teacher: Semi-Supervised Semantic Segmentation with Teacher’s Cross-Pseudo Supervision

This repository is the official implementation of _Ensemble Teacher: Semi-Supervised Semantic Segmentation with Teacher’s Cross-Pseudo Supervision_.

<img src="https://user-images.githubusercontent.com/37736774/169817612-9586d7cd-3b43-4e85-96c6-e9e2316378a3.PNG" width="800" height="450"/>



#### Environment
* Ubuntu 20.04.4 LTS
* NVIDIA GeForce RTX 3090(24GB RAM) *2 
* 128GB RAM


## Installation

Please refer to this [installation](docs/installation.md) document to install dependencies and datasets.
Thus, the training epochs are as follows.
|Dataset|1/16|1/8|1/4|1/2|
|------|---|---|
|테스트1|테스트2|테스트3|테스트3|테스트3|
|테스트1|테스트2|테스트3|테스트3|테스트3|
|테스트1|테스트2|테스트3|테스트3|테스트3|

## Training

Our experiments are performed with batch 8 on VOC dataset and batch 4 on Cityscapes dataset.


1. To train the CPS model re-implemented in the paper, run the following command.

```train
$ mkdir output_voc/{save_dir}  # i.e. mkdir output_voc/voc8_res50_cutmix_B8
$ cd 
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```



>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
