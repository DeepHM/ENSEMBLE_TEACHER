
# Ensemble Teacher: Semi-Supervised Semantic Segmentation with Teacher’s Cross-Pseudo Supervision

This repository is the official implementation of _Ensemble Teacher: Semi-Supervised Semantic Segmentation with Teacher’s Cross-Pseudo Supervision_.

<img src="https://user-images.githubusercontent.com/37736774/169817612-9586d7cd-3b43-4e85-96c6-e9e2316378a3.PNG" width="800" height="450"/>



#### Environment
* Ubuntu 20.04.4 LTS
* NVIDIA GeForce RTX 3090(24GB RAM) *2 
* 128GB RAM


## Installation

Please refer to this [installation](docs/installation.md) document to install dependencies and datasets.

## Training

Our experiments are performed with batch 8 on VOC dataset and batch 4 on Cityscapes dataset.

Thus, the training epochs are as follows :

|Dataset|1/16|1/8|1/4|1/2|
|------|---|---|---|---|
|VOC|32|34|40|60|
|Cityscapes|64|68|80|120|

#### - PASCAL VOC 2012

1. To train the CPS model re-implemented in the paper, run the following command.

```shell
$ mkdir output_voc/{save_dir}  # i.e. mkdir output_voc/voc8_res50_cutmix_B8
$ cd exp.voc/ensemble_teacher+cutmix

# Open the train_org.sh file and set up your training environment.
# For example : 
$ export volna="../../"
$ export NGPUS=2
$ export OUTPUT_PATH="../../output_voc/voc8_res50_cutmix_B8"
$ export snapshot_dir=$OUTPUT_PATH
$ export pretrained_model=$volna"DATA/pytorch-weight/resnet50_v1c.pth" # resnet50_v1c , resnet101_v1c.pth

$ export batch_size=8
$ export learning_rate=0.0025
$ export snapshot_iter=1
$ export labeled_ratio=8

# Start training
$ bash train_org.sh
```

2. To train our approach model in the paper, run the following command:

```shell
$ mkdir output_voc/{save_dir}  # i.e. mkdir output_voc/voc8_res50_cutmix_B8_EMA
$ cd exp.voc/ensemble_teacher+cutmix

# Open the train_ema.sh file and set up your training environment.
# For example : 
export volna="../../"
export NGPUS=2
export OUTPUT_PATH="../../output_voc/voc8_res50_cutmix_B8_EMA"
export snapshot_dir=$OUTPUT_PATH
export pretrained_model=$volna"DATA/pytorch-weight/resnet50_v1c.pth" # resnet50_v1c , resnet101_v1c.pth

export batch_size=8
export learning_rate=0.0025 
export snapshot_iter=1
export labeled_ratio=8

# Start training
$ bash train_ema.sh
```

#### - Cityscapes

1. To train the CPS model re-implemented in the paper, run the following command.

```shell
$ mkdir output_city/{save_dir}  # i.e. mkdir output_city/city8_res50_cutmix_B4
$ cd exp.city/ensemble_teacher+cutmix

# Open the train_org.sh file and set up your training environment.
# For example : 
$ export volna="../../"
$ export NGPUS=2
$ export OUTPUT_PATH="../../output_city/city8_res50_cutmix_B4" 
$ export snapshot_dir=$OUTPUT_PATH
$ export pretrained_model=$volna"DATA/pytorch-weight/resnet50_v1c.pth"  # resnet50_v1c , resnet101_v1c.pth

$ export batch_size=4   
$ export learning_rate=0.02
$ export snapshot_iter=1
$ export labeled_ratio=8

# Start training
$ bash train_org.sh
```

2. To train our approach model in the paper, run the following command:

```shell
$ mkdir output_city/{save_dir}  # i.e. mkdir output_city/city8_res50_cutmix_B4_EMA
$ cd exp.voc/ensemble_teacher+cutmix

# Open the train_ema.sh file and set up your training environment.
# For example : 
$ export volna="../../"
$ export NGPUS=2
$ export OUTPUT_PATH="../../output_city/city8_res50_cutmix_B4_EMA" 
$ export snapshot_dir=$OUTPUT_PATH
$ export pretrained_model=$volna"DATA/pytorch-weight/resnet50_v1c.pth"  # resnet50_v1c , resnet101_v1c.pth

$ export batch_size=4   
$ export learning_rate=0.02
$ export snapshot_iter=1
$ export labeled_ratio=8

# Start training
$ bash train_ema.sh
```


## Evaluation

When you run the evaluation process, the log_df.csv file is created and the results are accumulated and updated.

#### - PASCAL VOC 2012

```shell
$ cd exp.voc/ensemble_teacher+cutmix
# Open the eval.sh file and set up your training environment.
# For example : 
$ export volna="../../"
$ export NGPUS=2
$ export OUTPUT_PATH="../../output_voc/voc8_res50_cutmix_B8_EMA"
$ export snapshot_dir=$OUTPUT_PATH
$ export pretrained_model=$volna"DATA/pytorch-weight/resnet50_v1c.pth" # resnet50_v1c , resnet101_v1c.pth
## Set ensemble prediction options. See network.py
## In the paper, predictions for one model are made using the v0_2 or v1_2 option.
## v0 : Using Model_1 , v0_2 : Using Model_2 , v1 : Using EMA_Model_1 , v1_2 : Using EMA_Model_2 
## v2 : Ensemble prediction(soft-voting) using Model_1 and Model_2 , v3 : Ensemble prediction(soft-voting) using EMA_Model_1 and EMA_Model_2
## v4 :Ensemble prediction(soft-voting) using Model_1, Model_2, EMA_Model_1 and EMA_Model_2
$ export ensemble="v2"
$ export batch_size=8

# Start evaluation
$ bash eval.sh
```

#### - Cityscapes

```shell
$ cd exp.city/ensemble_teacher+cutmix
# Open the eval.sh file and set up your training environment.
# For example : 
$ export volna="../../"
$ export NGPUS=2
$ export OUTPUT_PATH="../../output_city/city8_res50_cutmix_B4_EMA"
$ export snapshot_dir=$OUTPUT_PATH
$ export pretrained_model=$volna"DATA/pytorch-weight/resnet50_v1c.pth" # resnet50_v1c , resnet101_v1c.pth
## Set ensemble prediction options. See network.py
## In the paper, predictions for one model are made using the v0_2 or v1_2 option.
## v0 : Using Model_1 , v0_2 : Using Model_2 , v1 : Using EMA_Model_1 , v1_2 : Using EMA_Model_2 
## v2 : Ensemble prediction(soft-voting) using Model_1 and Model_2 , v3 : Ensemble prediction(soft-voting) using EMA_Model_1 and EMA_Model_2
## v4 :Ensemble prediction(soft-voting) using Model_1, Model_2, EMA_Model_1 and EMA_Model_2
export ensemble="v3"
export batch_size=8

# Start evaluation
$ bash eval.sh
```



## Results

In semi-supervised learning, batch-size is important. (We train Cityscapes dataset in batch 4 due to GPU resource, The batch size of the PASCAL VOC dataset is 8)

Our model achieves the following performance on :

### [Semi-supervised semantic segmentation on PASCAL VOC 2012 val set](https://paperswithcode.com/task/semi-supervised-semantic-segmentation)

1. Resnet-50


| Model name  | 1/16(662)  | 1/8(1323) | 1/4(2646) | 1/2(5291) |
| -------- | -------- | -------- | -------- | -------- |
| CPS   |     72.50         |      74.75       |     **76.40**         |      76.60       |
| Our model   |     **72.86**         |      **75.20**       |     76.13         |      **76.95**       |


2. Resnet-101


| Model name  | 1/16(662)  | 1/8(1323) | 1/4(2646) | 1/2(5291) |
| -------- | -------- | -------- | -------- | -------- |
| CPS   |     75.13         |      **77.72**       |     78.81         |      79.20       |
| Our model   |     **76.04**         |      77.32       |     **79.38**         |      **79.69**       |


### [Semi-supervised semantic segmentation on Cityscapes val set](https://paperswithcode.com/task/semi-supervised-semantic-segmentation)

1. Resnet-50


| Model name  | 1/16(662)  | 1/8(1323) | 1/4(2646) | 1/2(5291) |
| -------- | -------- | -------- | -------- | -------- |
| CPS   |     66.29         |      68.35       |     69.17         |      69.29       |
| Our model   |     **68.03**         |      **70.01**       |     **72.93**         |      **72.84**       |


1. Resnet-101


| Model name  | 1/16(662)  | 1/8(1323) | 1/4(2646) | 1/2(5291) |
| -------- | -------- | -------- | -------- | -------- |
| CPS   |     68.73         |      69.41       |     72.36         |      73.54       |
| Our model   |     **68.73**         |      **70.74**       |     **73.62**         |      **73.73**       |




>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
