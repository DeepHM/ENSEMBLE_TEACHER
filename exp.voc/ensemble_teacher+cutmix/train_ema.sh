#!/usr/bin/env bash
nvidia-smi

export volna="../../"
export NGPUS=2
export OUTPUT_PATH="../../output_voc/voc8_res50_cutmix_B8_EMA"
export snapshot_dir=$OUTPUT_PATH
export pretrained_model=$volna"DATA/pytorch-weight/resnet50_v1c.pth" # resnet50_v1c , resnet101_v1c.pth

export batch_size=8
export learning_rate=0.0025 
export snapshot_iter=1
export labeled_ratio=8


echo "OUTPUT_PATH : {$OUTPUT_PATH}"
echo "pretrained_model : {$pretrained_model}"
echo "batch_size : {$batch_size}"
echo "learning_rate : {$learning_rate}"
echo "snapshot_iter : {$snapshot_iter}"
echo "labeled_ratio : {$labeled_ratio}"


export CUDA_VISIBLE_DEVICES=0,1
echo "CUDA_VISIBLE_DEVICES : {$CUDA_VISIBLE_DEVICES}"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m torch.distributed.launch --nproc_per_node=2 train_ema.py

python eval.py -e 1-99999 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results