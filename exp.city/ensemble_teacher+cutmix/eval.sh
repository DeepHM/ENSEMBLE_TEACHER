#!/usr/bin/env bash
nvidia-smi

export volna="../../"
export NGPUS=2
export OUTPUT_PATH="../../output_city/city8_res50_cutmix_B4_EMA"
export snapshot_dir=$OUTPUT_PATH
export pretrained_model=$volna"DATA/pytorch-weight/resnet50_v1c.pth" # resnet50_v1c , resnet101_v1c.pth

export ensemble="v3"
export batch_size=8


echo "OUTPUT_PATH : {$OUTPUT_PATH}"
echo "pretrained_model : {$pretrained_model}"
echo "ensemble : {$ensemble}"

export TARGET_DEVICE=$[$NGPUS-1]
python eval.py -e 1-99999 -d 0-$TARGET_DEVICE -ens $ensemble --save_path $OUTPUT_PATH/results

