#!/usr/bin/env bash
nvidia-smi

export volna="../../"
export NGPUS=2
export OUTPUT_PATH="../../output_path/DEBUG"
export snapshot_dir=$OUTPUT_PATH

export batch_size=4
export learning_rate=0.0025
export snapshot_iter=1

export debug='True'

python -m torch.distributed.launch --nproc_per_node=$NGPUS DEBUG.py
