#!/bin/bash

CKPT_NAME=${1:-"11-18-14-00-22"}
python task1.py \
    --ckpt_path checkpoints/$CKPT_NAME

python convert_mid_to_wav.py \
    --input_folder results/$CKPT_NAME/task1 \
    --output_folder results/$CKPT_NAME/task1_wav
