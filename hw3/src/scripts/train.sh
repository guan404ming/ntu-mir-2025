#!/bin/bash

python train.py --model_name EleutherAI/gpt-neo-125M --epochs 200 --tokenizer_name midilike
python train.py --model_name gpt2 --epochs 200 --tokenizer_name midilike

python train.py --model_name EleutherAI/gpt-neo-125M --epochs 200 --tokenizer_name remiplus
python train.py --model_name gpt2 --epochs 200 --tokenizer_name remiplus
