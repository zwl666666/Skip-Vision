#!/bin/bash

source activate llava
MODEL_PATH="checkpoints/sft/cos-clipvit-unlock_vision-448-q80-e024pt"

CUDA_VISIBLE_DEVICES=6 python -m llava.eval.model_vqa_loader \
    --model-path ${MODEL_PATH} \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./${MODEL_PATH}/pope/prediction.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./${MODEL_PATH}/pope/prediction.jsonl
