#!/bin/bash


source activate llava
MODEL_PATH="checkpoints/sft/cos-clipvit-unlock_vision-448-q80-e024pt"

python -m llava.eval.model_vqa \
    --model-path ${MODEL_PATH} \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./${MODEL_PATH}/mm-vet/prediction.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1.1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./${MODEL_PATH}/mm-vet/prediction.jsonl \
    --dst ./${MODEL_PATH}/mm-vet/results.json

