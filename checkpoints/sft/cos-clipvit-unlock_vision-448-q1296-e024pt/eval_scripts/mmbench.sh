#!/bin/bash


source activate llava
MODEL_PATH="checkpoints/sft/cos-clipvit-unlock_vision-448-q1296-e024pt"

SPLIT="mmbench_dev_en_20231003"


CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_mmbench \
    --model-path ${MODEL_PATH} \
    --question-file /gruntdata/heyuan4/workspace/pishi.hzy/data_annotations/mmbench/$SPLIT.tsv \
    --answers-file ./${MODEL_PATH}/mmb/${SPLIT}_prediction.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1.1

# mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

# python scripts/convert_mmbench_for_submission.py \
#     --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
#     --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
#     --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
#     --experiment llava-v1.5-13b
