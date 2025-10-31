#!/bin/bash
source activate llava
# MODEL_PATH="./checkpoints/sft/cos-unlock_vision-448-q336"
MODEL_PATH="checkpoints/sft/cos-clipvit-unlock_vision-448-q336-e024pt"
CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa_loader \
    --model-path ${MODEL_PATH} \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /gruntdata/heyuan4/workspace/public_datasets/cv/doin/vqa/TextVQA/train_images \
    --answers-file ./${MODEL_PATH}/textvqa/prediction.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file /gruntdata/heyuan4/workspace/public_datasets/cv/doin/vqa/TextVQA/TextVQA_0.5.1_val.json \
    --result-file ./${MODEL_PATH}/textvqa/prediction.jsonl
