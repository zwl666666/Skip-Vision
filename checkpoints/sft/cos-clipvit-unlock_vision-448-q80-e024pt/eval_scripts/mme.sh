#!/bin/bash

source activate llava
MODEL_PATH="checkpoints/sft/cos-clipvit-unlock_vision-448-q80-e024pt"
CUDA_VISIBLE_DEVICES=5 python -m llava.eval.model_vqa_loader \
    --model-path ${MODEL_PATH} \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./${MODEL_PATH}/mme/prediction.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_answer_to_mme.py --res_dir ./${MODEL_PATH}/mme/

python playground/data/eval/MME/eval_tool/calculation.py --results_dir ./${MODEL_PATH}/mme/
