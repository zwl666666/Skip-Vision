#!/bin/bash
# Evaluate a trained Skip-Vision checkpoint on POPE and MME.
# Usage:
#   bash recipe/eval.sh MODEL_PATH COS_CONFIG_VERSION BACKBONE_RESOLUTION EVAL_ROOT
set -e

MODEL_PATH=${1:?usage: eval.sh MODEL_PATH COS_CONFIG_VERSION BACKBONE_RESOLUTION EVAL_ROOT}
COS_CONFIG_VERSION=${2:?missing COS_CONFIG_VERSION, e.g. v1_448_q16+64+256}
BACKBONE_RESOLUTION=${3:?missing BACKBONE_RESOLUTION, e.g. 448}
EVAL_ROOT=${4:?missing EVAL_ROOT, e.g. ./playground/data/eval}

POPE_QUESTIONS="$EVAL_ROOT/pope/llava_pope_test.jsonl"
POPE_IMAGES="$EVAL_ROOT/pope/val2014"
POPE_OUTPUT="$EVAL_ROOT/pope/answers/$(basename "$MODEL_PATH").jsonl"

mkdir -p "$EVAL_ROOT/pope/answers" "$EVAL_ROOT/mme/answers"

echo "[eval] POPE"
python -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --question-file "$POPE_QUESTIONS" \
    --image-folder "$POPE_IMAGES" \
    --answers-file "$POPE_OUTPUT" \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir "$EVAL_ROOT/pope/coco" \
    --question-file "$POPE_QUESTIONS" \
    --result-file "$POPE_OUTPUT"

echo "[eval] MME"
python -m llava.eval.model_vqa \
    --model-path "$MODEL_PATH" \
    --question-file "$EVAL_ROOT/mme/mme_test.jsonl" \
    --image-folder "$EVAL_ROOT/mme/images" \
    --answers-file "$EVAL_ROOT/mme/answers/$(basename "$MODEL_PATH").jsonl" \
    --temperature 0 \
    --conv-mode vicuna_v1
