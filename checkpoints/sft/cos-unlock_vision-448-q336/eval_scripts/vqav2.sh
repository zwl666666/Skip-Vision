#!/bin/bash
conda activate llava

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b"
SPLIT="llava_vqav2_mscoco_test-dev2015"
MODEL_PATH="./checkpoints/sft/cos"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${MODEL_PATH} \
        --question-file ./playground/data/eval/vqav2/$SPLIT.jsonl \
        --image-folder  /gruntdata/heyuan_nas/workspace/caoyunhao.cyh/datasets/coco/images/test2015 \
        --answers-file ./${MODEL_PATH}/vqav2/$SPLIT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./${MODEL_PATH}/vqav2/$SPLIT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./${MODEL_PATH}/vqav2/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

