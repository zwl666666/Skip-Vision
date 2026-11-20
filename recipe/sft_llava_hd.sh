#!/bin/bash
# Skip-Vision on the LLaVA-HD backbone (CLIP ViT-L/336px, LLaMA3-8B-Instruct):
# dynamic 336-672px resolution, Nr=576 / Ns=576 (Table 2, LLaVA-HD setting).
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH=${MODEL_PATH:-./checkpoints/pretrained/llama-3-8b-instruct}
DATA_PATH=${DATA_PATH:-./playground/data/llava_v1_5_mix665k.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-./playground/data/image_folder.json}
PATH_CONVERSION=${PATH_CONVERSION:-./playground/data/ocrvqa_path-conversion.json}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/sft/sv-llava-hd}
NUM_GPUS=${NUM_GPUS:-8}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
LEARNING_RATE=${LEARNING_RATE:-1e-5}

cd "$ROOT"
deepspeed --num_gpus="$NUM_GPUS" llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path "$MODEL_PATH" \
    --version v1 \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --path_conversion "$PATH_CONVERSION" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints 336x336,672x672 \
    --mm_patch_merge_type spatial_unpad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION:-1}" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate "$LEARNING_RATE" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none
