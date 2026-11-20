#!/bin/bash
# Skip-Vision stage 1: train the Chain-of-Sight projector on 558K captions.
# The LLM stays frozen; only mm_projector receives gradients.
set -e

MODEL_PATH=${MODEL_PATH:-./checkpoints/pretrained/lmsys__vicuna-7b-v1.5}
VISION_TOWER=${VISION_TOWER:-own_implementation_clip}
COS_CONFIG_VERSION=${COS_CONFIG_VERSION:-v1}
DATA_PATH=${DATA_PATH:-./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-./playground/data/LLaVA-Pretrain/images}
PATH_CONVERSION=${PATH_CONVERSION:-./playground/data/ocrvqa_path-conversion.json}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/pretrain/skip-vision-7b-projector}
NUM_GPUS=${NUM_GPUS:-1}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-1e-3}

deepspeed --num_gpus="$NUM_GPUS" llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path "$MODEL_PATH" \
    --version plain \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --path_conversion "$PATH_CONVERSION" \
    --vision_tower "$VISION_TOWER" \
    --mm_projector_type chain-of-sight \
    --cos_config_version "$COS_CONFIG_VERSION" \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --is_pretrain True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
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
