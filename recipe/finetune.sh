#!/bin/bash
# Skip-Vision stage 2: full-model instruction tuning on the LLaVA v1.5 mixture.
# Default uses a 448px CLIP backbone resampled to 336 visual tokens.
set -e

MODEL_PATH=${MODEL_PATH:-./checkpoints/pretrained/lmsys__vicuna-7b-v1.5}
VISION_TOWER=${VISION_TOWER:-own_implementation_clip}
PRETRAIN_ADAPTER=${PRETRAIN_ADAPTER:-./checkpoints/pretrain/skip-vision-7b-projector/mm_projector.bin}
DATA_PATH=${DATA_PATH:-./playground/data/llava_v1_5_mix665k.json}
IMAGE_FOLDER=${IMAGE_FOLDER:-./playground/data/image_folder.json}
PATH_CONVERSION=${PATH_CONVERSION:-./playground/data/ocrvqa_path-conversion.json}
OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/sft/skip-vision-7b-448-q336}
COS_CONFIG_VERSION=${COS_CONFIG_VERSION:-v1_448_q16+64+256}
BACKBONE_RESOLUTION=${BACKBONE_RESOLUTION:-448}
COS_HIGHRES_EXTENSION=${COS_HIGHRES_EXTENSION:-res224to448_80to336}
NUM_GPUS=${NUM_GPUS:-1}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-1}
LEARNING_RATE=${LEARNING_RATE:-2e-5}

deepspeed --num_gpus="$NUM_GPUS" llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path "$MODEL_PATH" \
    --version v1 \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --path_conversion "$PATH_CONVERSION" \
    --vision_tower "$VISION_TOWER" \
    --pretrain_mm_mlp_adapter "$PRETRAIN_ADAPTER" \
    --mm_projector_type chain-of-sight \
    --cos_config_version "$COS_CONFIG_VERSION" \
    --backbone_resolution "$BACKBONE_RESOLUTION" \
    --cos_highres_extension_version "$COS_HIGHRES_EXTENSION" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION" \
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
