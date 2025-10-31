#!/bin/bash
cd /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main;
source activate llava;
node_idx=$RANK
world_size=$WORLD_SIZE
output_dir="/gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints_heyuan_nas2/weili/sft_linear_hd_672"

mkdir -p ${output_dir}

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /gruntdata/heyuan4/workspace/pishi.hzy/pretrained_models/llama3-8b-instruct/unsloth__llama-3-8b-Instruct \
    --version llama3 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data/image_folder.json \
    --path_conversion ./playground/data/ocrvqa_path-conversion.json \
    --pretrain_mm_mlp_adapter /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints/pretrain/llama3/linear/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --use_hd True \
    --hd_data_type True \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --image_aspect_ratio hd_pad \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3500 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to "none" \
    --vision_tower /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints/pretrained/openai__clip-vit-large-patch14-336 \
    # &> ${log_dir}/node_${node_idx}.log &
