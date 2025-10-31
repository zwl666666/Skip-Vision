#!/bin/bash
cd /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main;
source activate llava;
node_idx=$RANK
world_size=$WORLD_SIZE
exp_id="cos-clipvit-unlock_vision-448-q1104-80ffn-60Mpt-ln-add-q-causal-squeeze-pass-o_32gpu"
log_dir=/gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/logs/experiments/sft/zwl_sft/$exp_id
output_dir=/gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints_heyuan_nas2/weili/$exp_id

mkdir -p ${log_dir}
mkdir -p ${output_dir}

nohup torchrun --nproc_per_node=8 --nnode=${world_size} --node_rank=${node_idx} --master_addr=$MASTER_ADDR \
--master_port=$MASTER_PORT llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /gruntdata/heyuan4/workspace/pishi.hzy/pretrained_models/llama3-8b-instruct/unsloth__llama-3-8b-Instruct \
    --version llama3 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data/image_folder.json \
    --path_conversion ./playground/data/ocrvqa_path-conversion.json \
    --pretrain_mm_mlp_adapter ./checkpoints/pretrain/llama3/e003-cos-8b/mm_projector.bin \
    --mm_projector_type chain-of-sight \
    --cos_config_version v1_448_q16+64+1024 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to "none" \
    --vision_tower own_implementation_clip \
    --backbone_resolution 448 \
    --cos_highres_extension_version res224to448_80to1104\
    &> ${log_dir}/node_${node_idx}.log &

# deepspeed llava/train/train_mem.py \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path /gruntdata/heyuan4/workspace/pishi.hzy/pretrained_models/llama3-8b-instruct/unsloth__llama-3-8b-Instruct \
#     --version llama3 \
#     --data_path ./playground/data/llava_v1_5_mix665k.json \
#     --image_folder ./playground/data/image_folder.json \
#     --path_conversion ./playground/data/ocrvqa_path-conversion.json \
#     --pretrain_mm_mlp_adapter ./checkpoints/pretrain/llama3/e003-cos-8b/mm_projector.bin \
#     --mm_projector_type chain-of-sight \
#     --cos_config_version v1_448_q16+64+1024 \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ${output_dir} \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 8 \
#     --lazy_preprocess True \
#     --report_to "none" \
#     --vision_tower own_implementation_clip \
#     --backbone_resolution 448 \
#     --cos_highres_extension_version res224to448_80to1104\
    # &> ${log_dir}/node_${node_idx}.log &

