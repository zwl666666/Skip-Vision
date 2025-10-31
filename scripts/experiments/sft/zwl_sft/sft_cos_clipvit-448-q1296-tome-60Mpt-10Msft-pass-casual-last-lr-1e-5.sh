#!/bin/bash
/bin/bash -c "mkdir -p /heyuan_56_0630 && mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport alipay-heyuan-56-jht88.cn-heyuan-alipay.nas.aliyuncs.com:/ /heyuan_56_0630"
cd /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main;
source activate llava;

sleep 6

export LOGLEVEL=INFO
export MASTER_ADDR='11.235.18.162'
export WORLD_SIZE=4
export RANK=$1
node_index=$RANK

export ATORCH=1

# export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=22
export NCCL_IB_SL=5
export NCCL_IB_TC=136
export NCCL_IB_HCA=mlx5
export NCCL_DEBUG=INFO

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,5,6,8
export MASTER_PORT=14526
export HOSTE_NODE_ADDR=${MASTER_ADDR}:${MASTER_PORT}
node_idx=$RANK
world_size=$WORLD_SIZE

exp_id="cos-clipvit-unlock_vision-448-q1296-tome256-60Mpt-10Msft-pass-casual-last-lr-1e-5-ld-32gpu"
log_dir=/gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/logs/experiments/sft/zwl_sft/$exp_id
output_dir=/gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints_heyuan_nas2/weili/$exp_id
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FSDP_CPU_RAM_EFFICIENT_LOADING=1
mkdir -p ${log_dir}
mkdir -p ${output_dir}

nohup python -m atorch.distributed.launch --nproc_per_node=8 --nnode=${world_size} --master_addr=$MASTER_ADDR \
--master_port=$MASTER_PORT llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /gruntdata/heyuan4/workspace/pishi.hzy/pretrained_models/llama3-8b-instruct/unsloth__llama-3-8b-Instruct \
    --version llama3 \
    --data_path ./playground/data/llava_mix10m.json \
    --image_folder ./playground/data/image_folder.json \
    --path_conversion ./playground/data/ocrvqa_path-conversion.json \
    --pretrain_mm_mlp_adapter ./checkpoints/pretrain/llama3/e003-cos-8b/mm_projector.bin \
    --mm_projector_type chain-of-sight \
    --cos_config_version v2_448_q16+256+1024 \
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
    --save_steps 100000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --layerwise_decay True \
    --lr_scale 0.9 \
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
    --cos_highres_extension_version res224to448_80to1296_v2\
    &> ${log_dir}/node_${node_idx}.log &

# deepspeed llava/train/train_mem.py \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path /gruntdata/heyuan4/workspace/pishi.hzy/pretrained_models/llama3-8b-instruct/unsloth__llama-3-8b-Instruct \
#     --version llama3 \
#     --data_path ./playground/data/llava_mix10m.json \
#     --image_folder ./playground/data/image_folder.json \
#     --path_conversion ./playground/data/ocrvqa_path-conversion.json \
#     --pretrain_mm_mlp_adapter ./checkpoints/pretrain/llama3/e003-cos-8b/mm_projector.bin \
#     --mm_projector_type chain-of-sight \
#     --cos_config_version v1_448_q16+64 \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ${log_dir} \
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
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to "none" \
#     --vision_tower own_implementation_clip \
#     --backbone_resolution 448 \
#     --cos_highres_extension_version res224to448_80to80\
    # &> ${log_dir}/node_${node_idx}.log &

