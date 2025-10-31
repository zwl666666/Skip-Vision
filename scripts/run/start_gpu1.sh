echo `date` + "---start_gpu1" >> "scripts/run/run_log_1.log";

# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints/sft/llava-v1.5-7b_lr1/gqa.sh;
# bash scripts/experiments/ft_7b_lr1.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/pt_cos.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/pretrain/pt_cos.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos.sh;

# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos_unlock_vision.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/v1_5/finetune_13b_run2.sh;

# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/pretrain/pt_cos_2ep.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/pretrain/pt_cos_2ep.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/pretrain/pt_cos_2ep.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/pretrain/pt_cos_2ep.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/pretrain/pt_cos_2ep.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/pretrain/pt_cos_2ep.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos_pt2ep.sh;
bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/llama3/sft_cos_clipvit-448-q80baseline_datav2.sh;
echo `date` + "finished gpu1" >> "scripts/run/run_log_1.log";

bash scripts/run/start_gpu1_backup.sh;
