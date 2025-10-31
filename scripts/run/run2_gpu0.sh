echo `date` + "---run2_gpu0" >> "scripts/run/run_log_0.log";

# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/llama3/sft_cos_clipvit-448-q80.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos_clipvit-448-q1296-e024pt.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints/sft/cos-clipvit-unlock_vision-448-q1296-e024pt/eval_scripts/gqa.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/pretrain/llama3/pt_cos_clipvit_q80baseline.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints/sft/llama3/cos-clipvit-unlock_vision-448-q1296-60M_highres_pt/eval_scripts/gqa.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints/sft/llama3/cos-clipvit-unlock_vision-448-q1296/eval_scripts/gqa.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/llama3/sft_cos_clipvit-448-q80baseline.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/v1_5/finetune_13b_run2.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos_clipvit-448-q336-e024pt.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints/sft/cos-clipvit-unlock_vision-448-q336-e024pt/eval_scripts/gqa.sh;
# bash scripts/v1_5/finetune_13b_run3.sh;

# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/llama3/sft_cos_clipvit-448-q80-60Mpt_datav2.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints_heyuan_nas2/sft/llama3/cos-clipvit-unlock_vision-448-q1296-60M_highres_pt-datav2/eval_scripts/gqa.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/llama3/sft_cos_clipvit-448-q80.sh;

bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/pretrainv2/pretrain_mlp_llama3-8b.sh;

bash scripts/run/run_gpu0.sh;

