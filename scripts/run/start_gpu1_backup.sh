echo `date` + "---start_gpu1_backup" >> "scripts/run/run_log_1.log";

# bash scripts/v1_5/finetune_13b_run3.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos_pt2ep_sft2ep.sh;

# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints/sft/cos-clipvit-unlock_vision-448-q1296-e024pt/eval_scripts/mmbench.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints/sft/cos-clipvit-unlock_vision-448-q1296-e024pt/eval_scripts/mme.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints/sft/cos-clipvit-unlock_vision-448-q1296-e024pt/eval_scripts/pope.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints/sft/cos-clipvit-unlock_vision-448-q1296-e024pt/eval_scripts/textvqa.sh;

bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints/sft/llama3/cos-clipvit-unlock_vision-448-q1296-60Mpt-datav2/eval_scripts/gqa.sh;
echo `date` + "finished gpu1_backup" >> "scripts/run/run_log_1.log";

bash scripts/run/start_gpu1.sh;
