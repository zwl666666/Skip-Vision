echo `date` + "---run_gpu0" >> "scripts/run/run_log_0.log";

# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints/sft/llava-v1.5-7b/vqav2.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints/sft/llava-v1.5-7b/gqa.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/pretrain/pt_cos_wd0.05.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos_ptwd0.05.sh;

# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos_unlock_vision_448.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos_unlock_vision_448_vwd0.05.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos_unlock_vision_448_q336.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/pretrain/pt_cos_clipvit.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos_clipvit.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos_clipvit.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos_unlock_vision_448_vwd0.05_nopad.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos_unlock_vision_448.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos_clipvit-448-q80.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos_clipvit-448-q336.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos_clipvit-448-q80-e024pt.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos_clipvit-448-q1296.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos_clipvit-224-q80-e024pt-locked.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints/sft/cos-clipvit-unlock_vision-448-q80-e024pt/eval_scripts/gqa.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/sft_cos_clipvit-448-q1296-e024pt.sh;

# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/pretrain/pt_cos_unlock.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/pretrain/llama3/pt_cos_clipvit.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/sft/llama3/sft_cos_clipvit-448-q80.sh;

# bash ... ;

# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/pretrainv2/pretrain_mlp_llama3-8b.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/pretrainv2/pretrain_mlp_llama3-8b-sft-lr2.sh;
# bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/pretrainv2/pretrain_mlp_llama3-8b-unlock.sh;
bash /gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/scripts/experiments/pretrainv2/pretrain_mlp_llama3-8b-unlock-sft.sh;

bash scripts/run/run2_gpu0.sh;
