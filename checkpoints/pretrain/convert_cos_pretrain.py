
import os
import torch


############ config ############
# src_file = "/gruntdata/heyuan4/workspace/pishi.hzy/LAVIS-main/lavis/output/ELIP_VL_v2/coyo4m/vicuna7b/e024_multitaskpt_gqa_4m_shortprompt_init_text_ocrfull_stdoc_vqa0.8_okx3_q16_64_plainv2_3ep_v2_w16_4_lora_r16a16_aok_correct_novqg_lr2_cc12m_ref_l22_nolora/20240726162/checkpoint_0.pth"
# tgt_loc = "e024-cos-7b"
# tgt_base = "/gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints/pretrain"
# 
# src_file = "/gruntdata/heyuan4/workspace/pishi.hzy/LAVIS-main/lavis/output/ELIP_VL_v2/coyo4m/llama3-8b/e003_nolora/20240813061/checkpoint_0.pth"
# tgt_loc = "e003-cos-8b"
# tgt_base = "/gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints/pretrain/llama3"
# 
src_file = "/gruntdata/heyuan4/workspace/pishi.hzy/LAVIS-main/lavis/output/ELIP_VL_v2/coyo4m/llama3-8b/e003_nolora/final_config_v4_fromv2_highres_1296_run2/20240815211/checkpoint_0.pth"
tgt_loc = "e003-cos-8b-highres"
tgt_base = "/gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints/pretrain/llama3"
############ config ############

template_file = "/gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/checkpoints/pretrain/cos-7b-clipvit/mm_projector.bin"
tgt_file = "mm_projector.bin"


tgt_loc = os.path.join(tgt_base, tgt_loc)
os.makedirs(tgt_loc, exist_ok=True)

src_ckp = torch.load(src_file, map_location="cpu")
template_ckp = torch.load(template_file, map_location="cpu")

src_ckp_processed = {}

not_matched_keys = 0
matched_keys = 0
for k, v in src_ckp['model'].items():
    if "inv_freq" in k:
        continue
    if k.replace("visual_encoder.post_processor.tuner", "model.mm_projector") in template_ckp.keys() or ('highres' in tgt_loc and 'visual_encoder.post_processor.tuner' in k):
        new_k = k.replace("visual_encoder.post_processor.tuner", "model.mm_projector")
        if 'highres' not in tgt_loc:
            assert v.shape == template_ckp[new_k].shape
        src_ckp_processed[new_k] = v
        print(f"{k} -> {new_k}, {v.shape}")
        matched_keys += 1
    elif k.replace("vision_to_text_proj", "model.mm_projector.out_proj") in template_ckp.keys() or ('highres' in tgt_loc and 'vision_to_text_proj' in k):
        new_k = k.replace("vision_to_text_proj", "model.mm_projector.out_proj")
        if 'highres' not in tgt_loc:
            assert v.shape == template_ckp[new_k].shape
        src_ckp_processed[new_k] = v
        print(f"{k} -> {new_k}, {v.shape}")
        matched_keys += 1
    elif k.replace("ln_vision", "model.mm_projector.ln_out") in template_ckp.keys() or ('highres' in tgt_loc and 'ln_vision' in k):
        new_k = k.replace("ln_vision", "model.mm_projector.ln_out")
        if 'highres' not in tgt_loc:
            assert v.shape == template_ckp[new_k].shape
        src_ckp_processed[new_k] = v
        print(f"{k} -> {new_k}, {v.shape}")
        matched_keys += 1
    elif 'visual_encoder.base_encoder' in k:
        new_k = k.replace('visual_encoder.base_encoder', 'vision_tower')
        print(f"{k} -> {new_k}, {v.shape}")
        src_ckp_processed[new_k] = v
        matched_keys += 1
    else:
        print(f"{k} -> {k.replace('visual_encoder.post_processor.tuner', 'model.mm_projector')} not matched!!")
        not_matched_keys += 1
print(f"found {len(template_ckp.keys())} keys in template_ckp")
print(f"found {len(src_ckp_processed.keys())} in src_ckp")

print(f"convert finished. {matched_keys} matched keys, {not_matched_keys} not matched keys.")

torch.save(src_ckp_processed, os.path.join(tgt_loc, tgt_file))