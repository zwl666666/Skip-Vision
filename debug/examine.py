
import json
import csv
data_info_file1 = "/gruntdata/heyuan4/workspace/zhangqinglong.zql/Lark/dataset/data_info/bailing-10B-glm3-hd-vqa-0825_data_info_short.json"
data_info_file2 = "/gruntdata/heyuan4/workspace/zhangqinglong.zql/Lark/dataset/data_info/bailing-10B-glm3-hd-vqa-0824_data_info_long.json"
close_data1=["agent_prompt","com_vint_en","com_vint_zh","vg_vqa_en","biaopan_train","biaopan_train_part2","clock","bottle","scene_ocr_crop","gpt4gen_rd_cot","gpt4v_textvqa_chuanyang_refine_30k","clock_part2","coco_counter_train","RCTW","cog_vlm_sft_single_zh","cog_vlm_sft_multi_zh","cog_vlm_sft_detail_zh","clock_1_2w"]
close_data2=["co_instruct_multi_image","idl_gtp4","gpt4v_cn_ocr_caption","lrv_caption_en","lrv_caption_zh","dt_vqa","textocr_gpt4o_train","cninfo_500k","handwritten_0608_train","hfcard_dewarp_11k","THU_CTW","text_ocr_train","annual_report","eval_report","research_report","web_images","voucher","allava_vqa_zh","hme_train_100k","badcase_for_train"]

with open(data_info_file1, "r") as f1:
    data_info1 = json.load(f1)
with open(data_info_file2, "r") as f2:
    data_info2 = json.load(f2)
total_length = 0
filename = '/gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/debug/output_long.csv'

for eid, entry in enumerate(data_info1):
    if entry["data_name"] not in close_data1 and entry["data_name"] not in close_data2:
        with open(entry['text_file'], "r") as f:
            lines = f.readlines()
        length_entry = len(lines)
        data_name = entry['data_name']
        total_length += length_entry

for eid, entry in enumerate(data_info2):
    if entry["data_name"] not in close_data1 and entry["data_name"] not in close_data2:
        with open(entry['text_file'], "r") as f:
            lines = f.readlines()
        length_entry = len(lines)
        data_name = entry['data_name']
        total_length += length_entry
print(f"Read {eid+1} datasets, total length: {total_length}")
