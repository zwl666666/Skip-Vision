import json
import os
import oss2
from pcache_fileio import fileio
from pcache_fileio.pcache_manager import PcacheManager
data_info_file1 = "/gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/bailing-10B-glm3-hd-vqa-0825_data_info_short.json"
data_info_file2 = "/gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/bailing-10B-glm3-hd-vqa-0824_data_info_long.json"
close_data1=["agent_prompt","com_vint_en","com_vint_zh","vg_vqa_en","biaopan_train","biaopan_train_part2","clock","bottle","scene_ocr_crop","gpt4gen_rd_cot","gpt4v_textvqa_chuanyang_refine_30k","clock_part2","coco_counter_train","RCTW","cog_vlm_sft_single_zh","cog_vlm_sft_multi_zh","cog_vlm_sft_detail_zh","clock_1_2w"]
close_data2=["co_instruct_multi_image","idl_gtp4","gpt4v_cn_ocr_caption","lrv_caption_en","lrv_caption_zh","dt_vqa","textocr_gpt4o_train","cninfo_500k","handwritten_0608_train","hfcard_dewarp_11k","THU_CTW","text_ocr_train","annual_report","eval_report","research_report","web_images","voucher","allava_vqa_zh","hme_train_100k","badcase_for_train"]
oss_data=["gpt4v_instruction_ocr","viscot_363k_oss_train","sharegpt4o_qa","sharegpt4o_qa_0819"]
final_data_list=[]
with open(data_info_file1, "r") as f1:
    data_info1 = json.load(f1)
with open(data_info_file2, "r") as f2:
    data_info2 = json.load(f2)

def process_string(s):
    # 统计"<image>"出现的次数
    count=0
    for i,conv in enumerate(s):
        count = count+conv["value"].count("<image>")

        # 如果"<image>"出现次数大于1
        if count > 1:
            # 保留第一个"<image>"后面的内容
            s[i]["value"] = conv["value"].replace("<image>", "")

    return s,count
for eid, entry in enumerate(data_info1):
    if entry["data_name"] not in close_data1:
        if entry["data_name"] not in oss_data:
            print(entry["data_name"])
            with open(entry['text_file'], "r") as f:
                tmp = json.loads(f.readline().strip())
                if not os.path.exists(os.path.join(entry['image_path'],tmp["image"])):
                    print(os.path.join(entry['image_path'],tmp["image"]))
                for line in f:
                    data_line = json.loads(line.strip())
                    conv,count = process_string(data_line["conversations"])
                    final_data_list.append({"id":data_line["image"].split('.')[0],"image":os.path.join(entry['image_path'],data_line["image"]),"conversations":conv})
        else:
            print(entry["data_name"])
            pache_path='pcache://mmodalmastertwoproxy-pool.cz50c.alipay.com:39999/mnt/13fbe490fcb1e47a844d59a3d8da408b'
            with open(entry['text_file'], "r") as f:
                for line in f:
                    data_line = json.loads(line.strip())
                    conv,count = process_string(data_line["conversations"])
                    final_data_list.append({"id":data_line["image"].split('.')[0],"image":os.path.join(pache_path,data_line["image"][12:]),"conversations":conv})

for eid, entry in enumerate(data_info2):
    if entry["data_name"] not in close_data2:
        print(entry["data_name"])
        with open(entry['text_file'], "r") as f:
            tmp = json.loads(f.readline().strip())
            if not os.path.exists(os.path.join(entry['image_path'],tmp["image"])):
                print(os.path.join(entry['image_path'],tmp["image"]))
            for line in f:
                data_line = json.loads(line.strip())
                conv,count = process_string(data_line["conversations"])
                final_data_list.append({"id":data_line["image"].split('.')[0],"image":os.path.join(entry['image_path'],data_line["image"]),"conversations":conv})
print(len(final_data_list))
# filename = '/gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/playground/data/llava_mix10m.json'

# # 使用 with 语句打开文件并写入数据
# with open(filename, 'w') as jsonfile:
#     # 将数据写入 JSON 文件
#     json.dump(final_data_list, jsonfile, indent=4)
