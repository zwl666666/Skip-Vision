import json
import os
data_info_file1 = "/gruntdata/heyuan4/workspace/zhangqinglong.zql/Lark/dataset/data_info/bailing-10B-glm3-hd-vqa-0825_data_info_short.json"
data_info_file2 = "/gruntdata/heyuan4/workspace/zhangqinglong.zql/Lark/dataset/data_info/bailing-10B-glm3-hd-vqa-0824_data_info_long.json"
close_data1=["aokvqa_train","vqav2","mapqa_train","figureqa","plotqa","ocrvqa","dvqa","chart_to_text","docvqa","raven_train","iam_crop","screenqa_79k","idocvqa","tabfact","k12_printing_train","mathocr_v2","GeoGPT4V-1.1","image_textualization_train","textcaps","mathv360k","vizwiz_20k","chartbench","tallyqa_250k","viscot_363k_train_other","viscot_363k_train_text_doc","viscot_363k_oss_train","ICDAR19_ReCTS","RCTW","SynthText-800K","clevr_math_train_val","chartgemma","sharegpt4o_qa","sharegpt4o_qa_0819","mmc","agent_prompt","com_vint_en","com_vint_zh","vg_vqa_en","biaopan_train","biaopan_train_part2","clock","bottle","scene_ocr_crop","gpt4gen_rd_cot","gpt4v_textvqa_chuanyang_refine_30k","clock_part2","coco_counter_train","RCTW","cog_vlm_sft_single_zh","cog_vlm_sft_multi_zh","cog_vlm_sft_detail_zh","clock_1_2w"]
oss_data=["gpt4v_instruction_ocr","viscot_363k_oss_train","sharegpt4o_qa","sharegpt4o_qa_0819"]
final_data_list=[]
with open(data_info_file1, "r") as f1:
    data_info1 = json.load(f1)

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
                for line in f:
                    data_line = json.loads(line.strip())
                    conv,count = process_string(data_line["conversations"])
                    if count>1:
                        print(count,entry["data_name"])
                    final_data_list.append({"id":data_line["image"].split('.')[0],"image":os.path.join(entry['image_path'],data_line["image"]),"conversations":conv,"llava":"no"})
        else:
            print(entry["data_name"])
            pache_path='pcache://mmodalmastertwoproxy-pool.cz50c.alipay.com:39999/mnt/13fbe490fcb1e47a844d59a3d8da408b'
            with open(entry['text_file'], "r") as f:
                for line in f:
                    data_line = json.loads(line.strip())
                    conv,count = process_string(data_line["conversations"])
                    if count>1:
                        print(count,entry["data_name"])
                    final_data_list.append({"id":data_line["image"].split('.')[0],"image":os.path.join(pache_path,data_line["image"][12:]),"conversations":conv,"llava":"no"})

llava_path = '/gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/playground/data/llava_v1_5_mix665k.json'
llava_data_list = json.load(open(llava_path, "r"))
final_data_list = final_data_list+llava_data_list
filename = '/gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/playground/data/llava_mix1m.json'

# 使用 with 语句打开文件并写入数据
with open(filename, 'w') as jsonfile:
    # 将数据写入 JSON 文件
    json.dump(final_data_list, jsonfile, indent=4)
