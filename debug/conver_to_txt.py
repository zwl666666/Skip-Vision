import json
import os
data_info_file1 = "/gruntdata/heyuan4/workspace/zhangqinglong.zql/Lark/dataset/data_info/bailing-10B-glm3-hd-vqa-0825_data_info_short.json"
data_info_file2 = "/gruntdata/heyuan4/workspace/zhangqinglong.zql/Lark/dataset/data_info/bailing-10B-glm3-hd-vqa-0824_data_info_long.json"
data=["gpt4v_instruction_ocr","viscot_363k_oss_train","sharegpt4o_qa","sharegpt4o_qa_0819"]
final_data_list=[]
with open(data_info_file1, "r") as f1:
    data_info1 = json.load(f1)
with open(data_info_file2, "r") as f2:
    data_info2 = json.load(f2)
base_path="/gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/debug"
file_write_obj = open(os.path.join(base_path,"oss_path.txt"), 'w')
for eid, entry in enumerate(data_info1):
    if entry["data_name"] in data:
        print(entry["data_name"])
        print(entry['image_path'])
        with open(entry['text_file'], "r") as f:
            for line in f:
                data_line = json.loads(line.strip())
                print(data_line["image"][12:])
                file_write_obj.write(os.path.join(entry['image_path'],data_line["image"]))
                file_write_obj.write('\n')
                # final_data_list.append({"id":data_line["image"].split('.')[0],"image":os.path.join(entry['image_path'],data_line["image"]),"conversations":data_line["conversations"]})

