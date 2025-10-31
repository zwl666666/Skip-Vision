import jsonlines
from argparse import ArgumentParser
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

from PIL import Image
import math
from llava.evals.mmstar.mmstar import MMStar_eval


data_file = "/gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/llava/evals/mmstar/data/mmstar_val.jsonl"
image_path = "/gruntdata/heyuan4/workspace/pishi.hzy/LLaVA-main/llava/evals/mmstar/data/"
from pcache_fileio import fileio
from pcache_fileio.pcache_manager import PcacheManager
def save_json(json_list, save_path):
    with open(save_path, 'w') as file:
        json.dump(json_list, file, indent=4)

def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def eval_model(args):
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model( model_path = model_path, model_base = args.model_base, model_name = model_name)
    output_dir = args.output_folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_file = f'mmstar.json'
    results_file = os.path.join(output_dir, results_file)
    ds = jsonlines.open(data_file)

    outputs = []
    for data in tqdm(ds):
        image_file = os.path.join(image_path, data["image_path"])
        qs = data["question"].replace("<image 1>", "image")
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        qs += "\nAnswer with the option's letter from the given choices directly."
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        img_pil = Image.open(image_file).convert("RGB")

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image_tensor = process_images([img_pil], image_processor, model.config)[0]
        hd_num=None
        if model.config.use_hd:
            hd_num = [len(image_tensor)]
            image_tensor = torch.stack(image_tensor)
            images=image_tensor.half().cuda()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    image_sizes=[img_pil.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    hd_num=hd_num)
        else:
            images=image_tensor.unsqueeze(0).half().cuda()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    image_sizes=[img_pil.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True
                    )

        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # prompt = data["question"] + " Answer with the option's letter from the given choices directly."


        data["prediction"] = response
        outputs.append(data)

    save_json(outputs, results_file)
    print('Results saved to {}'.format(results_file))
    return results_file

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_folder", type=str, default="./results")
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--save_name", type=str, default="llava1_5_7b")
    parser.add_argument("--conv_mode", type=str, default="llama-3")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    results_file = eval_model(args)

    MMStar_eval(results_file)
