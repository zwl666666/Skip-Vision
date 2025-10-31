import os
import re
import time
import torch
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode

from lavis.models import load_model_and_preprocess
from lavis.datasets.datasets.dfs_pretrain_mix_glm_datasets import DFSPretrainGLMDataset
BAILING_VERSION = os.getenv("BAILING_VERSION", "V2")

if BAILING_VERSION == "V2":
    from solutions.antllm.antllm.models.glm.tokenization_glm import GLMTokenizer
elif BAILING_VERSION == "V3":
    from solutions.antllm_0515.antllm.models.glm.tokenization_glm import GLMTokenizer

import argparse

os.environ["PRETRAIN_MODEL_PATH"] = "/gruntdata/heyuan4/workspace/kaixiang.jkx/pretrain_models"
os.environ[
    "BERT_LOCAL_PATH"] = "/gruntdata/heyuan4/workspace/kaixiang.jkx/pretrain_models/models--bert-base-uncased/snapshots/0a6aa9128b6194f4f3c4db429b6cb4891cdb421b"
os.environ["ADD_POS_EMB"] = "add_pos_emb"

# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class TestGLM:
    def __init__(
            self,
            pretrained_ckpt=None,
            finetuned_ckpt=None,
            glm_model_path=None,
            model_type="instruct_bailing_mm_unified_768_wo_lora",
            max_input_length=256,
            max_output_length=4096,
            max_gen_len=4096,
            left_truncate=True,
            rotary_type="1d",
            num_query_token=128,
            model_name="bailing_mm_unified"):
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name=model_name, model_type=model_type, is_eval=True, device=device
        )
        if pretrained_ckpt is not None:
            self.model.load_checkpoint(pretrained_ckpt)
        if finetuned_ckpt is not None:
            self.model.load_checkpoint(finetuned_ckpt)

        # self.model.glm_model = self.model.glm_model.merge_and_unload()
        self.model.glm_model.half()

        self.tokenizer = GLMTokenizer.from_pretrained(glm_model_path)
        self.mask_id = self.tokenizer.convert_tokens_to_ids('[gMASK]')
        self.unk_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)
        self.cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        self.pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.eop_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        self.sop_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sop_token)

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.left_truncate = left_truncate
        self.rotary_type = rotary_type
        self.num_query_token = num_query_token
        self.max_gen_len = max_gen_len

        # self.model.glm_model.dynamic_query=False

    def test_beit3_glm(self, samples, dynamic_query=False, hd_num=None):
        if 'text_input' not in samples:
            samples['text_input'] = samples['instruct_input'][0]
            # print(f"text_input: samples['text_input']")

        if "image" in samples:
            if not dynamic_query:
                features = self.build_samples(samples['text_input'], num_query_token=self.num_query_token)
            else:
                features = self.build_samples(samples['text_input'],
                                            num_query_token=int(samples["image"].shape[0]) * self.num_query_token,
                                            max_num_query_token=(hd_num+1) * self.num_query_token)
                samples['sample_split_window_cnt'] = [int(samples['image'].shape[0])]

            if hasattr(self.vis_processors['eval'], 'hd_resolution') and self.vis_processors['eval'].hd_resolution is True:
                image_hd = samples['image']
                image = F.resize(image_hd, (self.vis_processors['eval'].small_size, self.vis_processors['eval'].small_size),
                                InterpolationMode.BICUBIC)
                image = self.vis_processors['eval'].normalize(image)
                image_hd = self.vis_processors['eval'].normalize(image_hd)
                samples['image'] = image
                samples['hd_image'] = image_hd.to(torch.float32)
        else:
            features = self.build_samples(samples['text_input'], num_query_token=self.num_query_token)

        samples['input_ids'] = features['input_ids'].unsqueeze(0).to(device)
        samples['position_ids'] = features['position_ids'].unsqueeze(0).to(device)
        if BAILING_VERSION == "V2":
            samples['generation_attention_mask'] = features['generation_attention_mask'].unsqueeze(0).to(device)
            # print("use bailing 2.0!")
        if BAILING_VERSION == "V3":
            samples['attention_mask'] = features['attention_mask'].unsqueeze(0).to(device)
            # print("use bailing 3.0!")

        output_text = self.model.generate(samples, max_length=self.max_gen_len, num_beams=5)
        output_text = self.tokenizer.batch_decode(output_text)[0]

        if "<|startofpiece|> " in output_text:
            st = output_text.find("<|startofpiece|> ") + len("<|startofpiece|> ")
        else:
            st = output_text.find("<|startofpiece|>") + len("<|startofpiece|>")
        end = output_text.find("<|endoftext|>")
        pred_cap = output_text[st:end]

        return pred_cap

    def build_samples(self, prompt, num_query_token, max_num_query_token=None):
        features = DFSPretrainGLMDataset.build_feature_from_sample(
            tokenizer=self.tokenizer,
            instruction=prompt,
            for_generation=True,
            num_query_token=num_query_token,
            left_truncate=self.left_truncate,
            rotary_type=self.rotary_type,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
            cls_id=self.cls_id,
            mask_id=self.mask_id,
            unk_id=self.unk_id,
            sop_id=self.sop_id,
            eop_id=self.eop_id,
            pad_id=self.pad_id,
            max_num_query_token=max_num_query_token, )
        return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate bailing-mm')
    parser.add_argument('--model-path', type=str, default="")
    parser.add_argument('--max-epoch', type=int, default=3)
    parser.add_argument('--base-model', type=str, default="")
    parser.add_argument('--model-name', type=str, default="bailing_mm_unified")
    parser.add_argument('--model-type', type=str, default="instruct_bailing_mm_glm3_448")
    parser.add_argument('--mixed-processor', type=bool, default=False)
    parser.add_argument('--benchmarks', nargs='+', default=["mmbench-en"])
    args = parser.parse_args()

    # monitor directory
    pretrain_path = None
    if BAILING_VERSION == "V2":
        glm_model_path = "/gruntdata/heyuan4/workspace/zhangqinglong.zql/AntGMM/antmmf/prj/LAVIS-main/work_dirs/AntGLM-10B-RLHF-unified"
    elif BAILING_VERSION == "V3":
        glm_model_path = "/gruntdata/heyuan4/workspace/zhangqinglong.zql/AntGMM/antmmf/prj/LAVIS-main/work_dirs/bailing-3.0-10B-4K-Chat-unified"

    with_agent = False
    bin_ratio = 1 / 8
    is_unify = True

    print(f"model_name: {args.model_name}")

    model_names = ["bailing_mm_unified", "bailing_mm_unified_eva"]
    assert args.model_name in model_names, f"model: {args.model_name} are not in {model_names}"

    mm_model_types = [
        "instruct_bailing_mm_unified_768_wo_lora",
        "instruct_bailing_mm_unified_dynamic_wo_lora",
        "instruct_bailing_mm_unified_448_wo_lora",
        "instruct_bailing_mm_glm3_448",
        "instruct_bailing_mm_glm3_dynamic_hd",
        "instruct_bailing_mm_unified_dynamic_internvl",
        "instruct_bailing_mm_glm3_dynamic_internvl",
    ]
    if args.model_name == "bailing_mm_unified" and args.model_type not in mm_model_types:
        print("wrong bailing_mm_unified model type")

    clip_model_types = [
        "inference_bailing_mm_unified_clip_glmv3_chat",
        "inference_bailing_mm_unified_internvit6b_glm10b"
    ]

    if args.model_name == "bailing_mm_unified_eva" and args.model_type not in clip_model_types:
        print("wrong bailing_mm_eva model type")

    if args.model_type != "":
        model_type = args.model_type
    print(f"model_type: {model_type}")

    if "dynamic" in model_type:
        dynamic_query=True
        print("inference with dynamic query")
    else:
        dynamic_query=False

    if dynamic_query:
        max_gen_len = 4096
    else:
        max_gen_len = 1024

    # maximum monitor time, such as 36 hours
    # max_monitor_time = 3600 * 72
    max_monitor_time = 3600 * 0.5
    # once we have detected max_epoch.pth, exit
    # the checkpoint for skipping evaluating
    # executed_pth_list = [f"{i + 1}.pth" for i in range(args.max_epoch)]

    executed_pth_list = [args.base_model]
    hist_pth_list = ["checkpoint_best.pth"]
    start = time.time()
    pattern = re.compile(r'([\d]+).pth')
    while (time.time() - start) < max_monitor_time:
        for executed_pth in executed_pth_list:
            find = re.findall(pattern, executed_pth)
            # print(find)
        files = os.listdir(args.model_path)
        pth_list = [file for file in files if file.endswith('.pth')]
        flag = 0
        for pth in executed_pth_list[::-1]:
            if pth in pth_list and pth not in hist_pth_list:
                flag = 1
                break
        if flag == 0:
            print("No more checkpoint to be evaluated, waiting...", flush=True)
            time.sleep(20 * 60)
            continue
        cur_file = pth
        hist_pth_list.append(cur_file)
        instruct_path = os.path.join(args.model_path, cur_file)
        cur_dir = cur_file.split('.pth')[0]
        print('cur_test_ckpt:', instruct_path)

        if is_unify:
            model = TestGLM(
                pretrained_ckpt=pretrain_path,
                finetuned_ckpt=instruct_path,
                glm_model_path=glm_model_path,
                model_type=model_type,
                max_input_length=512,
                max_output_length=4096,
                max_gen_len=max_gen_len,
                num_query_token=256,
                model_name=args.model_name
            )
        else:
            model = TestGLM(
                pretrained_ckpt=pretrain_path,
                finetuned_ckpt=instruct_path,
                glm_model_path=glm_model_path)
        is_glm = True

        if dynamic_query and "hd12" in instruct_path.lower():
            model.vis_processors["eval"].hd_num = 12

        model.vis_processors["eval"].mixed_processor = args.mixed_processor

        if "llava-bench" in args.benchmarks:
            from evals.llava_bench.eval import evaluate_llava

            llava_data_path = "/gruntdata/heyuan4/workspace/zhangqinglong.zql/AntGMM/antmmf/prj/LAVIS-main/evals/llava-bench-in-the-wild/"
            question_file = os.path.join(llava_data_path, "questions.jsonl")

            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            os.makedirs(os.path.dirname(os.path.join(llava_data_path, "answers")), exist_ok=True)
            answer_file = "answer_" + pth.split(".")[0] + f"{time_prefix}.jsonl"
            answer_file = os.path.join(llava_data_path, "answers", answer_file)

            evaluate_llava(
                model,
                data_file=question_file,
                is_glm=is_glm,
                with_agent=with_agent,
                bin_ratio=bin_ratio,
                dynamic_query=dynamic_query,
                answers_file=answer_file
            )

        if "mme" in args.benchmarks:  # mmmu validation
            from evals.mme.eval import evaluate_mme

            data_path = "/gruntdata/heyuan4/workspace/zhangqinglong.zql/AntGMM/antmmf/prj/LAVIS-main/evals/mme/eval_tool/Your_Results"
            output_dir = "/gruntdata/heyuan4/workspace/zhangqinglong.zql/AntGMM/antmmf/prj/LAVIS-main/evals/mme/eval_tool/LaVIN"

            evaluate_mme(
                model,
                data_path=data_path,
                is_glm=is_glm,
                with_agent=with_agent,
                bin_ratio=bin_ratio,
                output_dir=output_dir
            )

        if "MMMU_validation" in args.benchmarks or "MMMU_test" in args.benchmarks:
            from evals.mmmu.evaluate_mmmu import evaluate_mmmu
            ds_names = []
            if "MMMU_validation" in args.benchmarks:
                ds_names.append("MMMU_validation")
            if "MMMU_test" in args.benchmarks:
                ds_names.append("MMMU_test")
            for ds_name in ds_names:
                evaluate_mmmu(
                    model,
                    instruct_path=instruct_path,
                    ds_name=ds_name,
                    is_glm=is_glm,
                    dynamic_query=dynamic_query
                )

        if "magnifier" in args.benchmarks:  # Magnifier Bench
            from evals.magnifier.magnifier_bench import evalulate_mag
            evalulate_mag(
                model,
                instruct_path=instruct_path,
                instruction_file="/gruntdata/heyuan4/workspace/kaixiang.jkx/data/MagnifierBench/data_instructions.json",
                images_file="/gruntdata/heyuan4/workspace/kaixiang.jkx/data/MagnifierBench/images.json",
                is_glm=is_glm,
                with_agent=with_agent,
                bin_ratio=bin_ratio)

        if "mm-vet" in args.benchmarks:  # Magnifier Bench
            # from evals.mmvet.eval import evaluate_mmvet
            from evals.mmvet.eval_v2 import evaluate_mmvet
            evaluate_mmvet(
                model,
                instruct_path=instruct_path,
                ds_name = "mm-vet",
                is_glm=is_glm,
                with_agent=with_agent,
                bin_ratio=bin_ratio,
                dynamic_query=dynamic_query
            )

        if "OCRBench" in args.benchmarks:
            from evals.ocrbench.evaluate_ocrbench import evaluate_ocrbench
            evaluate_ocrbench(
                model,
                instruct_path=instruct_path,
                is_glm=is_glm,
                dynamic_query=dynamic_query
            )

        if "MathVista_testmini" in args.benchmarks or "MathVista_test" in args.benchmarks:
            from evals_zy.mathvista.evaluate_mathvista import evaluate_mathvista
            ds_names = []
            if "MathVista_testmini" in args.benchmarks:
                ds_names.append("MathVista_testmini")
            if "MathVista_test" in args.benchmarks:
                ds_names.append("MathVista_test")
            for ds_name in ds_names:
                evaluate_mathvista(
                    model,
                    instruct_path=instruct_path,
                    ds_name = ds_name,
                    is_glm=is_glm,
                    dynamic_query=dynamic_query
                )
        if "ai2diagram_test" in args.benchmarks:
            from evals.ai2diagram.eval import evaluate_ai2d
            ds_name = "ai2diagram_test"
            evaluate_ai2d(
                model,
                instruct_path=instruct_path,
                ds_name = ds_name,
                is_glm=is_glm,
                dynamic_query=dynamic_query
            )

        # mmbench
        from evals.mmbench.eval import ds_collections, evalulate_mmb
        for ds_name in list(ds_collections.keys()):
            if ds_name in args.benchmarks:
                evalulate_mmb(
                    model,
                    instruct_path=instruct_path,
                    ds_name=ds_name,
                    sys_prompt="There are several options:",
                    is_glm=is_glm,
                    with_agent=with_agent,
                    bin_ratio=bin_ratio,
                    dynamic_query=dynamic_query
                )

        if "pope" in args.benchmarks:  # pope
            from evals.pope.pope_test import evalulate_pope

            evalulate_pope(
                model,
                instruct_path=instruct_path,
                data_file="/gruntdata/heyuan4/workspace/kaixiang.jkx/data/POPE/POPE/coco/coco_pope_random.json",
                image_dir="/gruntdata/heyuan_nas/workspace/caoyunhao.cyh/datasets/coco/images/val2014/",
                is_glm=is_glm,
                with_agent=with_agent,
                bin_ratio=bin_ratio,
                dynamic_query=dynamic_query
            )

        if "seed_bench" in args.benchmarks:  # seed bench
            from evals.seed.eval import evaluate_seed

            evaluate_seed(
                model,
                instruct_path=instruct_path,
                dev_file="/gruntdata/heyuan4/workspace/pishi.hzy/data_annotations/SEED-Bench/SEED-Bench-Img.json",
                is_glm=is_glm,
                dynamic_query=dynamic_query
            )

        if "science_qa" in args.benchmarks:  # science qa
            from evals.scienceqa.eval import evaluate_science_qa

            evaluate_science_qa(
                model,
                instruct_path=instruct_path,
                is_glm=is_glm,
                split="test",
                use_hint=True)

        if "HallusionBench" in args.benchmarks:  # hallusion bench
            from evals.HallusionBench.evaluation import evaluate_HallusionBench

            evaluate_HallusionBench(
                model,
                instruct_path=instruct_path,
                is_glm=is_glm,
                dynamic_query=dynamic_query)

        del model
        torch.cuda.empty_cache()
