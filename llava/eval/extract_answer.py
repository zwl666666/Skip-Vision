import argparse

from tqdm import tqdm
from utilities import *
from llava.eval.ext_ans import demo_prompt

# api = ChatGPT()

number_conversion = {
    "zero": '0',
    "one": '1',
    "two": '2',
    "three": '3',
    "four": '4',
    "five": '5',
    "six": '6',
    "seven": '7',
    "eight": '8',
    "nine": '9',
    "ten": '10'
}

def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == '' or extraction is None:
        return False
    return True


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f'{query}\n\n{response}'
    full_prompt = f'{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: '
    return full_prompt


def extract_answer(response, problem, quick_extract=False, answer=None):
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']

    if response == '':
        return ''

    if question_type == 'multi_choice' and response in choices:
        return response

    if answer_type == 'integer':
        try:
            extraction = int(response)
            return str(extraction)
        except:
            pass

    if answer_type == 'float':
        try:
            extraction = str(float(response))
            return extraction
        except:
            pass

    # quick extraction
    if quick_extract:

        # print('Quickly extracting answer...', flush=True)
        # The answer is "text". -> "text"
        try:
            result = re.search(r'The answer is (.*)', response)
            if result:
                extraction = result.group(1)
                if extraction in number_conversion:
                    extraction = number_conversion[extraction]
                if extraction == answer:
                    pass
                else:
                    print(f"Response: {response}", flush=True)
                    print(f"Answer: {answer}", flush=True)
                    print(f"Extraction: {extraction}")
                    # print(f"{extraction==answer}")
                    # print(type(extraction))
                    # print(type(answer))
                    print("----------------")
                return extraction.replace("%", "").replace(',', '')
            else:
                print("--- Extraction: ''")
        except:
            pass


    # print("Running general extraction", flush=True)

    # general extraction
    # try:
    #     import time
    #     time.sleep(0.2)
    #     full_prompt = create_test_prompt(demo_prompt, query, response)
    #     extraction = api.generate(sys_prompt="", prompt=full_prompt)
    #     return extraction
    # except Exception as e:
    #     print(e)
    #     import traceback
    #     print(traceback.format_exc())
    #     print(f"Full_prompt: {full_prompt}")
    #     print(f'Error in extracting answer for {pid}', flush=True)

    return ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--output_file', type=str, default='mathvista_answer.json')
    parser.add_argument('--response_label', type=str, default='response', help='response label for the input file')
    # model
    parser.add_argument('--llm_engine', type=str, default='gpt-4-0613', help='llm engine',
                        choices=['gpt-3.5-turbo', 'gpt-3.5', 'gpt-4', 'gpt-4-0314', 'gpt-4-0613'])
    parser.add_argument('--number', type=int, default=-1, help='number of problems to run')
    parser.add_argument('--quick_extract', action='store_true', help='use rules to extract answer for some problems')
    parser.add_argument('--rerun', action='store_true', help='rerun the answer extraction')
    # output
    parser.add_argument('--save_every', type=int, default=10, help='save every n problems')
    parser.add_argument('--output_label', type=str, default='', help='label for the output file')
    args = parser.parse_args()

    # args
    label = args.response_label
    # result_file = os.path.join(args.output_dir, args.output_file)
    result_file = args.output_file

    if args.output_label != '':
        output_file = result_file.replace('.json', f'{args.output_label}.json')
    else:
        output_file = result_file

    # read results
    print(f'Reading {result_file}...')
    results = read_json(result_file)

    # full pids
    full_pids = list(results.keys())
    if args.number > 0:
        full_pids = full_pids[:min(args.number, len(full_pids))]
    print('Number of testing problems:', len(full_pids))

    # test pids
    if args.rerun:
        test_pids = full_pids
    else:
        test_pids = []
        for pid in full_pids:
            # print(pid)
            if 'extraction' not in results[pid] or not verify_extraction(results[pid]['extraction']):
                test_pids.append(pid)

    test_num = len(test_pids)
    print('Number of problems to run:', test_num)
    # print(test_pids)

    # tqdm, enumerate results
    for i, pid in enumerate(tqdm(test_pids)):
        problem = results[pid]

        assert label in problem
        response = problem[label]
        answer = problem['answer']
        question_type = problem['question_type']
        if question_type == "multi_choice":
            for answer_idx, r in enumerate(problem['choices']):
                if r == answer:
                    answer = ['A', 'B', 'C', 'D', 'E', 'F', 'G'][answer_idx]
                    break

        extraction = extract_answer(response, problem, args.quick_extract, answer)
        results[pid]['extraction'] = extraction

        if i % args.save_every == 0 or i == test_num - 1:
            print(f'Saving results to {output_file}...')
            save_json(results, output_file)
            print(f'Results saved.')
