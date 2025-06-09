import logging
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
import os
from utils.utils import Utils
import time


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    
    info_handler = logging.StreamHandler()
    info_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    # logger.addHandler(info_handler)
    logger.addHandler(fh)
    return logger

class SotaModel:
    def __init__(self):
        # your task is to generate only the next line
        self.prompt = '''
You are a code completion assistant. Given a Python code prefix, your task is to generate the next few lines of code that would logically follow the prefix. Stop at the first newline character. Do not include explanations or additional output.

Only output the generated code between the special tokens <output_start> and <output_end>.

Here is the code prefix:

<prefix>

Complete the code (do not repeat the prefix):
'''
        self.client = OpenAI(
            base_url='*',
            api_key="*"
        )
        self.model="claude-3-7-sonnet-20250219"

    def complete(self, prefix):
        content = self.prompt.replace("<prefix>", prefix)
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": content}
            ],
            max_tokens=128
        )
        return self._process_output(completion.choices[0].message.content)
    
    def _process_output(self, output):
        start = output.find("<output_start>") + len("<output_start>")
        if "<output_end>" in output:
            end = output.find("<output_end>")
        elif "</output_end>" in output:
            end = output.find("</output_end>")
        elif "</output_start>" in output:
            end = output.find("</output_start>")
        else:
            end = len(output)
        ans = output[start:end]
        return ans

class ApiModel:
    def __init__(self, m_type='n3_ds'):
        self.url_dict = {
            'v6_ds': {
                'url': "http://*:8004/v1",
                'model_name': '/nasdata/Model/deepseek-coder-6.7b-base',
                'model': 'deepseek'
            },
            'n3_ds': {
                'url': "http://localhost:8011/v1",
                'model_name': '/nasdata/Model/deepseek-coder-6.7b-base',
                'model': 'deepseek'
            },
            'n3_cl': {
                'url': "http://localhost:8010/v1",
                'model_name': '/nasdata/Model/CodeLlama-7b-hf',
                'model': 'codellama'
            },
            'n3_sc': {
                'url': "http://localhost:8009/v1",
                'model_name': '/home/dengle/starcoder2-7b',
                'model': 'starcoder'
            },
        }
        self.config = self.url_dict[m_type]
        self.client = OpenAI(api_key="*", base_url=self.config['url'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        
    def complete(self, prompt, retries=3, max_tokens=128):
        for i in range(retries):
            try:
                completion = self.client.completions.create(
                    model=self.config['model'],
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0,
                    stream=False
                )
                content = completion.choices[0].text
                return content
            except Exception as e:
                print(e)
                if i < retries - 1:  # 如果不是最后一次尝试，则等待一段时间后重试
                    sleep_time = 2 * (2 ** i)
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    return ''
                

def generate_codes(in_path, out_path, logger: logging.Logger, m_type='n3_ds'):
    logger.info(f'generates: {in_path}')
    in_file = open(in_path, encoding='utf-8')
    lines = in_file.readlines()
    in_file.close()
    o_prev = []
    o_prev_set = set()
    if os.path.exists(out_path):
        print(out_path)
        o_prev = Utils.load_jsonl(out_path)
        o_prev_set = set([i['metadata']['task_id'] for i in o_prev])

    if m_type == 'sota':
        model = SotaModel()
    else:
        model = ApiModel(m_type)
        tokenizer = model.tokenizer
    logger.info('loaded model')
    for line in tqdm(lines):
        # 生成代码
        example_json = json.loads(line)
        task_id = example_json['metadata']['task_id']
        if task_id in o_prev_set:
            logger.info(f'{task_id}: cahce')
            continue
        prompt = example_json['prompt']
        # print(example_json['metadata']['task_id'])
        if m_type != 'sota':
            input_ids = tokenizer(prompt, return_tensors="pt")
            if len(input_ids[0]) > 8000:
                logger.info(f'{task_id}: too many tokens')
                continue
        # logger.info('generating...')
        completion = model.complete(prompt)

        example_json['pred_res'] = completion
        choices = []
        res_lines = [line.strip() for line in completion.splitlines() if line.strip()]
        if len(res_lines) == 0:
            res_lines.append("")
        choice = {
            'text': res_lines[0]
        }
        choices.append(choice)
        example_json['choices'] = choices
        
        with open(out_path, mode='a', encoding='utf-8') as f:
            f.write(json.dumps(example_json) + '\n')
    


if __name__ == '__main__':

    m_type = 'sota'
    benchmark = 'sota_test'
    file_names = ['rc_python.jsonl', 'ours_python.jsonl']
    for file_name in file_names:
        in_path = f'prompts/{benchmark}/{file_name}'
        out_path = f'predictions/{benchmark}/{file_name}'

        logger = get_logger(log_file=f'logs/{file_name}.log')
        generate_codes(in_path, out_path, logger, m_type=m_type)