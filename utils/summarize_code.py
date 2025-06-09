from concurrent.futures import ThreadPoolExecutor, as_completed
import os
# from summrize.summrize_unit import use_llm
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from utils.utils import Utils, UnixCoder
from openai import OpenAI
import time
from vllm import LLM, SamplingParams


class SummaryModel:
    def __init__(self, cuda=3):
        self.device = f'cuda:{cuda}'

        # model_name = "/home/dengle/deepseek-v2-lite-chat"
        model_name = "/data1/dengle/llama3-8b-instruct"
        # model_name = "/data/dengle/models/llama3-8b-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(self.device)
        # self.model.generation_config = GenerationConfig.from_pretrained(model_name)
        # self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

    def summarize_code(self, code):
        messages = [
            {"role": "system", "content": "You are an expert programmer"},
            {"role": "user", "content": code},
        ]
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        input_tensor = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        outputs = self.model.generate(input_tensor.to(self.model.device), eos_token_id=terminators, max_new_tokens=100,
                                      do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        result = self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        return result
        # print(summary)

        
class SummaryVllmModel:
    def __init__(self, cuda=[3,4]):
        device = ','.join(map(str, cuda))
        os.environ["CUDA_VISIBLE_DEVICES"]=device
        model_name = "/data1/dengle/llama3-8b-instruct"
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=len(cuda),
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            max_model_len=8192
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.sampling_params = SamplingParams(temperature=0, max_new_tokens=100)
    
    def summarize_code(self, code):
        sampling_params = SamplingParams(temperature=0, max_tokens=min(8192, len(self.tokenizer(code, return_tensors="pt")[0])+100))
        outputs = self.model.generate([code], sampling_params)
        generated_text = outputs[0].outputs[0].text
        return generated_text

class SummaryAPIModel:
    
    def __init__(self, url="http://localhost:8085/v1"):
        self.client = OpenAI(api_key="*", base_url=url)
        
    def summarize_code(self, prompt, retries=3):
        for i in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model="llama0",
                    messages=[
                        {"role": "system", "content": "You are an expert programmer"},
                        {"role": "user", "content": prompt},
                ],
                    max_tokens=100,
                    temperature=0,
                    stream=False
                )
                content = response.choices[0].message.content
                return content
            except Exception as e:
                if i < retries - 1:  # 如果不是最后一次尝试，则等待一段时间后重试
                    sleep_time = 2 * (2 ** i)
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print('Failed.')
                    return ''



def postprocess_llama(summary):
    summary = summary.lower().replace('here is the summary:', '').replace('summary:', '').replace(
        'summarize the code:', '').strip()
    return summary

def summary_codes(repos, lang, benchmark=None, summary_cuda=3):
    if summary_cuda == -1:
        model = SummaryAPIModel()
    else:
        model = SummaryModel(cuda=summary_cuda)
        
    prompt_template = open(f'./utils/{lang}.prompt').read()

    for repo in repos:
        out_path = f'./cache/func_base/{benchmark + "_" + repo}_summary.pkl'
        if os.path.exists(out_path):
            print(f'{repo}: summary cache')
            continue
        func_base = Utils.load_pickle(f'./cache/func_base/{benchmark + "_" + repo}.pkl')
        summary_list = []
        for func in tqdm(func_base, desc=f'process {repo}'):
            body = func['metadata']['func_body']
            prompt = prompt_template.replace('@{}@', body)
            func_summary = model.summarize_code(prompt).strip()
            func_summary = postprocess_llama(func_summary)
            summary_list.append(func_summary)
            # print(func_summary)

        Utils.dump_pickle(summary_list, out_path)

# def summary_code_use_llm(repos):
#     prompt_template = open('./prompt').read()

#     for repo in repos:
#         func_base = Utils.load_pickle(f'./cache/func_base/{repo}.pkl')
#         summary_list = []
#         for func in tqdm(func_base):
#             body = func['metadata']['func_body']
#             func_summary = use_llm(prompt_template.replace('@{}@', body))
#             summary_list.append(func_summary)
#             # print(func_summary)

#         Utils.dump_pickle(summary_list, f'./cache/func_base/{repo}_summary.pkl')


# def summary_one_code_use_llm(code):
#     prompt_template = open('./prompt').read()
#     func_summary = use_llm(prompt_template.replace('@{}@', code))
#     return func_summary


def process_not_has_e3(example):
    lines = [i.strip() for i in example.splitlines() if i.strip()]
    res = []
    # A:
    for line in lines:
        if not line.startswith('A:'):
            res.append(line)
        else:
            break
    if len(res) == 0:
        return None
    else:
        return '\n'.join(res).strip()


def encode_texts(repos, encode_cuda, benchmark=None):
    unixcoder_enc = UnixCoder(encode_cuda)
    for repo in repos:
        summary_list = Utils.load_pickle(f'./cache/func_base/{benchmark + "_" + repo}_summary.pkl')
        func_base = Utils.load_pickle(f'./cache/func_base/{benchmark + "_" + repo}.pkl')
        # 多线程encode：doc_list[idx][doc_vec] = encode(doc_list[idx])
        # summary_vec = encode(summary)
        def process_item(idx):
            func_item = func_base[idx]
            summary = summary_list[idx]
            for doc_item in func_item['doc_list']:
                doc_item['doc_vec'] = unixcoder_enc.encode_text(doc_item['doc'])
            if summary:
                summary_vec = unixcoder_enc.encode_text(summary)
            else:
                summary_vec = func_item['doc_list'][0]['doc_vec']
            func_item['summary_vec'] = summary_vec
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(process_item, idx) for idx in range(len(summary_list))]
            for future in tqdm(
                as_completed(futures),
                total=len(summary_list),
                colour="MAGENTA",
            ):
                future.result()
        Utils.dump_pickle(func_base, f'./cache/func_base/{benchmark + "_" + repo}_encoded.pkl')


