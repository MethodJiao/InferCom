from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq

import numpy as np
import torch
from tqdm import tqdm
import os
from utils.utils import UnixCoder, Utils, CodexTokenizer, Similarity
from utils.summarize_code import SummaryAPIModel, SummaryModel, postprocess_llama
from concurrent.futures import ThreadPoolExecutor, as_completed

class FuncPromptBuilder:
    def __init__(self, repos, benchmark, lang, summary_cuda=3, encode_cuda='cpu'):
        self.encoder = UnixCoder(encode_cuda)
        self.tokenizer = CodexTokenizer()
        self.repos = repos
        self.func_database_dict = dict()
        self.uer_base = dict()
        self.fsr_base = dict()
        self.lang = lang
        if summary_cuda == -1:
            self.summary_model = SummaryAPIModel()
        else:
            self.summary_model = SummaryModel(cuda=summary_cuda)
        for repo in tqdm(repos, desc='loading database...'):
            # print(f"loading database: {repo}")
            cache_path = f'./cache/func_base/{benchmark}_{repo}_encoded.pkl' if benchmark is not None else f'./cache/func_base/{repo}_encoded.pkl'
            repo_base = Utils.load_pickle(cache_path)
            idx_mapping, uer_matrix, fsr_matrix = self.build_database(repo_base)
            self.func_database_dict[repo] = {
                'data': repo_base,
                'idx_mapping': idx_mapping,
                'uer_matrix': uer_matrix,
                'fsr_matrix': fsr_matrix
            }
            

    def build_database(self, repo_base):
        # 建立全向量 以及 新index到原数据库index的映射
        idx_mapping = []
        uer_tensors = []
        fsr_tensors = []
        for o_idx, item in enumerate(repo_base):
            fsr_tensors.append(item['summary_vec'])
            ue_set = set()
            for ue in item['doc_list']:
                if ue['doc'] in ue_set:
                    continue
                else:
                    ue_set.add(ue['doc'])
                    uer_tensors.append(ue['doc_vec'])
                    idx_mapping.append(o_idx)
        uer_result = torch.cat(uer_tensors, dim=0)
        fsr_result = torch.cat(fsr_tensors, dim=0)
        return idx_mapping, uer_result, fsr_result

    @staticmethod
    def get_last_line(example):
        # task_type = example['metadata']['task_type']
        # prompt = example['prompt'].splitlines()
        pred_res = [line for line in example['pred_res'].splitlines() if line.strip()]
        code = example['prompt'] + pred_res[0] if len(pred_res) != 0 else example['prompt']
        return code.splitlines()[-1]


    @staticmethod
    def get_last_lines(example):
        # task_type = example['metadata']['task_type']
        pred_res = [line for line in example['pred_res'].splitlines() if line.strip()]
        code = example['prompt'] + pred_res[0] if len(pred_res) != 0 else example['prompt']
        return '\n'.join([code.splitlines()[-1]] + pred_res[1:])


    def get_topk_func(self, in_doc, database, k=3, example=None):
        in_embedding = self.encoder.encode_text(in_doc)
        scores = []
        input_fpath = tuple(example['metadata']['fpath_tuple'])
        input_lineno = example['metadata']['context_start_lineno']
        
        matrix = database['uer_matrix']
        idx_mapping = database['idx_mapping']
        scores = Similarity.cossim_tensors(in_embedding, matrix)

        idxs = heapq.nlargest(k, range(len(scores)), scores.__getitem__)
        real_idxs = list(dict.fromkeys([idx_mapping[i] for i in idxs]))
        # idx = scores.index(max(scores))
        res = [database['data'][i] for i in real_idxs]
        return res

    def get_topk_func_by_summary(self, in_doc, database, k=3):
        # summary = summary_one_code_use_llm(in_doc)
        lang = self.lang
        prompt_template = open(f'./utils/{lang}.prompt').read()
        summary = self.summary_model.summarize_code(prompt_template.replace('@{}@', in_doc)).strip()
        summary = postprocess_llama(summary)
        in_embedding = self.encoder.encode_text(summary)
        
        result = database['fsr_matrix']
        scores = Similarity.cossim_tensors(in_embedding, result)
                
        idxs = heapq.nlargest(k, range(len(scores)), scores.__getitem__)
        # idx = scores.index(max(scores))
        res = [database['data'][i] for i in idxs]
        return res, summary

    def build_new_prompt(self, func_list):
        # new_prompt = 'Here are some APIs that may be used from other files\n'
        new_prompt = ''
        for func in func_list:
            func_path = '/'.join(func['fpath'])
            func_prompt = func_path + '\n' + func['info']
            func_prompt = '\n'.join(['# ' + i for i in func_prompt.splitlines()])
            func_prompt = func_prompt + '\n\n'
            new_prompt += func_prompt
        return new_prompt

    def process_example(self, example, use_doc=True, use_summary=True, k=4):
        repo = example['metadata']['task_id'].split('/')[0]
        if repo not in self.func_database_dict:
            return None
        ret_funcs = []
        ret_funcs0 = []
        ret_funcs1 = []
        summary_query = ''
        last_line = self.get_last_line(example).strip()
        if last_line:  
            if use_doc:
                ret_funcs0 = self.get_topk_func(last_line, self.func_database_dict[repo], k=k*4, example=example)
                ret_funcs.extend(ret_funcs0[:k])
            if use_summary:
                last_lines = self.get_last_lines(example).strip()
                ret_funcs1, summary_query = self.get_topk_func_by_summary(last_lines, self.func_database_dict[repo], k=k*4)
                ret_funcs.extend(ret_funcs1[:k])
        new_prompt_prefix = self.build_new_prompt(ret_funcs)
        doc_prompt = self.build_new_prompt(ret_funcs0[:k])
        summary_prompt = self.build_new_prompt(ret_funcs1[:k])
        new_prompt = new_prompt_prefix
        # new_prompt = new_prompt_prefix + example['prompt']
        example['func_context'] = {'uer': ret_funcs0, 'fsr': ret_funcs1}
        # example['rg_prompt'] = example['prompt']
        # example['prompt'] = new_prompt + '\n' + example['prompt']
        example['func_prompt'] = new_prompt
        example['func_detail'] = {'doc': doc_prompt, 'summary': summary_prompt}
        example['summary_query'] = summary_query
        return example
 
    def run(self, examples, use_doc=True, use_summary=True, k=4):
        new_examples = []
        for example in tqdm(examples):
            result = self.process_example(example, use_doc=use_doc, use_summary=use_summary, k=k)
            if result:
                new_examples.append(result)
        return new_examples


