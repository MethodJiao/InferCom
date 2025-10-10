import json
import pickle

import torch
from transformers import AutoTokenizer, AutoModel
import tiktoken
import numpy as np
import platform
import os
from utils.unixcoder import UniXcoder
import torch.nn.functional as F

# repo_dir = 'repos/python/'
# repo_dir = 'repocoder_repos/'
def mk_dir(path): # path是指定文件夹路径
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)


class RCPromptBuilder:
    def __init__(self):
        self.seperator = '# ' + '-' * 50
        self.tokenizer = CodexTokenizer()
        
    def _make_a_block(self, retrieved_context):
        content = retrieved_context
        metadata = content['metadata']
        # put the file path in the comment
        assert metadata[0]['fpath_tuple'][0] == metadata[0]['repo']
        f_paths = ['/'.join(x['fpath_tuple'][1:]) for x in metadata]
        f_paths_str = '\n'.join([f'# {f_path}' for f_path in f_paths])
        f_path_comment = f'# the below code fragment can be found in:'
        # put code lines in the comment
        content_lines = content['context'].splitlines()
        content_lines_comment = [f'# {line}' for line in content_lines]
        # aggregate the comment and the code lines
        
        block_str = '\n'.join([f_path_comment, f_paths_str, self.seperator] + content_lines_comment + [self.seperator]) + '\n'
        tokenized_block = self.tokenizer.tokenize(block_str)
        token_len = len(tokenized_block)
        return block_str, token_len

    def build_rc_prompt(self, top_k_context, max_retrieval_length=2000):
        prepend_context = "# Here are some relevant code fragments from other files of the repo:\n"
        prepend_context += self.seperator + '\n'
        current_token_length = 20  # the length of the head_prompt, same for codex and codegen tokenizer
        prepend_blocks = []
        make_block_func = self._make_a_block
        for retrieved_context in top_k_context:
            block_str, token_len = make_block_func(retrieved_context)
            if current_token_length + token_len < max_retrieval_length:
                prepend_blocks.insert(0, block_str)
                current_token_length += token_len
            else:
                continue
        prepend_context += ''.join(prepend_blocks)  # all the blocks already have a line break at the end
        return prepend_context

class Similarity:
    @staticmethod
    def jaccard_similarity(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        if intersection == 0 or union == 0:
            return 0
        return float(intersection) / union

    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        # em1_cpu = embedding1.cpu()[0]
        # em2_cpu = embedding2.cpu()[0]
        # return np.dot(em1_cpu, em2_cpu) / (np.linalg.norm(em1_cpu) * np.linalg.norm(em2_cpu))
        return F.cosine_similarity(embedding1, embedding2).item()
    
    @staticmethod
    def cossim_tensors(query, database):
        # 计算余弦相似度
        # 使用torch.mm进行矩阵乘法，因为两个张量都是归一化的，所以这相当于计算余弦相似度
        cosine_similarities = torch.mm(query, database.t())

        # 如果需要，将结果移回CPU
        cosine_similarities_cpu = cosine_similarities.cpu().detach().numpy().tolist()[0]
        # 输出相似度
        return cosine_similarities_cpu


class Utils:
    @staticmethod
    def load_pickle(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def dump_pickle(obj, fname):
        with open(fname, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def read_code(fname):
        with open(fname, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def load_jsonl(fname):
        with open(fname, 'r', encoding='utf-8') as f:
            lines = []
            for line in f:
                lines.append(json.loads(line))
            return lines

    @staticmethod
    def dump_jsonl(obj, fname):
        with open(fname, 'w', encoding='utf8') as f:
            for item in obj:
                f.write(json.dumps(item) + '\n')




class UnixCoder:
    def __init__(self, device='7'):
        # model_name = '/home/dengle/unixcoder'
        model_name ="microsoft/unixcoder-base"
        if device != 'cpu':
            device = f'cuda:{device}'
        self.device = torch.device(device)
        self.model = UniXcoder(model_name)
        self.model.to(device)

    def encode_texts(self, texts):
        with torch.no_grad():
            tokens_ids = self.model.tokenize(texts,max_length=512,mode="<encoder-only>", padding=True)
            source_ids = torch.tensor(tokens_ids).to(self.device)
            _, func_embedding = self.model(source_ids)
            norm_embedding =  torch.nn.functional.normalize(func_embedding, p=2, dim=1)
        return norm_embedding

    def encode_text(self, text):
        with torch.no_grad():
            tokens_ids = self.model.tokenize([text],max_length=512,mode="<encoder-only>")
            source_ids = torch.tensor(tokens_ids).to(self.device)
            _, func_embedding = self.model(source_ids)
            norm_embedding =  torch.nn.functional.normalize(func_embedding, p=2, dim=1)
        return norm_embedding

class CodexTokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("p50k_base")

    def tokenize(self, text):
        # return self.tokenizer.encode(text)
        return self.tokenizer.encode_ordinary(text)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

# class DeepSeekCoderTokenizer:

class BlockGroupBuilder:
    def __init__(self, code, k=10):
        self.code = code
        self.sub_blocks = self.split_to_blocks1(code)
        self.k = k
        self.code_lines = code.splitlines()

    def split_to_blocks(self, code):
        lines = code.splitlines()

        res = []
        temp = []
        for line_no, line in enumerate(lines):
            if line != '':
                temp.append((line, line_no))
                continue
            if len(temp) != 0:
                res.append(temp)
            temp = []

        if len(temp) != 0:
            res.append(temp)
        return res

    def split_to_blocks1(self, code):
        codelines = code.splitlines()
        SPLIT_MARK = '@split mark@'
        pre_is_comment = False
        process_lines = []
        for line_no, line in enumerate(codelines):
            curr = line.strip()
            if curr == '':
                if len(process_lines) !=0 and process_lines[-1] != SPLIT_MARK:
                    process_lines.append(SPLIT_MARK)
            elif curr.startswith('#'):
                if pre_is_comment:
                    process_lines.append((line, line_no))
                else:
                    if len(process_lines) !=0 and process_lines[-1] != SPLIT_MARK:
                        process_lines.append(SPLIT_MARK)
                    process_lines.append((line, line_no))
                    pre_is_comment = True
            else:
                process_lines.append((line, line_no))
                pre_is_comment = False

        subchunk_list = []
        temp_subchunk = []
        for i in process_lines:
            if i != SPLIT_MARK:
                temp_subchunk.append(i)
            else:
                if len(temp_subchunk) != 0:
                    subchunk_list.append(temp_subchunk)
                temp_subchunk = []

        if len(temp_subchunk) != 0:
            subchunk_list.append(temp_subchunk)
        
        return subchunk_list

    def get_block_line(self, text):
        return len([i for i in text.splitlines() if i.strip()])

    def build_group_obj(self, group_lines):
        first_line_no = group_lines[0][1]
        last_line_no = group_lines[-1][1]
        return {
            'context': '\n'.join([i[0] for i in group_lines]),
            'first_no': first_line_no,
            'last_no': last_line_no
        }

    def get_block_groups(self):
        block_group_list = []
        for i in range(len(self.sub_blocks)):
            sub_block = self.sub_blocks[i]
            if len(sub_block) < self.k:
                temp_group_arr = sub_block
                j = i + 1
                while len(temp_group_arr) < self.k and j != len(self.sub_blocks):
                    sub_block = self.sub_blocks[j]
                    temp_group_arr.extend(sub_block)
                    j += 1

                block_group_list.append(self.build_group_obj(temp_group_arr[:self.k]))
            else:
                i = 0
                temp_group_arr = []
                while i < len(sub_block):
                    if len(temp_group_arr) == self.k:
                        block_group_list.append(self.build_group_obj(temp_group_arr))
                        temp_group_arr = []
                        i -= int(self.k / 2)
                    temp_group_arr.append(sub_block[i])
                    i += 1
                block_group_list.append(self.build_group_obj(temp_group_arr))
        return block_group_list


if __name__ == '__main__':
    # func_base = Utils.load_pickle('cache/func_base/agentscope-main_with_summary.pkl')
    unixcoder = UnixCoder()
    a = unixcoder.encode_text('def say_hello():')
    print(1)