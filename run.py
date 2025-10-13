import json
import os

from tqdm import tqdm

from build_func_prompt import FuncPromptBuilder
from utils.summarize_code import summary_codes, encode_texts, SummaryModel
from utils.utils import RCPromptBuilder, Utils, UnixCoder, CodexTokenizer
from build_infile import build_infile
from build_py_func_base import FuncBaseBuilder as PyFuncBaseBuilder
from build_j_func_base import FuncBaseBuilder as JFuncBaseBuilder


def process_infile(in_file, out_file, repo_dir, context_len=1000):
    build_infile(in_file, out_file, infile_len=context_len, repo_dir=repo_dir)


def build_function_database(args):
    cache_dir = 'cache/func_base'
    os.makedirs(cache_dir, exist_ok=True)
    if args.lang == 'python':
        func_base_builder = PyFuncBaseBuilder(args.repos, args.repo_dir)
    elif args.lang == 'java':
        func_base_builder = JFuncBaseBuilder(args.repos, args.repo_dir)
    func_base_builder.build(benchmark=args.benchmark)
    summary_codes(args.repos, lang=args.lang, benchmark=args.benchmark, summary_cuda=args.summary_cuda)
    encode_texts(args.repos, benchmark=args.benchmark, encode_cuda=args.encode_cuda)


def build_func_prompt(args):
    examples = Utils.load_jsonl(args.rg_file)
    cache_dir = './cache/func_retrieval'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{args.benchmark}_{args.lang}.pkl')
    if not os.path.exists(cache_file):
        func_builder = FuncPromptBuilder(args.repos, summary_cuda=args.summary_cuda, benchmark=args.benchmark, lang=args.lang, encode_cuda=args.encode_cuda)
        new_examples = func_builder.run(examples, use_doc=True, use_summary=True, k=args.k)
        Utils.dump_pickle(new_examples, cache_file)
    new_examples = Utils.load_pickle(cache_file)
    for example in new_examples:
        temp_example = example
        del temp_example['func_context']
    return new_examples



def combine_rc_and_api(args):
    flag = True
    PROMPT_LEN = 4096
    tokenizer = CodexTokenizer()
    rc_prompt_builder = RCPromptBuilder()
    api_examples = Utils.load_pickle(os.path.join('./cache/func_retrieval', f'{args.benchmark}_{args.lang}.pkl'))
    rc_examples = Utils.load_jsonl(args.rc_file)
    rc_record = {i['metadata']['task_id']:i for i in rc_examples}
    for i in api_examples:
        task_id = i['metadata']['task_id']
        rc_example = rc_record[task_id]
        o_prompt = i['old_prompt']
        api_prompt = ''
        # if args.uer:
        uer_api_prompt = i['func_detail']['doc'] + '\n'
        # if args.fsr:
        fsr_api_prompt = i['func_detail']['summary'] + '\n'

        rest_prompt_len = min(PROMPT_LEN - len(tokenizer.tokenize(uer_api_prompt + fsr_api_prompt + '\n' + o_prompt)), 2048)
        
        if args.uer:
            api_prompt += uer_api_prompt
        if args.fsr:
            api_prompt += fsr_api_prompt
        
        similar_prompt = rc_prompt_builder.build_rc_prompt(rc_example['metadata']['top_k_context'], rest_prompt_len)
        new_prompt = similar_prompt + '\n' + f'{"# Here are some APIs that may be used from other files" if flag else ""}' + '\n' + api_prompt + '\n' + o_prompt
        i['prompt'] = new_prompt
        del i['func_context']
    Utils.dump_jsonl(api_examples, args.prompt_output)
        
import argparse

def main1():
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument('--summary_cuda', type=int, default=-1)
    parser.add_argument('--lang', type=str, default='python', choices=['python', 'java'])
    parser.add_argument('--repo_dir', type=str, default='repos/sota_test', choices=['repos/python', 'repos/java', 'repos/sota_test', 'repos/repoeval_api', '/data/dengle/repofuse/crosscodeeval_rawdata'])
    parser.add_argument('--encode_cuda', type=str, default='0')
    parser.add_argument('--benchmark', type=str, default='sota_test', choices=['projbench', 'cceval', 'sota_test', 'repoeval_api'])
    parser.add_argument('--rg_file', type=str, help='需要用到第一次检索推理的结果')
    parser.add_argument('--rc_file', type=str, help='需要用到第二次检索的相似代码')
    parser.add_argument('--api_output', type=str)
    parser.add_argument('--process', type=str, default='build_database',choices=['build_infile', 'build_database', 'infer_api', 'build_prompt'])
    parser.add_argument('--infile_len', type=int, default=2048)
    parser.add_argument('--infile_input', type=str)
    parser.add_argument('--infile_output', type=str)
    parser.add_argument('--k', type=int, default=4, help='推理的api数量')
    parser.add_argument('--fsr', type=int, default=1, choices=[0, 1])
    parser.add_argument('--uer', type=int, default=1, choices=[0, 1])
    parser.add_argument('--prompt_output', type=str)
    # 解析参数
    args = parser.parse_args()
    # 设置repos
    if args.benchmark == 'cceval':
        repos = json.loads(open('cceval/data.json', 'r', encoding='utf-8').read())[args.lang]
    else:
        entries = os.listdir(args.repo_dir)
        repos = [entry for entry in entries if os.path.isdir(os.path.join(args.repo_dir, entry))]
    setattr(args, 'repos', repos)
    print(args)
    
    '''cceval
    data: /data1/dengle/python/cc_line_completion.jsonl
    repos: /data1/dengle/crosscodeeval_rawdata/
    '''
    if args.process == 'build_infile':
        process_infile(args.infile_input, args.infile_output, context_len=args.infile_len, repo_dir=args.repo_dir)#应该是第二步构建草稿
    elif args.process == 'build_database':
        # build api database
        build_function_database(args)#第一步，已经生成pkl
    elif args.process == 'infer_api':
        # search api info
        build_func_prompt(args)#应该是第三步检索相关api信息
        # Utils.dump_jsonl(res_examples, args.api_output)
    elif args.process == 'build_prompt':
        combine_rc_and_api(args)#应该是最后一步，把相关api信息和代码草稿一起移送llm推理
    
def main2():
    parser = argparse.ArgumentParser()
    # 添加参数 
    parser.add_argument('--summary_cuda', type=int, default=-1)
    parser.add_argument('--lang', type=str, default='python', choices=['python', 'java'])
    parser.add_argument('--repo_dir', type=str, default='repos/sota_test', choices=['repos/python', 'repos/java', 'repos/sota_test', 'repos/repoeval_api', '/data/dengle/repofuse/crosscodeeval_rawdata'])
    parser.add_argument('--encode_cuda', type=str, default='0')
    parser.add_argument('--benchmark', type=str, default='sota_test', choices=['projbench', 'cceval', 'sota_test', 'repoeval_api'])
    parser.add_argument('--rg_file', type=str, help='需要用到第一次检索推理的结果')
    parser.add_argument('--rc_file', type=str, help='需要用到第二次检索的相似代码')
    parser.add_argument('--api_output', type=str)
    parser.add_argument('--process', type=str, default='build_infile',choices=['build_infile', 'build_database', 'infer_api', 'build_prompt'])
    parser.add_argument('--infile_len', type=int, default=2048)
    parser.add_argument('--infile_input', type=str,default='datasets/projbench/pybenchmark_own.jsonl')
    parser.add_argument('--infile_output', type=str,default='datasets/projbench/pybenchmark_2k.jsonl')
    parser.add_argument('--k', type=int, default=4, help='推理的api数量')
    parser.add_argument('--fsr', type=int, default=1, choices=[0, 1])
    parser.add_argument('--uer', type=int, default=1, choices=[0, 1])
    parser.add_argument('--prompt_output', type=str)
    # 解析参数
    args = parser.parse_args()
    # 设置repos
    if args.benchmark == 'cceval':
        repos = json.loads(open('cceval/data.json', 'r', encoding='utf-8').read())[args.lang]
    else:
        entries = os.listdir(args.repo_dir)
        repos = [entry for entry in entries if os.path.isdir(os.path.join(args.repo_dir, entry))]
    setattr(args, 'repos', repos)
    print(args)
    
    '''cceval
    data: /data1/dengle/python/cc_line_completion.jsonl
    repos: /data1/dengle/crosscodeeval_rawdata/
    '''  
    if args.process == 'build_infile':
        process_infile(args.infile_input, args.infile_output, context_len=args.infile_len, repo_dir=args.repo_dir)#当前 应该是第二步构建APIbase，已经生成
    elif args.process == 'build_database':
        # build api database
        build_function_database(args)#第一步，已经生成pkl
    elif args.process == 'infer_api':
        # search api info
        build_func_prompt(args)#应该是第三步检索相关api信息
        # Utils.dump_jsonl(res_examples, args.api_output)
    elif args.process == 'build_prompt':
        combine_rc_and_api(args)#应该是最后一步，把相关api信息和代码草稿一起移送llm推理
    
def main3():
    parser = argparse.ArgumentParser()
    # 添加参数 
    parser.add_argument('--summary_cuda', type=int, default=-1)
    parser.add_argument('--lang', type=str, default='python', choices=['python', 'java'])
    parser.add_argument('--repo_dir', type=str, default='repos/sota_test', choices=['repos/python', 'repos/java', 'repos/sota_test', 'repos/repoeval_api', '/data/dengle/repofuse/crosscodeeval_rawdata'])
    parser.add_argument('--encode_cuda', type=str, default='0')
    parser.add_argument('--benchmark', type=str, default='sota_test', choices=['projbench', 'cceval', 'sota_test', 'repoeval_api'])
    parser.add_argument('--rg_file', type=str, default='predictions/sota_test/pybenchmark_2k.jsonl',help='需要用到第一次检索的相似代码')
    parser.add_argument('--rc_file', type=str, help='需要用到第二次检索的相似代码')
    parser.add_argument('--api_output', type=str)
    parser.add_argument('--process', type=str, default='infer_api',choices=['build_infile', 'build_database', 'infer_api', 'build_prompt'])
    parser.add_argument('--infile_len', type=int, default=2048)
    parser.add_argument('--infile_input', type=str,default='datasets/projbench/pybenchmark_own.jsonl')
    parser.add_argument('--infile_output', type=str,default='datasets/projbench/pybenchmark_2k.jsonl')
    parser.add_argument('--k', type=int, default=4, help='推理的api数量')
    parser.add_argument('--fsr', type=int, default=1, choices=[0, 1])
    parser.add_argument('--uer', type=int, default=1, choices=[0, 1])
    parser.add_argument('--prompt_output', type=str)
    # 解析参数
    args = parser.parse_args()
    # 设置repos
    if args.benchmark == 'cceval':
        repos = json.loads(open('cceval/data.json', 'r', encoding='utf-8').read())[args.lang]
    else:
        entries = os.listdir(args.repo_dir)
        repos = [entry for entry in entries if os.path.isdir(os.path.join(args.repo_dir, entry))]
    setattr(args, 'repos', repos)
    print(args)
    
    '''cceval
    data: /data1/dengle/python/cc_line_completion.jsonl
    repos: /data1/dengle/crosscodeeval_rawdata/
    '''  
    if args.process == 'build_infile':
        process_infile(args.infile_input, args.infile_output, context_len=args.infile_len, repo_dir=args.repo_dir)#应该是第二步构建草稿，已经生成草稿prompt
    elif args.process == 'build_database':
        # build api database
        build_function_database(args)#第一步，已经生成pkl
    elif args.process == 'infer_api':
        # search api info
        build_func_prompt(args)#  ！当前完成！ 应该是第三步检索相关api信息  在这之前先得执行一下generate_api，把predictions/sota_test/pybenchmark_2k.jsonl传入才行 输出产物cache/func_retrieval/sota~~.pkl
        # Utils.dump_jsonl(res_examples, args.api_output)
    elif args.process == 'build_prompt':
        combine_rc_and_api(args)#应该是第四步，把相关api信息和代码草稿一起移送llm推理
        
    #在这之后 第五步最终要执行generate_api.py这个是最终成果了，我暂时伪造了prompt文件夹下的文件直接跳过第四步直接使用generate_api可以运行

def main4():
    parser = argparse.ArgumentParser()
    # 添加参数 
    parser.add_argument('--summary_cuda', type=int, default=-1)
    parser.add_argument('--lang', type=str, default='python', choices=['python', 'java'])
    parser.add_argument('--repo_dir', type=str, default='repos/sota_test', choices=['repos/python', 'repos/java', 'repos/sota_test', 'repos/repoeval_api', '/data/dengle/repofuse/crosscodeeval_rawdata'])
    parser.add_argument('--encode_cuda', type=str, default='0')
    parser.add_argument('--benchmark', type=str, default='sota_test', choices=['projbench', 'cceval', 'sota_test', 'repoeval_api'])
    parser.add_argument('--rg_file', type=str, default='datasets/projbench/pybenchmark_2k.jsonl',help='需要用到第一次检索的相似代码')
    parser.add_argument('--rc_file', type=str, help='需要用到第二次检索的相似代码')
    parser.add_argument('--api_output', type=str)
    parser.add_argument('--process', type=str, default='infer_api',choices=['build_infile', 'build_database', 'infer_api', 'build_prompt'])
    parser.add_argument('--infile_len', type=int, default=2048)
    parser.add_argument('--infile_input', type=str,default='datasets/projbench/pybenchmark_own.jsonl')
    parser.add_argument('--infile_output', type=str,default='datasets/projbench/pybenchmark_2k.jsonl')
    parser.add_argument('--k', type=int, default=4, help='推理的api数量')
    parser.add_argument('--fsr', type=int, default=1, choices=[0, 1])
    parser.add_argument('--uer', type=int, default=1, choices=[0, 1])
    parser.add_argument('--prompt_output', type=str)
    # 解析参数
    args = parser.parse_args()
    # 设置repos
    if args.benchmark == 'cceval':
        repos = json.loads(open('cceval/data.json', 'r', encoding='utf-8').read())[args.lang]
    else:
        entries = os.listdir(args.repo_dir)
        repos = [entry for entry in entries if os.path.isdir(os.path.join(args.repo_dir, entry))]
    setattr(args, 'repos', repos)
    print(args)
    
    '''cceval
    data: /data1/dengle/python/cc_line_completion.jsonl
    repos: /data1/dengle/crosscodeeval_rawdata/
    '''  
    if args.process == 'build_infile':
        process_infile(args.infile_input, args.infile_output, context_len=args.infile_len, repo_dir=args.repo_dir)#是第二步构建草稿，已经生成草稿prompt
    elif args.process == 'build_database':
        # build api database
        build_function_database(args)#第一步，已经生成pkl
    elif args.process == 'infer_api':
        # search api info
        build_func_prompt(args)#是第三步检索相关api信息    生成了prompt,这个环节处理了联想代码相似度
        # Utils.dump_jsonl(res_examples, args.api_output)
    elif args.process == 'build_prompt':
        combine_rc_and_api(args)#是第四步，把相关api信息和代码草稿推理进一步完善prompt（缺失）
        
    #在这之后 第五步最终要在执行一次generate_api.py 进行LLM输出最终成果了，目前暂时伪造了prompt文件夹下的文件直接跳过第四步直接使用generate_api可以运行
if __name__ == '__main__':
    main3()