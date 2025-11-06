import json
import os
import shutil
import sys

from tqdm import tqdm

from build_func_prompt import FuncPromptBuilder
from utils.summarize_code import summary_codes, encode_texts, SummaryModel
from utils.utils import RCPromptBuilder, Utils, UnixCoder, CodexTokenizer
from build_infile import build_infile
from build_py_func_base import FuncBaseBuilder as PyFuncBaseBuilder
from build_j_func_base import FuncBaseBuilder as JFuncBaseBuilder
from build_cpp_func_base import FuncBaseBuilder as CppFuncBaseBuilder
from read_undo_code import ReadUndoCode
import generate_api

def process_infile(infile_path, in_file, out_file, repo_dir, context_len=1000):
    build_infile(infile_path, in_file, out_file, infile_len=context_len, repo_dir=repo_dir)


def build_function_database(args):
    cache_dir = 'cache/func_base'
    os.makedirs(cache_dir, exist_ok=True)
    if args.lang == 'python':
        func_base_builder = PyFuncBaseBuilder(args.repos, args.repo_dir)
    elif args.lang == 'java':
        func_base_builder = JFuncBaseBuilder(args.repos, args.repo_dir)
    elif args.lang == 'cpp':
        func_base_builder = CppFuncBaseBuilder(args.repos, args.repo_dir)
    func_base_builder.build(benchmark=args.benchmark)
    summary_codes(args.repo_dir, lang=args.lang, benchmark=args.benchmark, summary_cuda=args.summary_cuda)
    encode_texts(args.repo_dir, benchmark=args.benchmark, encode_cuda=args.encode_cuda)


def build_func_prompt(args):
    examples = Utils.load_jsonl(args.rg_file)
    cache_dir = './cache/func_retrieval'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{args.benchmark}_{args.lang}.pkl')
    if not os.path.exists(cache_file):
        func_builder = FuncPromptBuilder(args.repo_dir, summary_cuda=args.summary_cuda, benchmark=args.benchmark, lang=args.lang, encode_cuda=args.encode_cuda)
        new_examples = func_builder.run(examples, use_doc=True, use_summary=True, k=args.k)
        Utils.dump_pickle(new_examples, cache_file)
    new_examples = Utils.load_pickle(cache_file)
    for example in new_examples:
        temp_example = example
        del temp_example['func_context']
    return new_examples

# newFunction
# 搜索相似代码片段
# 输出路径为test_res_rc.jsonl
def search_similar_code(args):
    examples = Utils.load_jsonl(args.infile_input)
    task_id =examples[0]['metadata']['task_id']
    input_code = examples[0]['prompt']
    func_builder = FuncPromptBuilder(args.repo_dir, summary_cuda=args.summary_cuda, benchmark=args.benchmark, lang=args.lang, encode_cuda=args.encode_cuda)
    new_examples = func_builder.get_similar_code(input_code, task_id, k=args.k)
    Utils.dump_jsonl(new_examples, 'test_res_rc.jsonl')

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
        o_prompt = i['prompt'] #['old_prompt']!!!!!!!!!
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

# 第一步只需要进行代码仓库的构建，只需要读repo文件夹下的文件
# 构建仓库时，可能出现编码格式的问题，因此需要用到decode_tool.py中的handle函数
def main_build_base():
    #main1()
    parser = argparse.ArgumentParser()
    # 添加参数 
    parser.add_argument('--summary_cuda', type=int, default=-1)
    parser.add_argument('--lang', type=str, default='cpp', choices=['python','cpp'])
    parser.add_argument('--repo_dir', type=str, default='repos/sota_test/C++Examples', choices=['repos/sota_test/pythonExamples', 'repos/sota_test/C++Examples'])
    parser.add_argument('--encode_cuda', type=str, default='0')
    parser.add_argument('--benchmark', type=str, default='sota_test', choices=['projbench', 'cceval', 'sota_test', 'repoeval_api'])
    parser.add_argument('--rg_file', type=str, default='datasets/projbench/pybenchmark_2k.jsonl',help='需要用到第一次检索的相似代码')
    parser.add_argument('--rc_file', type=str, default='test_res_rc.jsonl',help='需要用到第二次检索的相似代码') # datasets/projbench/rc_template.jsonl
    parser.add_argument('--api_output', default="apioutput/temp_out.jsonl",type=str)
    parser.add_argument('--process', type=str, default='build_database',choices=['build_infile', 'build_database', 'infer_api', 'build_prompt'])
    parser.add_argument('--infile_len', type=int, default=2048)
    parser.add_argument('--infile_input', type=str,default='datasets/projbench/pybenchmark_own.jsonl')
    parser.add_argument('--infile_output', type=str,default='datasets/projbench/pybenchmark_2k.jsonl')
    parser.add_argument('--k', type=int, default=4, help='推理的api数量')
    parser.add_argument('--fsr', type=int, default=1, choices=[0, 1])
    parser.add_argument('--uer', type=int, default=1, choices=[0, 1])
    parser.add_argument('--prompt_output',default='prompts/sota_test/pybenchmark_4k.jsonl',type=str)
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

    build_function_database(args)




def main_generate_code():

    #清除缓存和中间文件
    #删除prompts和predictions文件夹下的目录
    predictions_path = 'predictions/sota_test'
    prompts_path = 'prompts/sota_test'
    if not os.path.exists(prompts_path):
        os.makedirs(prompts_path)
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)
    shutil.rmtree(predictions_path)
    shutil.rmtree(prompts_path)
    if not os.path.exists(prompts_path):
        os.makedirs(prompts_path)
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)

    #删除main3中所创建的pkl文件
    file_path = 'cache/func_retrieval'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    shutil.rmtree(file_path)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    parser = argparse.ArgumentParser()
    # 添加参数 
    parser.add_argument('--summary_cuda', type=int, default=-1)
    parser.add_argument('--lang', type=str, default='cpp', choices=['python','cpp'])
    parser.add_argument('--repo_dir', type=str, default='repos/sota_test/C++Examples', choices=['repos/sota_test/pythonExamples', 'repos/sota_test/C++Examples'])
    parser.add_argument('--encode_cuda', type=str, default='0')
    parser.add_argument('--benchmark', type=str, default='sota_test', choices=['projbench', 'cceval', 'sota_test', 'repoeval_api'])
    parser.add_argument('--rg_file', type=str, default='predictions/sota_test/pybenchmark_2k.jsonl',help='需要用到第一次检索的相似代码') # datasets/projbench/pybenchmark_2k.jsonl
    parser.add_argument('--rc_file', type=str, default='test_res_rc.jsonl',help='需要用到第二次检索的相似代码') # datasets/projbench/rc_template.jsonl
    parser.add_argument('--api_output', default="apioutput/temp_out.jsonl",type=str)
    parser.add_argument('--process', type=str, default='build_infile',choices=['build_infile', 'build_database', 'infer_api', 'build_prompt'])
    parser.add_argument('--infile_len', type=int, default=2048)
    parser.add_argument('--infile_input', type=str,default='datasets/projbench/pybenchmark_test.jsonl')
    parser.add_argument('--infile_output', type=str,default='datasets/projbench/pybenchmark_2k.jsonl')
    parser.add_argument('--k', type=int, default=4, help='推理的api数量')
    parser.add_argument('--fsr', type=int, default=1, choices=[0, 1])
    parser.add_argument('--uer', type=int, default=1, choices=[0, 1])
    parser.add_argument('--prompt_output',default='prompts/sota_test/pybenchmark_4k.jsonl',type=str)
    parser.add_argument('--infile_path',default='',type=str, help='输入的待完善的代码文件路径')
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

    #设置输入的文件路径和输出的jsonl文件路径
    # 输出的jsonl文件路径为args.infile_input
    input_filepath = 'TestCode.cpp'
    output_jsonlpath = 'datasets/projbench/pybenchmark_test.jsonl'
    if args.infile_path == '':
        args.infile_path = input_filepath
    repos = args.repo_dir
    ReadUndoCode.read_undo_code(repos, input_filepath, output_jsonlpath)

    #main2
    process_infile(args.infile_path, args.infile_input, args.infile_output, context_len=args.infile_len, repo_dir=args.repo_dir)
    args.process = 'search_similar_code'
    #进行完一个步骤之后，需要相应的改变args.process，方便后续调试


    #main_similar_code
    #生成的jsonl文件的位置为test_res_rc.jsonl，在search_similar_code方法中写死了
    search_similar_code(args)
    args.process = 'infer_api'

    #生成初始的预测
    generate_api.generate_code('pybenchmark_2k.jsonl')
    #main3
    build_func_prompt(args)
    args.process = 'build_prompt'

    #构建最终提示词并最终生成预测结果
    #main4
    combine_rc_and_api(args)
    generate_api.generate_code('pybenchmark_4k.jsonl')


if __name__ == '__main__':
    # 提示用户选择要执行的主流程：1 执行构建基础，2 执行生成代码
    try:
        print("请选择要执行的操作：\n1) 构建代码仓库基础 (main_build_base)\n2) 生成推理代码 (main_generate_code)")
        choice = input('输入 1 或 2: ').strip()
    except (KeyboardInterrupt, EOFError):
        print('\n未输入，程序退出')
        sys.exit(1)

    if choice == '1':
        main_build_base()
    elif choice == '2':
        main_generate_code()
    else:
        print('输入错误：请输入 1 或 2')
        sys.exit(1)