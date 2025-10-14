
import json
import os
import random
from utils.utils import Utils, RCPromptBuilder

def generate_rc_file(api_examples_file, rc_file_output, repo_dir, k=5):
    """
    生成rc_file文件，包含与api_examples中每个任务对应的相似代码上下文

    参数:
        api_examples_file: api_examples的pickle文件路径
        rc_file_output: 输出的rc_file JSONL文件路径
        repo_dir: 仓库目录
        k: 每个任务要检索的相似代码片段数量
    """
    # 加载api_examples
    api_examples = Utils.load_pickle(api_examples_file)

    # 创建模拟的相似代码上下文
    rc_entries = []

    # 遍历所有api_examples，为每个任务生成相似代码上下文
    for api_example in api_examples:
        task_id = api_example['metadata']['task_id']
        print(f"处理任务: {task_id}")

        # 获取任务相关的文件路径信息
        fpath_tuple = api_example['metadata']['fpath_tuple']
        repo_name = fpath_tuple[0]
        file_path = '/'.join(fpath_tuple[1:])

        # 模拟生成top_k_context
        top_k_context = []

        # 在仓库中查找相似文件
        repo_path = os.path.join(repo_dir, repo_name)
        if os.path.exists(repo_path):
            # 获取仓库中的所有Python文件
            python_files = []
            for root, dirs, files in os.walk(repo_path):
                for file in files:
                    if file.endswith('.py'):
                        rel_path = os.path.relpath(os.path.join(root, file), repo_path)
                        python_files.append(rel_path)

            # 随机选择k个不同的文件（排除原文件）
            other_files = [f for f in python_files if f != file_path]
            selected_files = random.sample(other_files, min(k, len(other_files)))

            # 为每个选中的文件生成代码片段
            for selected_file in selected_files:
                file_full_path = os.path.join(repo_path, selected_file)

                # 读取文件内容
                with open(file_full_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # 随机选择一段代码（至少5行，最多20行）
                start_line = random.randint(0, max(0, len(lines) - 20))
                end_line = min(start_line + random.randint(5, 20), len(lines))
                code_lines = lines[start_line:end_line]
                code_context = ''.join(code_lines)

                # 构建metadata
                metadata = [{
                    'fpath_tuple': [repo_name] + selected_file.split('/'),
                    'repo': repo_name,
                    'file': selected_file,
                    'start_line': start_line + 1,
                    'end_line': end_line
                }]

                # 创建代码片段
                context_entry = {
                    'metadata': metadata,
                    'context': code_context,
                    'file': selected_file,
                    'repo': repo_name
                }

                top_k_context.append(context_entry)

        # 创建rc_entry
        rc_entry = {
            'metadata': {
                'task_id': task_id,
                'top_k_context': top_k_context
            },
            'task_id': task_id,
            'top_k_context': top_k_context
        }

        rc_entries.append(rc_entry)

    # 将结果保存为JSONL文件
    with open(rc_file_output, 'w', encoding='utf-8') as f:
        for entry in rc_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"rc_file已生成: {rc_file_output}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--api_examples_file', type=str, required=True, help='api_examples的pickle文件路径')
    parser.add_argument('--rc_file_output', type=str, required=True, help='输出的rc_file JSONL文件路径')
    parser.add_argument('--repo_dir', type=str, required=True, help='仓库目录')
    parser.add_argument('--k', type=int, default=5, help='每个任务要检索的相似代码片段数量')

    args = parser.parse_args()

    generate_rc_file(args.api_examples_file, args.rc_file_output, args.repo_dir, args.k)
