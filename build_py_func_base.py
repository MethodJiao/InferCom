import glob
from pathlib import Path
import re
from tqdm import tqdm
from tree_sitter_languages import get_language, get_parser
import os

from utils.utils import Utils, UnixCoder


def process_class(node):
    name = ''
    arg_list = ''
    for child in node.children:
        if child.type == 'argument_list':
            arg_list = child.text.decode()
        elif child.type == 'identifier':
            name = child.text.decode()
    class_tag = f'class {name}{arg_list}'
    return {'name': name, 'arg_list': arg_list, 'sign': class_tag}

def process_params(params_node):
    id_list = []
    for child in params_node.children:
        if child.type == 'identifier':
            if child.text.decode().strip() not in ['self', 'cls'] :
                id_list.append(child.text.decode())
        else:
            for param_child in child.children:
                if param_child.type == 'identifier':
                    id_list.append(param_child.text.decode())
    params_str = ''

    if len(id_list) >= 1:
        params_str = id_list[0]

        for identifier in id_list[1:]:
            params_str += f', {identifier}'
    
    return f'({params_str})'


def process_func(node):
    name = ''
    params = ''
    params_wo_type = ''
    return_type = ''
    for child in node.children:
        if child.type == 'parameters':
            params = child.text.decode()
            params_wo_type = process_params(child)
        elif child.type == 'identifier':
            name = child.text.decode()
        elif child.type == 'type':
            return_type = child.text.decode()
    sign = f'def {name}{params}{" -> " + return_type if return_type != "" else ""}'
    return {'name': name, 'params': params, 'params_wo_type': params_wo_type, 'return_type': return_type, 'sign': sign}

# def camel_to_snake(camel_str):
#     return re.sub(r'(?<!^)(?=[A-Z])', '_', camel_str).lower()

def camel_to_snake(name):
    # 查找每个大写字母，并在其前面加上下划线，但是不包括第一个字符前和连续大写字母之间
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    # 处理连续的大写字母（例如：URLConfig -> URL_Config）
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()

def extract_decorators(node):
    res = []
    if node:
        for child in node.children:
            if child.type == 'decorator':
                res.append(child.text.decode().strip())
    return res

def extract_func_name(node):
    for child in node.named_children:
        if child.type == 'identifier':
            return child.text.decode()
    return None


class FuncBaseBuilder:
    def __init__(self, repos, repo_dir, encode_cuda='cpu'):
        self.repos = repos
        self.language = get_language('python')
        self.parser = get_parser('python')
        # self.encoder = UnixCoder(encode_cuda)
        self.repo_dir = repo_dir

    def build(self, benchmark=None):
        for repo in self.repos:
            out_path = f'./cache/func_base/{benchmark}_{repo}.pkl'
            if os.path.exists(out_path):
                print(f"{repo}: cache")
                continue
            func_list, class_list = self.get_func_list(repo_name=repo)
            func_database = []
            # for class_dict in class_list:
            #     print(class_dict)
            for example in tqdm(func_list, desc=f'processing {repo}'):
                func_def = example['func_def']
                class_def = example['class_def']
                example['func'] = process_func(func_def)
                doc_list = []
                if class_def:
                    example['class'] = process_class(class_def)
                    if example['func']['name'] == '__init__':
                        # doc = example['class']['name'] + example['func']['params_wo_type']
                        for i_idx, i in enumerate([example['class']['name'], f"{camel_to_snake(example['class']['name'])} = {example['class']['name']}"]):
                            for j_idx, j in enumerate([example['func']['params_wo_type'], '()']):
                                doc_list.append({
                                    'doc': i + j,
                                    'doc_type': ('init', i_idx, j_idx)
                                })
                    else:
                        # 静态方法或类方法
                        if any([kw in example['decorators'] for kw in ['@staticmethod', '@classmethod']]):
                            for i_idx, i in enumerate([f"{camel_to_snake(example['class']['name'])}.{example['func']['name']}", f"{example['class']['name']}.{example['func']['name']}"]):
                                for j_idx, j in enumerate([example['func']['params_wo_type'], '()']):
                                    doc_list.append({
                                        'doc': i + j,
                                        'doc_type': ('decorated', i_idx, j_idx)
                                    })
                        # doc = camel_to_snake(example['class']['name']) + "." + example['func']['name'] + example['func']['params_wo_type']
                        else:
                            for i_idx, i in enumerate([f"{camel_to_snake(example['class']['name'])}.{example['func']['name']}"]):
                                for j_idx, j in enumerate([example['func']['params_wo_type'], '()']):
                                    doc_list.append({
                                        'doc': i + j,
                                        'doc_type': ('common_class', i_idx, j_idx)
                                    })
                    info = example['class']['sign'] + ':\n    ' + '\n    '.join(example['decorators']) + '\n    ' + example['func']['sign']
                else:
                    example['class'] = None
                    for i_idx, i in enumerate([f"{example['func']['name']}", f"{Path(example['file_path']).stem}.{example['func']['name']}"]):
                        for j_idx, j in enumerate([example['func']['params_wo_type'], '()']):
                            doc_list.append({
                                'doc': i + j,
                                'doc_type': ('common', i_idx, j_idx)
                            })
                    # doc = example['func']['name'] + example['func']['params_wo_type']
                    info = example['func']['sign']
                
                # embedding = self.encoder.encode_text(doc)
                fpath = tuple([i for i in example['file_path'].replace(self.repo_dir, '').split('/') if i.strip()])
                func_body = func_def.text.decode()
                metadata = {
                    'func': example['func'],
                    'func_body': func_body,
                    'class': example['class'],
                    'lineno': func_def.start_point[0]
                }
                func_database.append({
                    'fpath': fpath,
                    'metadata': metadata,
                    'doc_list': doc_list,
                    'info': info,
                })
            Utils.dump_pickle(func_database, out_path)

    def get_func_list(self, repo_name):
        files_list = glob.glob(os.path.join(self.repo_dir, repo_name, '**/*.py'), recursive=True)
        func_list = []
        class_dict_list = []
        for file in files_list:
            file_func_list, file_class_dict = self.parse_pyfile(file)
            if len(file_func_list) == 0:
                continue
            func_list.extend(file_func_list)
            if file_class_dict:
                class_dict_list.append(file_class_dict)

        return func_list, class_dict_list

    def parse_pyfile(self, py_file):
        func_list = []
        class_dict = {}
        try:
            root = self.parser.parse(open(py_file, 'r', encoding='utf-8').read().encode()).root_node
        except FileNotFoundError as _:
            print(f'cannot find {py_file}')
            return func_list, class_dict
        except IsADirectoryError as _:
            print(f'isdir {py_file}')
            return func_list, class_dict
        
        def traverse(node, class_def, decorated_def):
            '''
            获取所有实体（class和function）
            '''
            if len(node.children) == 0:
                return
            for i in node.children:
                if i.type == 'class_definition':
                    class_dict[i] = {'type': 'class', 'class_def': i, 'file_path': py_file, 'init': None, 'property': []}
                    traverse(i, i, None)
                elif i.type == 'function_definition':
                    decorators = extract_decorators(decorated_def)
                    func_list.append({'type': 'function', 'class_def': class_def, 'func_def': i, 'file_path': py_file, 'decorators': decorators})
                    if class_def and extract_func_name(i) == '__init__' and class_def in class_dict:
                        class_dict[class_def]['init'] = i
                    if class_def and '@property' in decorators:
                        class_dict[class_def]['property'].append(i)
                    traverse(i, class_def, None)
                elif i.type == 'decorated_definition':
                    traverse(i, class_def, i)
                else:
                    traverse(i, class_def, decorated_def)

        traverse(root, None, None)
        
                
        return func_list, class_dict

