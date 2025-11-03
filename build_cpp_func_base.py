
import glob
from pathlib import Path
import re
from tqdm import tqdm
from tree_sitter_languages import get_language, get_parser
import os

from utils.utils import Utils, UnixCoder


def process_class(node):
    name = ''
    template_params = ''
    inheritance_list = ''

    for child in node.children:
        if child.type == 'type_identifier':
            name = child.text.decode()
        elif child.type == 'template_parameter_list':
            template_params = child.text.decode()
        elif child.type == 'field_declaration_list':
            # In C++, inheritance is part of class declaration
            for sub_child in child.children:
                if sub_child.type == 'base_class_clause':
                    inheritance_list = sub_child.text.decode()

    class_tag = f'class {name}{template_params}{": " + inheritance_list if inheritance_list else ""}'
    return {'name': name, 'template_params': template_params, 'inheritance_list': inheritance_list, 'sign': class_tag}

def process_params(params_node):
    param_list = []
    if not params_node:
        return '()'
    for child in params_node.children:
        if child.type == 'parameter_declaration' or child.type.endswith('declaration'):
            param_str = child.text.decode().strip()
            # Remove default values if any
            if '=' in param_str:
                param_str = param_str.split('=')[0].strip()
            param_list.append(param_str)

    params_str = ''
    if len(param_list) >= 1:
        params_str = param_list[0]
        for param in param_list[1:]:
            params_str += f', {param}'

    return f'({params_str})'


def process_func(node):
    name = ''
    params = ''
    return_type = ''
    template_params = ''

    for child in node.children:
        if child.type == 'parameter_list':
            params = child.text.decode()
        elif child.type == 'identifier' or child.type == 'field_identifier':
            name = child.text.decode()
        elif child.type == 'type_qualifier':
            return_type += child.text.decode() + ' '
        elif child.type == 'type_identifier' or child.type == 'primitive_type':
            return_type += child.text.decode()
        elif child.type == 'template_parameter_list':
            template_params = child.text.decode()

    # 若参数未在一层找到，再在子孙中查找一次
    if not params:
        stack = [node]
        while stack and not params:
            n = stack.pop()
            for c in getattr(n, 'named_children', []):
                if c.type == 'parameter_list':
                    params = process_params(c)
                    break
                stack.append(c)

    # Clean up return_type (remove trailing spaces)
    return_type = return_type.strip()

    sign = f'{return_type + " " if return_type else ""}{name}{template_params}{params}'
    return {'name': name, 'params': params, 'return_type': return_type, 'template_params': template_params, 'sign': sign}

def camel_to_snake(name):
    # 查找每个大写字母，并在其前面加上下划线，但是不包括第一个字符前和连续大写字母之间
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    # 处理连续的大写字母（例如：URLConfig -> URL_Config）
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()

def extract_specifiers(node):
    res = []
    if node:
        for child in node.children:
            if child.type == 'type_qualifier' or child.type == 'storage_class_specifier':
                res.append(child.text.decode().strip())
    return res

def extract_func_name(node):
    # 递归搜索常见名称节点，支持 qualified/scoped/destructor
    id_types = {'identifier', 'field_identifier', 'qualified_identifier', 'scoped_identifier', 'destructor_name'}
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type in id_types:
            return n.text.decode()
        stack.extend(list(getattr(n, 'named_children', [])))
    return None


class FuncBaseBuilder:
    def __init__(self, repos, repo_dir, encode_cuda='cpu'):
        self.repos = repos
        self.language = get_language('cpp')
        self.parser = get_parser('cpp')
        # self.encoder = UnixCoder(encode_cuda)
        self.repo_dir = repo_dir

    def build(self, benchmark=None):
        func_list = []
        dir = self.repo_dir.split('/')[-1]
        out_path = f'./cache/func_base/{benchmark}_{dir}.pkl'
        if os.path.exists(out_path):
            print(f"{dir}: cache")
            return
        if len(self.repos) == 0:
            func_list_temp, class_list = self.get_func_list(repo_name='')
            func_list.extend(func_list_temp)
        for repo in self.repos:
            func_list_temp, class_list = self.get_func_list(repo_name=repo)
            func_list.extend(func_list_temp)
        func_database = []
        # for class_dict in class_list:
        #     print(class_dict)
        for example in tqdm(func_list, desc=f'tree_sitter {dir}'):
            func_def = example['func_def']
            class_def = example['class_def']
            example['func'] = process_func(func_def)
            doc_list = []
            if class_def:
                example['class'] = process_class(class_def)
                if example['func']['name'] in ['__init__', example['class']['name']]:  # C++ constructors
                    for i_idx, i in enumerate([example['class']['name'], f"{camel_to_snake(example['class']['name'])} = {example['class']['name']}"]):
                        for j_idx, j in enumerate([example['func']['params'], '()']):
                            doc_list.append({
                                'doc': i + j,
                                'doc_type': ('init', i_idx, j_idx)
                            })
                elif example['func']['name'].startswith('~'):  # C++ destructors
                    continue  # Skip destructors for now
                else:
                    # Check for static methods
                    if 'static' in example['specifiers']:
                        for i_idx, i in enumerate([f"{camel_to_snake(example['class']['name'])}.{example['func']['name']}", f"{example['class']['name']}.{example['func']['name']}"]):
                            for j_idx, j in enumerate([example['func']['params'], '()']):
                                doc_list.append({
                                    'doc': i + j,
                                    'doc_type': ('static', i_idx, j_idx)
                                })
                    else:
                        for i_idx, i in enumerate([f"{camel_to_snake(example['class']['name'])}.{example['func']['name']}"]):
                            for j_idx, j in enumerate([example['func']['params'], '()']):
                                doc_list.append({
                                    'doc': i + j,
                                    'doc_type': ('common_class', i_idx, j_idx)
                                })
                info = example['class']['sign'] + ':\n    ' + '\n    '.join(example['specifiers']) + '\n    ' + example['func']['sign']
            else:
                example['class'] = None
                for i_idx, i in enumerate([f"{example['func']['name']}", f"{Path(example['file_path']).stem}.{example['func']['name']}"]):
                    for j_idx, j in enumerate([example['func']['params'], '()']):
                        doc_list.append({
                            'doc': i + j,
                            'doc_type': ('common', i_idx, j_idx)
                        })
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
        files_list = glob.glob(os.path.join(self.repo_dir, repo_name, '**/*.cpp'), recursive=True)
        files_list.extend(glob.glob(os.path.join(self.repo_dir, repo_name, '**/*.cc'), recursive=True))
        files_list.extend(glob.glob(os.path.join(self.repo_dir, repo_name, '**/*.cxx'), recursive=True))
        files_list.extend(glob.glob(os.path.join(self.repo_dir, repo_name, '**/*.h'), recursive=True))
        files_list.extend(glob.glob(os.path.join(self.repo_dir, repo_name, '**/*.hpp'), recursive=True))
        files_list.extend(glob.glob(os.path.join(self.repo_dir, repo_name, '**/*.hxx'), recursive=True))

        func_list = []
        class_dict_list = []
        for file in files_list:
            file_func_list, file_class_dict = self.parse_cppfile(file)
            if len(file_func_list) == 0:
                continue
            func_list.extend(file_func_list)
            if file_class_dict:
                class_dict_list.append(file_class_dict)

        return func_list, class_dict_list

    def parse_cppfile(self, cpp_file):
        func_list = []
        class_dict = {}
        try:
            root = self.parser.parse(open(cpp_file, 'r', encoding='utf-8').read().encode()).root_node
        except FileNotFoundError as _:
            print(f'cannot find {cpp_file}')
            return func_list, class_dict
        except IsADirectoryError as _:
            print(f'isdir {cpp_file}')
            return func_list, class_dict
        except Exception as e:
            print(f'parse error {cpp_file}: {e}')
            return func_list, class_dict

        def get_class_name(class_node):
            # 在 class_specifier 子孙中寻找 type_identifier 作为类名，稳健提取
            nodes = [class_node]
            while nodes:
                n = nodes.pop()
                if n.type == 'type_identifier':
                    return n.text.decode()
                nodes.extend(list(getattr(n, 'named_children', [])))
            return ''

        def traverse(node, class_def, specifiers):
            """
            获取所有实体（class和function）
            """
            if len(node.children) == 0:
                return
            for i in node.children:
                if i.type == 'class_specifier':
                    class_dict[i] = {'type': 'class', 'class_def': i, 'file_path': cpp_file, 'constructor': None, 'methods': []}
                    traverse(i, i, None)
                elif i.type == 'function_definition' or i.type == 'declaration':
                    current_specifiers = extract_specifiers(i)
                    if specifiers:
                        current_specifiers.extend(specifiers)

                    # Skip declarations that aren't function definitions in header files
                    if i.type == 'declaration' and cpp_file.endswith(('.h', '.hpp', '.hxx')):
                        continue

                    func_list.append({'type': 'function', 'class_def': class_def, 'func_def': i, 'file_path': cpp_file, 'specifiers': current_specifiers})

                    # Check for constructor
                    # First ensure the class_def is present in class_dict before accessing it
                    #if class_def and class_def in class_dict and extract_func_name(i) == class_dict[class_def]['class_def'].children[1].text.decode():
                    #    class_dict[class_def]['constructor'] = i
                    #elif class_def:
                        # register method under the class entry
                    #    class_dict[class_def]['methods'].append(i)


                    # Check for constructor：使用稳健的类名提取，而非依赖 children[1]
                    func_name = extract_func_name(i)
                    if class_def:
                        class_name = get_class_name(class_def)
                        if func_name and class_name and func_name == class_name:
                            class_dict[class_def]['constructor'] = i
                        else:
                            class_dict[class_def]['methods'].append(i)

                    traverse(i, class_def, None)
                elif i.type == 'template_declaration':
                    # For templates, we traverse inside the template declaration
                    traverse(i, class_def, specifiers)
                else:
                    traverse(i, class_def, specifiers)

        traverse(root, None, None)

        return func_list, class_dict
