import glob
import re
from tqdm import tqdm
from tree_sitter_languages import get_language, get_parser
import os

from utils.utils import Utils, UnixCoder


def process_class(node):
    class_block = ''
    name = ''
    for child in node.children:
        if child.type == 'class_body':
            class_block = child.text.decode()
        elif child.type == 'identifier':
            name = child.text.decode()
    def_info = ''
    if class_block in node.text.decode():
        def_info = node.text.decode().replace(class_block, '')
        def_info_lines = [i.strip() for i in def_info.splitlines()]
        def_info = '\n'.join(def_info_lines)
    return {'name': name, 'def_info': def_info}

def process_formal_parameters(params_node):
    id_list = []
    for child in params_node.children:
        if child.type == 'formal_parameter':
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
    modifier_type = 'package_private'
    is_static = False
    is_constructor = node.type == 'constructor_declaration'
    name = ''
    params_w_type = ''
    params_wo_type = ''
    func_body = node.text.decode()
    def_info = ''
    block = ''
    return_type = ''
    for child in node.children:
        if child.type == 'block' or child.type == 'constructor_body':
            block = child.text.decode()
        elif child.type == 'modifiers':
            func_modifiers = child.text.decode()
            if 'static' in func_modifiers:
                is_static = True
            if 'public' in func_modifiers:
                modifier_type = 'public'
            elif 'private' in func_modifiers:
                modifier_type = 'private'
            elif 'protected' in func_modifiers:
                modifier_type = 'protected'
        elif child.type == 'identifier':
            name = child.text.decode()
        elif child.type == 'formal_parameters':
            params_w_type = child.text.decode()
            params_wo_type = process_formal_parameters(child)
        elif child.type in ['type_identifier', 'scoped_type_identifier']:
            return_type = child.text.decode()
        elif child.type in ['generic_type']:
            for named_child in child.named_children:
                if named_child.type == 'type_identifier':
                    return_type =  named_child.text.decode()
                    break
    if block in func_body:
        def_info = func_body.replace(block, '').strip()
        def_info_lines = [i.strip() for i in def_info.splitlines()]
        def_info = '\n'.join(def_info_lines)
    
    return {
        'name': name,
        'modifier_type': modifier_type,
        'return_type': return_type,
        'is_static': is_static,
        'params_w_type': params_w_type,
        'params_wo_type': params_wo_type,
        'func_body': func_body,
        'def_info': def_info,
        'is_constructor': is_constructor
    }


def upper_camel_to_lower(name):
    name = name[0].lower() + name[1:]
    return name

class FuncBaseBuilder:
    def __init__(self, repos, repo_dir, encode_cuda='cpu'):
        self.repos = repos
        self.language = get_language('java')
        self.parser = get_parser('java')
        # self.encoder = UnixCoder(encode_cuda)
        self.repo_dir = repo_dir

    def build(self, benchmark=None):
        for repo in self.repos:
            if benchmark is None:
                out_path = f'./cache/func_base/{repo}.pkl'
            else:
                out_path = f'./cache/func_base/{benchmark}_{repo}.pkl'
            if os.path.exists(out_path):
                print(f"{repo}: cache")
                continue
            func_list = self.get_func_list(repo_name=repo)
            func_database = []
            for example in tqdm(func_list):
                func_def = example['func_def']
                class_def = example['class_def']

                example['func'] = process_func(func_def)
                example['class'] = process_class(class_def)
                doc_list = []
                # 非构造函数
                if not example['func']['is_constructor']:
                    first_part = [""]
                    if example['func']['return_type']:
                        first_part.append(f"{example['func']['return_type']} {upper_camel_to_lower(example['func']['return_type'])} = ")
                    second_part = [f"{upper_camel_to_lower(example['class']['name'])}.{example['func']['name']}{example['func']['params_wo_type']}"]
                    if example['func']['is_static']:
                        second_part.append(f"{example['class']['name']}.{example['func']['name']}{example['func']['params_wo_type']}")
                    for i_idx, i in enumerate(first_part):
                        for j_idx, j in enumerate(second_part):
                            doc_list.append({
                                'doc': i + j,
                                'doc_type': (i_idx, j_idx)
                            })
                else:
                    for i_idx, i in enumerate([f"{example['class']['name']} {upper_camel_to_lower(example['class']['name'])} = new {example['func']['name']}", f"new {example['func']['name']}"]):
                        doc_list.append({
                            'doc': i + example['func']['params_wo_type'],
                            'doc_type': (i_idx,)
                        })


                info = example['class']['def_info'] + ' {\n' + '\n'.join(['    ' + i for i in example['func']['def_info'].splitlines()]) + ';\n}'
                # 处理类中类
                if example['parent']:
                    p_class = process_class(example['parent'])
                    extra_doc_list = []
                    for o_doc in doc_list:
                        extra_doc_list.append({
                            'doc': f"{upper_camel_to_lower(p_class['name'])}.{o_doc['doc']}",
                            'doc_type': o_doc['doc_type'] + (2,)
                        })
                    
                    info = p_class['def_info'] + ' {\n' + '\n'.join(['    ' + i for i in info.splitlines()]) + ';\n}'
                    doc_list.extend(extra_doc_list)
                # embedding = self.encoder.encode_text(ue)
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
        files_list = glob.glob(os.path.join(self.repo_dir, repo_name, '**/*.java'), recursive=True)
        func_list = []
        for file in files_list:
            file_func_list = self.parse_jfile(file)
            func_list.extend(file_func_list)
        return func_list

    def parse_jfile(self, j_file):
        root = self.parser.parse(open(j_file, 'r', encoding='utf-8').read().encode()).root_node
        package = None
        func_list = []

        for child in root.children:
            if child.type == 'package_declaration':
                package = child.children[1].text.decode()
            break
        
        def process_class(class_node, parent=None):
            temp_func_list = []
            class_body = None
            for child in class_node.children:
                child_type = child.type
                if child_type == 'class_body':
                    class_body = child

            for child in class_body.children:
                child_type = child.type
                if child_type in ['method_declaration', 'constructor_declaration']:
                    temp_func_list.append({'class_def': class_node, 'parent': parent,'func_def': child, 'file_path': j_file, 'package': package})
                if child_type in ['class_declaration']:
                    process_class(child, class_node)
            func_list.extend(temp_func_list)
        
        for child in root.children:
            if child.type == 'class_declaration':
                process_class(child)
        
        return func_list



