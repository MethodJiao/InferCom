import os
from build_cpp_func_base import FuncBaseBuilder as CppFuncBaseBuilder
from utils import Utils

@staticmethod
def read_repo(repo_dir):
    entries = os.listdir(repo_dir)
    repos = [entry for entry in entries if os.path.isdir(os.path.join(repo_dir, entry))]
    func_base_builder = CppFuncBaseBuilder(repos, repo_dir)
    func_base_builder.build(benchmark='test')
    # 设置需要读取的文件路径
    temp2 = Utils.load_pickle('test.pkl')
    # 设置输出文件路径
    outpath = 'cpp_func_test.txt'
    with open(outpath, 'w', encoding='utf-8') as outfile:
        for func in temp2:
            # 将每个方法写入txt文件中，方法之间用****分隔
            outfile.write(func['metadata']['func_body'] + '\n****\n')

if __name__ == '__main__':
    # 设置需要读取的文件路径
    repo_dir = 'Cknowledge'  # Replace with your actual repo directory
    read_repo(repo_dir)



