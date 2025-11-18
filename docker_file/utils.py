import json
import pickle

import os


# repo_dir = 'repos/python/'
# repo_dir = 'repocoder_repos/'
def mk_dir(path): # path是指定文件夹路径
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)




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




