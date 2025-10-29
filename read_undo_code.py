from utils.utils import Utils

#此函数所接受的文件路径，必须放置在repo文件夹之下，否则系统将会报错，在第二步的执行中

class ReadUndoCode:

    @staticmethod
    def read_undo_code(repos, filepath, outpath, context_start_linen=0):
        index = len(repos.split('/'))
        path = filepath.split('/')[index:]
        task_id = filepath
        task_type = 's1'
        ground_truth = ''
        line_no = 0
        col = 0
        import_no = 0
        content = ''
        todo=[]
        with open (filepath, 'r') as file:
            lines = file.readlines()
            for index, line in enumerate(lines):
                content = content + line
                if index == len(lines) - 1:
                    line_no = len(lines)-1
                    col = len(line)
                    if line.endswith('\n'):
                        col = col - 1
                if(line.startswith('import ') or line.startswith('from ')):
                    import_no = index+1
            file.close()
    
        import_index = [import_no, import_no]
        # ground_truth 字段描述代码原有的正确的补充，用于评估填充效果
        metadata = {'task_id': task_id, 'task_type': task_type, 'ground_truth': ground_truth, 'fpath_tuple': path, 'context_start_lineno': context_start_linen, 'line_no': line_no, 'col': col, 'dep': "", 'import_no': import_index}
        res = {'prompt':content, 'metadata': metadata}
        todo.append(res)
        Utils.dump_jsonl(todo, outpath)
        in_path = 'prompts/sota_test/pybenchmark_2k.jsonl'
        Utils.dump_jsonl(todo, in_path)

if __name__ == '__main__':
    filepath = 'repos/sota_test/projectA/Triangularprism参数化.py'
    repo_path = 'repos/sota_test/'
    output_path = '11111undocode.jsonl'
    ReadUndoCode.read_undo_code(filepath, output_path)
