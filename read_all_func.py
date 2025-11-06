from utils.utils import Utils

# 读取在cache/func_base路径之下的func.pkl文件，其作用是将所有的方法写入txt文件中，方便后续使用
if __name__ == '__main__':
    # 设置需要读取的文件路径
    temp2 = Utils.load_pickle('cache/func_base/sota_test_C++Examples_func.pkl')
    # 设置输出文件路径
    outpath = 'cpp_func.txt'
    with open(outpath, 'w', encoding='utf-8') as outfile:
        for func in temp2:
            # 将每个方法写入txt文件中，方法之间用****分隔
            outfile.write(func + '\n****\n')
