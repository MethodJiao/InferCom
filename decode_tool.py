#编码格式处理脚本

import os
from chardet.universaldetector import UniversalDetector


def detcect_encoding(filepath):
    """检测文件编码
    Args:
        detector: UniversalDetector 对象
        filepath: 文件路径
    Return:
        fileencoding: 文件编码
        confidence: 检测结果的置信度，百分比
    """
    detector = UniversalDetector()
    detector.reset()
    for each in open(filepath, 'rb'):
        detector.feed(each)
        if detector.done:
            break
    detector.close()
    fileencoding = detector.result['encoding']
    confidence = detector.result['confidence']
    if fileencoding is None:
        fileencoding = 'unknown'
        confidence = 0.99
    return fileencoding, confidence * 100

@staticmethod
def handle(input_file, output_dir, target_encoding):
    if output_dir:
        if not os.path.exists(output_dir):
            answer = input(
                f'[-] 无效的导出路径: {output_dir} [-]\n要用转码后的文件直接替换源文件吗? y or n\n')
            if 'y' in answer:
                output_dir = None
            else:
                exit(1)
    for file in input_file:
        if not os.path.isfile(file):
            print(f'[-] 无效的文件路径: {file} [-]')
            continue
        encoding, confidence = detcect_encoding(file)
        print(f'[+] {file}: 编码 -> {encoding} (置信度 {confidence}%) [+]')
        if target_encoding and (encoding != 'unknown') and (confidence > 75.0):
            if target_encoding == encoding:
                print(f'[*] {file} 已经是 {encoding} 编码了，无需转换！[*]')
                continue
            f = open(file, 'r', encoding=encoding, errors='replace')
            text = f.read()
            f.close()
            outpath = os.path.join(output_dir, file) if output_dir else file
            f = open(outpath, 'w', encoding=target_encoding, errors='replace')
            f.write(text)
            f.close()
            print('[+] 转码成功: %s(%s) -> %s(%s) [+] ' % (file, encoding, outpath,target_encoding))

if __name__ == '__main__':
    print("This is a decode tool module.")
    target_encoding = 'utf-8'
    output_dir = 'zzz_decode_test'

    path = "repos/sota_test/C++Examples" #文件夹目录
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    file_list = ['repos/sota_test/C++Examples/自定义对象定义/TowerDemo.cpp']

    handle(file_list, None, target_encoding)