import os
import sys

exe_path = os.path.abspath(__file__)
if getattr(sys, 'frozen', False):
    exe_path = sys.executable
exe_dir = os.path.dirname(exe_path)
print(f"清理路径: {exe_dir}")

user_input = input("确认清理当前文件夹吗：[Y/N]")

if user_input.lower() == 'y':
    print("开始清理")
else:
    print("取消清理")
    exit(0)
    
for root, _, files in os.walk(exe_dir):
    for f in files:
        if not f.endswith(('.cpp', '.h')):
            allpath = os.path.join(root, f)
            os.remove(allpath)
            print(f"删除：{allpath}")
