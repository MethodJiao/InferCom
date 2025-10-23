import torch

# 创建一些数据并移至GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.tensor([1.0, 2.0, 3.0], device=device)
y = torch.tensor([4.0, 5.0, 6.0], device=device)

# 在GPU上执行加法操作
z = x + y
print(z)