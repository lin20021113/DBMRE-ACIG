import torch

# 直接加载整个模型
model = torch.load('55.pth')

# 切换到评估模式，用于推理
model.eval()