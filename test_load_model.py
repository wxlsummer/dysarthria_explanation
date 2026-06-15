# ==============================================================
# 废物文件！验证 checkpoint 能不能加载的一次性测试脚本，用完就没用了。
# ==============================================================

import torch
import numpy as np

# 你的模型定义
class MultiTaskMLP(torch.nn.Module):
    def __init__(self, dim, task):
        super(MultiTaskMLP, self).__init__()
        self.task = task
        if task == 'severity':
            self.head_severity = torch.nn.Linear(dim, 4)
       
    def forward(self, x):
        # x: [batch, seq_len, 80]
        x = x.mean(dim=1)  # [batch, 80]
        return self.head_severity(x)

# 测试加载
model_path = "/srv/xw4e25/dysarthria_explanation/models/fbanks_83_severity/fold_1/epoch_001.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiTaskMLP(dim=80, task='severity')
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

print("Checkpoint包含的信息:")
print(f"  fold: {checkpoint['fold']}")
print(f"  epoch: {checkpoint['epoch']}")
print(f"  task_type: {checkpoint['task_type']}")
print(f"  feature: {checkpoint['feature']}")
print(f"  dim: {checkpoint['dim']}")

# 加载模型参数
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print("\n模型加载成功!")
print(f"设备: {device}")

# 测试一个样本
test_input = torch.randn(1, 10, 80).to(device)  # [batch=1, seq_len=10, dim=80]
with torch.no_grad():
    output = model(test_input)
    print(f"\n测试输出形状: {output.shape}")  # 应该是 [1, 4]
    print(f"预测类别: {output.argmax(dim=1).item()}")