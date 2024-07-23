from train import RNN
import torch
import matplotlib.pyplot as plt
import os, sys
from data import get_loader

def load_last_model():
    # 读取最后一个模型
    path = './model'
    files = os.listdir(path)
    files.sort(key=lambda x: os.path.getmtime(path + '/' + x))
    model = RNN(4)
    model.load_state_dict(torch.load(path + '/' + files[-1]))
    return model

# 读取数据
loader, test = get_loader(50)
inputs = [data for data, _ in test]
targets = [data for _, data in test]
inputs = torch.stack(inputs)
targets = torch.stack(targets)


model = load_last_model()
# model.eval()


outputs, _ = model(inputs, None)
outputs = outputs.detach().numpy().squeeze()

# 画图
plt.plot(targets[:100], label='true')
plt.plot(outputs[:100], label='forecast')
plt.legend()
plt.show()