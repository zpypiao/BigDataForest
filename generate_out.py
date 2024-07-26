from train import RNN
import torch
import matplotlib.pyplot as plt
import os, sys
from data import get_loader, get_test_data
import pandas as pd
import pickle
from xgboost import DMatrix
from torchsummary import summary

df = pd.read_csv('./数据/测试/Ks20_22_01.csv')

def load_last_rnn_model():
    # 读取最后一个模型
    path = './model/rnn'
    files = os.listdir(path)
    files.sort(key=lambda x: os.path.getmtime(path + '/' + x))
    model = RNN(4)
    model.load_state_dict(torch.load(path + '/' + files[-1]))
    return model

def load_xgb_model():
    path = './model/xgboost'
    files = os.listdir(path)
    files.sort(key=lambda x: os.path.getmtime(path + '/' + x))
    model_path = path + '/' + files[-1]
    model = pickle.load(open(model_path, 'rb'))
    return model

def load_rf_mdoel():
    path = './model/random_forest'
    files = os.listdir(path)
    files.sort(key=lambda x: os.path.getmtime(path + '/' + x))
    model_path = path + '/' + files[-1]
    model = pickle.load(open(model_path, 'rb'))
    return model

# 读取数据
# loader, test = get_loader(50)
# inputs = [data for data, _ in test]
# targets = [data for _, data in test]
# inputs = torch.stack(inputs)
# targets = torch.stack(targets)


# model = load_last_model()
# # model.eval()


# outputs, _ = model(inputs, None)
# outputs = outputs.detach().numpy().squeeze()


# RNN outcome
test_data = get_test_data()
model = load_last_rnn_model()
inputs = torch.from_numpy(test_data).float()
outputs, _ = model(inputs, None)
outputs = outputs.detach().numpy().squeeze()
df['RNN'] = outputs

test_data = get_test_data(trans=False)
# XGB outcome
model = load_xgb_model()
test_data = DMatrix(test_data)
outputs = model.predict(test_data)
df['XGB'] = outputs

# Random Forest outcome

model = load_rf_mdoel()
test_data = get_test_data(trans=False)

outputs = model.predict(test_data)
df['Random Forest'] = outputs

df.to_csv('./outcome/final.csv', index=False)


split = 100

# # 画图
# plt.plot(targets[:200], label='true')
# plt.plot(outputs[:200], label='forecast')
# plt.legend()
# plt.show()