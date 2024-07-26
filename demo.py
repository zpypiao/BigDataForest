import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 计算mse
def mse(x, y):
    return np.mean((x - y) ** 2)


# 读取数据
data = pd.read_csv('./outcome/final.csv')
print(data.head(100))
# 分离数据
label = ['cno', 'RNN', 'XGB', 'Random Forest']
data = data[label]

# 去除缺失值
data = data[data['cno'] != 0]

# 计算MSE
for i in range(1, 4):
    print(f'{label[i]} MSE:', mse(data['cno'], data[label[i]]))

# 转换
data = data.values

# split data
split = 200

data = data[:split]

# 画多张图
fig, ax = plt.subplots(2, 2, figsize=(20, 10))


# 画图
for i in range(1,4):
    ax = plt.subplot(2, 2, i)
    ax.plot(data[:split, 0], label='cno')
    ax.plot(data[:split,i], label=label[i])
    ax.set_title(f'{label[i]}model')
    ax.set_xlabel('cno')
    ax.set_ylabel('predict')
    ax.legend()

ax = plt.subplot(2, 2, 4)

for i in range(3):
    ax.plot(data[:split, i], label=label[i])

ax.legend()
plt.show()
