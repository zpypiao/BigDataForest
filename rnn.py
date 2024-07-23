# 使用pytorch-lstm网络进行回归预测
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import datetime


def save_model(model):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'./model/{current_time}.pth'
    torch.save(model.state_dict(), model_path)

# 对数据集进行处理
def read_test_data():
    data = pd.read_csv('\数据\测试\Ks20_22_01.csv')
    factors = ['itime', 'iw', 'ib', 't']
    df = data[factors]
    df['itime'] = pd.to_datetime(df['itime'])
    df['itime'] = df['itime'].dt.hour * 3600 + df['itime'].dt.minute * 60 + df['itime'].dt.second
    return data, sc


# 读取数据
data = pd.read_csv('./数据/cno.csv')
data = data.values


# 数据预处理
# 划分特征与标签
X = data[:, :-1]
y = data[:, -1]
# 特征归一化
sc1 = MinMaxScaler()
X = sc1.fit_transform(X)
# 标签归一化
# sc2 = MinMaxScaler()
# y = sc2.fit_transform(y.reshape(-1, 1))


# 划分数据集
bench = 0.8
bench = int(len(data) * bench)
train_x, train_y = X[:bench], y[:bench]
test_x, test_y = X[bench:], y[bench:]


# sc = MinMaxScaler()

# data = sc.fit_transform(data)

# # 打乱数据
# np.random.shuffle(data)

# # 划分数据集
# bench = 0.8
# bench = int(len(data) * bench)
# train, test = data[:bench], data[bench:]

# # 划分特征与标签
# train_x, train_y = train[:, :-1], train[:, -1]
# test_x, test_y = test[:, :-1], test[:, -1]



# 转换数据格式
train_x = np.array(train_x, dtype=np.float32)
train_y = np.array(train_y, dtype=np.float32)
test_x = np.array(test_x, dtype=np.float32)
test_y = np.array(test_y, dtype=np.float32)



class RNN(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Sequential(nn.Linear(32, 128),
                                nn.ReLU(),
                                nn.Linear(128,1))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        outs = []
        for time in range(r_out.size(1)):
            outs.append(self.fc(self.dropout(r_out[:, time, :])))
        return torch.stack(outs, dim=1), h_state
    

if __name__ == '__main__':
    # 创建模型
    model = RNN(4)  # Example model initialization
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # 学习率下降
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    h_state = None

    plt.ion()  # Turn on interactive mode for live updates
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # Create two subplots vertically
    losses = []  # To store loss values
    model.train()
    epochs = 200
    for epoch in tqdm(range(epochs)):  # Example range for epochs
        inputs = torch.from_numpy(train_x).unsqueeze(0)
        labels = torch.from_numpy(train_y).unsqueeze(0)

        optimizer.zero_grad()
        outputs, h_state = model(inputs, h_state)
        h_state = h_state.data
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step(loss/10000)

        losses.append(loss.item())  # Store scaled loss

        # Update loss plot
        ax1.cla()  # Clear previous lines
        ax1.plot(losses, label='Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Update forecast vs true plot
        ax2.cla()  # Clear previous lines
        ax2.plot(outputs.detach().numpy().squeeze()[:100], label='Forecast')
        ax2.plot(labels.detach().numpy().squeeze()[:100], label='True')
        ax2.legend()

        plt.pause(0.1)  # Pause to update the plots

        if epoch % 10 == 0:
            tqdm.write(f'Epoch {epoch}, Loss {loss.item()}')

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show the final plots


    # save_model(model)


    # 测试模型
    # model.eval()
    inputs = torch.from_numpy(test_x).unsqueeze(0)
    labels = test_y
    # print(labels)
    outputs, _ = model(inputs, None)
    outputs = outputs.detach().numpy().squeeze()
    # labels = sc2.inverse_transform(labels.reshape(-1, 1))
    # outputs = sc2.inverse_transform(outputs.reshape(-1,1))

    # 画图
    plt.plot(labels, label='true')
    plt.plot(outputs, label='forecast')
    plt.legend()
    plt.show()