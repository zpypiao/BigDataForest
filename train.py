import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import datetime

from data import get_loader
def save_model(model):
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'./model/{current_time}.pth'
    torch.save(model.state_dict(), model_path)

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
        for time in range(r_out.size(0)):
            outs.append(self.fc(self.dropout(r_out[time,:])))
        return torch.stack(outs, dim=1), h_state


if __name__ == '__main__':
    # 读取数据
    loader, test = get_loader(50)

    # 创建一个RNN网络
    model = RNN(4)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    # # 训练模型
    # model.train()

    # 定义参数
    EPOCHS = 500
    h_state = None

    # 训练过程
    plt.ion()  # Turn on interactive mode for live updates
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # Create two subplots vertically
    losses = []

    

    for epoch in tqdm(range(EPOCHS)):
        h_state = None
        lossss = 0
        for i, (inputs, labels) in enumerate(loader):
            optimizer.zero_grad()
            outputs, h_state = model(inputs, h_state)
            h_state = h_state.data
            loss = criterion(outputs, labels)
            lossss += loss.item()
            loss.backward()
            optimizer.step()
            
        
        scheduler.step(lossss/2000)

        losses.append(lossss/2000)
        ax1.cla()
        ax1.plot(losses, label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.cla()
        ax2.plot(outputs.detach().numpy().squeeze()[:100], label='Forecast')
        ax2.plot(labels.detach().numpy().squeeze()[:100], label='True')
        ax2.legend()
        plt.pause(0.1)
        
        # 更新图像

        if epoch % 10 == 0:
            tqdm.write(f'Epoch {epoch}, Loss {lossss}')

    plt.ioff()
    plt.show()

    save_model(model)
    # 测试模型
    model.eval()

    inputs = torch.from_numpy(test.data).unsqueeze(0)
    labels = test.target
    outputs, _ = model(inputs, None)
    outputs = outputs.detach().numpy().squeeze()
    # 画图
    plt.plot(labels, label='true')
    plt.plot(outputs, label='forecast')
    plt.legend()
    plt.show()