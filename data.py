import torch.utils.data as Data
import pandas as pd
import numpy as np
import torch
import pickle

class MyData(Data.Dataset):
    def __init__(self):
        self.data, self.target = load_data()
        self.trans()

    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]
        return data, target

    def __len__(self):
        return len(self.data)
    
    def trans(self):
        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0)
        # self.mean = np.mean(self.target, axis=0)
        # self.std = np.std(self.target, axis=0)
        self.data = (self.data - mean)/std
        # self.target = (self.target - self.mean)/self.std
        self.data = torch.from_numpy(self.data).float()
        self.target = torch.from_numpy(self.target).float()
        # 存储mean 和 std
        my_dict = {'mean': mean, 'std': std}
        with open('./params/mean_std.pkl', 'wb') as f:
            pickle.dump(my_dict, f)

    # def inverse(self, labels):
    #     labels = labels * self.std + self.mean
    #     return labels
    
class MyDataTest(Data.Dataset):
    def __init__(self):
        self.data, self.target = load_data('./data/test.csv')
        self.trans()

    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]
        return data, target

    def __len__(self):
        return len(self.data)
    
    def trans(self):
        with open('./params/mean_std.pkl', 'rb') as f:
            my_dict = pickle.load(f)

        mean = my_dict['mean']
        std = my_dict['std']
        self.data = (self.data - mean)/std
        self.data = torch.from_numpy(self.data).float()
        self.target = torch.from_numpy(self.target).float()


def load_data(path='./数据/cno.csv'):
    data = pd.read_csv(path)
    data = data.values
    X = data[:, :-1]
    y = data[:, -1]
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    # X = torch.from_numpy(X).float()
    # y = torch.from_numpy(y).float()
    return X, y

def get_loader(batch_size):
    torch_dataset = MyData()
    # 区分训练集和测试集
    train_size = int(0.8 * len(torch_dataset))
    test_size = len(torch_dataset) - train_size
    train_dataset, test_dataset = Data.random_split(torch_dataset, [train_size, test_size])
    # 标准化
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    return train_loader, test_dataset

if __name__ == '__main__':
    loader, test = get_loader(100)
    inputs = [data for data, _ in test]
    targets = [data for _, data in test]
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    print(inputs.shape)