import torch
import numpy as np
from torch.utils import data
from torch import nn
from torch.autograd import Variable
import json
import torch.optim as optim

traj = json.load(open('dataset/processed/traj_samples_10000_without_time'))
times = json.load(open('dataset/processed/time_samples_10000'))

traj = np.asarray(traj)
times = np.asarray(times)

split_ratio = 0.7
count_of_samples = traj.shape[0]
batch_size = 512

train_features = torch.Tensor(traj[:int(split_ratio * count_of_samples), :])
train_labels = torch.Tensor(times[:int(split_ratio * count_of_samples)])

test_features = torch.Tensor(traj[int(split_ratio * count_of_samples):, :, :])
test_labels = torch.Tensor(times[int(split_ratio * count_of_samples):])

log = open('log_200.txt', 'w')
dataset_train = data.TensorDataset(train_features, train_labels)
torch_dataloader = data.DataLoader(dataset_train, shuffle=True, batch_size=batch_size)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def mape_cal(y_true, y_label):
    return np.mean()


class RNNEncoder(nn.Module):
    def __init__(self):
        super(RNNEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=64, num_layers=2, dropout=0.5, batch_first=True)
        self.res = nn.Sequential(nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU())
        self.fc = nn.Sequential(nn.Linear(64, 1), nn.LeakyReLU())

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        r_out = r_out[:, -1, :]
        out = self.fc(self.res(r_out) + r_out)
        return out


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=64, num_layers=2, batch_first=True)
        self.res = nn.Sequential(nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU())
        self.fc = nn.Sequential(nn.Linear(64, 1), nn.LeakyReLU())

    def forward(self, data):
        r_out, _ = self.lstm(data, None)
        r_out = r_out[:, -1, :]
        out = self.fc(self.res(r_out) + r_out)
        return out


model = Network()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
max_epoch = 10000
criterion = torch.nn.MSELoss()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=10, hidden_size=64, num_layers=2, batch_first=True)
        self.res = nn.Sequential(nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU())
        self.fc = nn.Sequential(nn.Linear(64, 1), nn.LeakyReLU())

    def forward(self, data):
        out, _ = self.lstm(data, None)
        r_out = out[:, -1, :]
        out = self.fc(self.res(r_out) + r_out)
        return out


# init Model
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_model = Model()
# test_model.to(device)
# optimizer = torch.Adam.optimizer(test_model.parameters(),lr = 0.1)
optimizer = optim.Adam(model.parameters(), lr=0.1)
criterion = nn.MSELoss()
epoch = 1

# Training Model
for ep in range(epoch):
    for step, (train_x, train_y) in enumerate(torch_dataloader):
        train_x = Variable(train_x.view(-1, 200, 10))
        train_y = Variable(train_y.view(-1, 1))
        # train_x = train_x.to(device)
        # train_y = train_y.to(device)
        out = test_model(train_x)
        loss = criterion(out, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print('Epoch:{},|loss:{},'.format(epoch, loss))

# Init Model
rpg = RNNEncoder()
# rpg.to(device)
optimizer = torch.optim.Adam(rpg.parameters(), lr=0.1)

max_epoch = 10000

loss_func = torch.nn.MSELoss()

with torch.no_grad():
    test_features = Variable(test_features.view(-1, 200, 10))
    test_labels = Variable(test_labels.view(-1, 1))

for epoch in range(max_epoch):
    loss = None
    for step, (x, y) in enumerate(torch_dataloader):
        b_x = Variable(x.view(-1, 200, 10))
        b_y = Variable(y.view(-1, 1))
        # b_x = b_x.to(device)
        # b_y = b_y.to(device)
        output = rpg(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # test_features = test_features.to(device)

        # 释放多余内存 防止报错
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()

        test_output = rpg(test_features).cpu()
        np_test_output = test_output.detach().numpy()
        np_labels_test = test_labels.numpy()
        mse = ((np_test_output - np_labels_test) ** 2).mean()
        mae = np.abs((np_test_output - np_labels_test)).mean()
        mape_value = mape(np_test_output, np_labels_test)
        print('Epoch: ', epoch, '| loss:%.4f ' % loss.item(), '| mae:%.4f ' % mae, '| mse:%.4f ' % mse,
              '| mape:%.4f' % mape_value)
