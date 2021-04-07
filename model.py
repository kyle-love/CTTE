import torch
import numpy as np
from torch.utils import data
from torch import nn
from torch.autograd import Variable

features = np.load('dataset/processed/features.npy')
np.random.shuffle(features)
labels = np.load('dataset/processed/labels.npy')
np.random.shuffle(labels)

split_ratio = 0.7
count_of_samples = features.shape[0]
count_of_links = labels.shape[3]
batch_size = 512

train_features = torch.Tensor(features[:int(split_ratio * count_of_samples),:,:,:])
train_labels = torch.Tensor(labels[:int(split_ratio * count_of_samples),:,:,:])

test_features = torch.Tensor(features[int(split_ratio * count_of_samples):,:,:,:])
test_labels = torch.Tensor(labels[int(split_ratio * count_of_samples):,:,:,:])

dataset_train = data.TensorDataset(train_features, train_labels)
torch_dataloader = data.DataLoader(dataset_train, shuffle=True, batch_size=batch_size)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

class TrendModel(nn.Module):
    '''Learning Trend'''
    def __init__(self):
        super(TrendModel, self).__init__()
        self.lstm = nn.LSTM(input_size=count_of_links * 12, dropout=0.5, hidden_size=64, num_layers=2, batch_first=True,)
        self.fc = nn.Sequential(nn.Linear(64, count_of_links), nn.LeakyReLU())

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        r_out = r_out[:, -1, :]
        r_out = self.fc(r_out)
        return r_out

class TTEModel(nn.Module):
    "TTE"
    def __init__(self):
        super(TTEModel, self).__init__()
        self.lstm = nn.LSTM(input_size = 1, dropout=0.1, hidden_size=64, num_layers=2, batch_first=True,)
        self.fc == nn.Sequential(nn.Linear(64, 1), nn.LeakyReLU())
    
    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        r_out = r_out[: -1, :]
        r_out = self.fc(r_out)
        return r_out
        
class RPGaoModel(nn.Module):
    def __init__(self):
        super(RPGaoModel, self).__init__()
        self.trend = TrendModel()
    def forward(self, x):
        return self.trend(x)



# Init Model
rpg = RPGaoModel()

optimizer = torch.optim.Adam(rpg.parameters(),lr=0.01)

max_epoch = 10000

loss_func = torch.nn.MSELoss()

with torch.no_grad():
    test_features = Variable(test_features.view(-1, 13, count_of_links * 12))
    test_labels = Variable(test_labels.view(-1, count_of_links * 1))

for epoch in range(max_epoch):
    loss = None
    for step, (x, y) in enumerate(torch_dataloader):
        b_x = Variable(x.view(-1, 13, count_of_links * 12))
        b_y = Variable(y.view(-1, count_of_links * 1))
        output = rpg(b_x)
        loss = loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        test_output = rpg(test_features)
        np_test_output = test_output.detach().numpy()
        np_labels_test = test_labels.numpy()
        mse = ((np_test_output - np_labels_test)**2).mean()
        mape_value = mape(np_test_output, np_labels_test)
        print('Epoch: ',epoch, '| loss:%.4f ' % loss.item(), '| mse:%.4f ' % mse, '| mape:%.4f' % mape_value)