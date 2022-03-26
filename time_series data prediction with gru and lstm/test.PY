'''test完整模块'''

# 用户：Ejemplarr
# 编写时间:2022/3/24 22:10
from train import device
from data_preparation import lengths, targets
from train import x, y, dataset_features # 为了保持原始数据相同
from GRU import GRU
from LSTM import LSTM

import torch
import matplotlib.pyplot as plt

# 导入保存好的网络
net_gru = GRU().to(device)
net_gru.load_state_dict(torch.load('gru.pt'))
net_lstm = LSTM().to(device)
net_lstm.load_state_dict(torch.load('lstm.pt'))

# 定义测试函数
def test_for_gru(dataset_features):
    dataset_features = dataset_features.reshape([len(dataset_features), lengths, 1])
    y_pred = net_gru(torch.from_numpy(dataset_features).to(device))
    y_pred = y_pred_to_numpy(y_pred)
    y_pred = y_pred.reshape(y_pred.size,1)
    plt.plot(x, y)
    plt.plot(x[lengths:y_pred.size+lengths], y_pred)
    plt.legend(('data', 'data_pred:{}'.format(targets)), loc='upper right')
    plt.title('GRU')
    plt.show()

def test_for_lstm(dataset_features):
    dataset_features = dataset_features.reshape([len(dataset_features), lengths, 1])
    y_pred = net_lstm(torch.from_numpy(dataset_features).to(device))
    y_pred = y_pred_to_numpy(y_pred)
    y_pred = y_pred.reshape(y_pred.size,1)
    plt.plot(x, y)
    plt.plot(x[lengths:y_pred.size+lengths], y_pred)
    plt.legend(('data', 'data_pred:{}'.format(targets)), loc='upper right')
    plt.title('LSTM')
    plt.show()

def y_pred_to_numpy(y_pred):
    '''
    :param y_pred: 网络的输出
    :return: 一个numpy数组
    '''
    y_pred = y_pred.detach().cpu().numpy()
    return y_pred

if __name__ == '__main__':
    plt.plot(x, y)
    plt.show()
    test_for_gru(dataset_features)
    test_for_lstm(dataset_features)

