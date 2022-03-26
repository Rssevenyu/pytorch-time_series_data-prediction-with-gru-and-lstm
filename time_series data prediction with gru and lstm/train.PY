'''train完整模块'''

# 用户：Ejemplarr
# 编写时间:2022/3/24 22:10
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from GRU import GRU
from LSTM import LSTM
from data_preparation import data_start,data_prediction_to_f_and_t,dataset_to_Dataset,dataset_split_4sets,lengths,targets

'''
   数据的导入
   可调优数据的定义
   网络实例化
   优化器的定义
   数据搬移至gpu
   损失函数的定义
   开始训练
'''

# 可调参数的定义
BATCH_SIZE = 16
EPOCH = 10
LEARN_RATE = 1e-3


# 数据的导入
x, y = data_start()
dataset_features, dataset_target = data_prediction_to_f_and_t(y, lengths, targets)
trian_features, train_target, test_features, test_target = dataset_split_4sets(dataset_features, dataset_target)
train_set = dataset_to_Dataset(data_features=trian_features, data_target=train_target)

train_set_iter = DataLoader(dataset=train_set,# 将数据封装进Dataloader类
                            batch_size=BATCH_SIZE,
                            shuffle=True,  # 打乱batch与batch之间的顺序
                            drop_last=True)# drop_last = True表示最后不够一个batch就舍弃那些多余的数据

# gpu的定义
device = ('cuda'if torch.cuda.is_available else 'cpu')

# 网络的实例化
net_gru = GRU().to(device)
net_lstm = LSTM().to(device)

# 优化器的定义
optim_gru = optim.Adam(params=net_gru.parameters(), lr=LEARN_RATE)
optim_lstm = optim.Adam(params=net_lstm.parameters(),lr=LEARN_RATE)

# 损失函数的定义
loss_fuc = nn.MSELoss()

# 训练函数的定义
def train_for_gru(data, device, loss_fuc, net, optim, Epoch):
    for epoch in range(Epoch):
        loss_print = []
        for batch_idx, (x, y) in enumerate(data):
            x = x.reshape([BATCH_SIZE, lengths, 1])
            x = x.to(device)
            # print(y.shape)
            y = y.reshape((len(y),targets))
            y = y.to(device)
            # print(y.shape)
            y_pred = net(x)
            loss = loss_fuc(y, y_pred)
            loss_print.append(loss.item())
            # 三大步
            # 网络的梯度值更为0
            net.zero_grad()
            # loss反向传播
            loss.backward()
            # 优化器更新
            optim.step()
        print('GRU:loss:',sum(loss_print)/len(data))

def train_for_lstm(data, device, loss_fuc, net, optim, Epoch):
    for epoch in range(Epoch):
        loss_print = []
        for batch_idx, (x, y) in enumerate(data):
            x = x.reshape([BATCH_SIZE, lengths, 1])
            x = x.to(device)
            # print(y.shape)
            y = y.reshape((len(y),targets))
            y = y.to(device)
            # print(y.shape)
            y_pred = net(x)
            loss = loss_fuc(y, y_pred)
            loss_print.append(loss.item())
            # 三大步
            # 网络的梯度值更为0
            net.zero_grad()
            # loss反向传播
            loss.backward()
            # 优化器更新
            optim.step()
        print('LSTM:loss:',sum(loss_print)/len(data))


def main():
    start = time.perf_counter()
    train_for_gru(train_set_iter, device, loss_fuc, net_gru, optim_gru, EPOCH)
    train_for_lstm(train_set_iter, device, loss_fuc, net_lstm, optim_lstm, EPOCH)
    end = time.perf_counter()
    print('训练时间为：{:.2f}s'.format(end-start))
    #保存模型
    torch.save(net_gru.state_dict(), 'gru.pt')
    torch.save(net_lstm.state_dict(), 'lstm.pt')
if __name__ == '__main__':
    main()