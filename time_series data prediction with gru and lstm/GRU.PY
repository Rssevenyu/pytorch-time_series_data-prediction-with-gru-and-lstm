'''GRU完整模块'''

# 用户：Ejemplarr
# 编写时间:2022/3/24 22:09
import torch
import torch.nn as nn
from data_preparation import targets
'''
GRU:
   		对于每个网络框架具体的学习最好参考官网进行学习：

    	https://pytorch.org/docs/master/generated/torch.nn.GRU.html#torch.nn.GRU

    	因为官网对于一个网络的输入和输出的数据的shape讲的特别清楚，对于我来说，看完相关基本原理之后，直接就是打开官网
    仔细阅读一下整个网络的各种数据的shape，以及各种参数的实际意义，最后就是借助简单的数据集跑一个demo。这仅仅是我
    个人的习惯，仅供参考。
    	关于GRU的原理，可以参考某站的李沐老师的动手学习深度学习系列。
'''
'''
    	定义Parameters,从官网上可以看见除了我们下面定义的这两个参数，其他参数都有默认值，如果实现最简单的GRU网络，自己定义一下
    前面两个参数就行了，后面的例如dropout是防止过拟合的，bidirectional是控制是否实现双向的，等等，但是这边我们还需要设置
    batch_first = True，因为一般我们的数据格式都是batch_size在前
'''
INPUT_SIZE = 1# The number of expected features in the input x，就是我们表示子序列中一个数的描述的特征数量，只有一个就填1，一个数字就是1
HIDDEN_SIZE = 64# The number of features in the hidden state h，隐藏状态的特征数
# h0 = torch.zeros([])# h0的shape与hn的shape一样为(D * num_layers, batch_size, hidden_size)
                    # 其中的D = 2 if bidirectional=True otherwise 1，num_layers为GRU的层数
                    # 如果这边不对h0进行定义，则网络中的forward中h0可以直接用None替代，默认全零。

# 定义我们的类
class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size=INPUT_SIZE,# 传入我们上面定义的参数
            hidden_size=HIDDEN_SIZE,# 传入我们上面定义的参数
            batch_first=True,# 为什么设置为True上面解释过了
        )
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 32), # 加入线性层的原因是，GRU的输出，参考官网为(batch_size, seq_len, hidden_size)
            nn.LeakyReLU(),             # 这边的多层全连接，根据自己的输出自己定义就好，
            nn.Linear(32, 16),          # 我们需要将其最后打成（batch_size, output_size）比如单值预测，这个output_size就是1，
            nn.LeakyReLU(),             # 这边我们等于targets
            nn.Linear(16, targets)      # 这边输出的（batch_size, targets）且这个targets是上面一个模块已经定义好了
        )

    def forward(self, input):
        output, h_n = self.gru(input, None)# output:(batch_size, seq_len, hidden_size)，h0可以直接None
        # print(output.shape)
        output = output[:, -1, :]# output:(batch_size, hidden_size)
        output = self.mlp(output)# 进过一个多层感知机，也就是全连接层，output:(batch_size, output_size)
        return output

