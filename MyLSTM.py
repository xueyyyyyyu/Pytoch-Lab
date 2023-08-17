import torch
from torch import nn
from torch.autograd import Variable


# Define model
class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(MyLSTM, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True)  # 此处input_size是我们word2vec的词向量的维度

    def init_hidden(self, batch_size):  # 初始化两个隐藏向量h0和c0
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, input):  # 不可以自己显式调用，pytorch内部自带调用机制
        self.hidden = self.init_hidden(input.size(0))
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        return lstm_out
