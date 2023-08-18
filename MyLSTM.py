import torch
from torch import nn
from torch.autograd import Variable


# Define model
class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes=2):
        super(MyLSTM, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        # 此处 input_size 是我们 word2vec 的词向量的维度；
        # 这里设置了输入的第一个维度为 batchsize，那么在后面构造输入的时候，需要保证第一个维度是 batch size 数量
        self.lstm = nn.LSTM(input_size, hidden_dim, batch_first=True)  # 此处input_size是我们word2vec的词向量的维度
        # 添加全连接层（二分类层）
        self.fc = nn.Linear(hidden_dim, num_classes)

    def init_hidden(self, batch_size):  # 初始化两个隐藏向量h0和c0
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, input):  # 不可以自己显式调用，pytorch内部自带调用机制
        # input 是传递给 lstm 的输入，它的 shape 应该是（每一个文本的词语数量，batch size，词向量维度）
        # 输入的时候需要将 input 构造成
        self.hidden = self.init_hidden(input.size(0))
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        # 将 LSTM 输出送入全连接层
        output = self.fc(lstm_out[:, -1, :])  # 取最后一个时间步的输出
        return output
