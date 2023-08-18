import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from IMDBDataset import IMDBDataset


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


if __name__ == '__main__':
    training_data = IMDBDataset(file='data/vectors_train.csv')
    test_data = IMDBDataset(file='data/vectors_test.csv')

    batch_size = 64

    # Create data loaders.
    # 这个也是标准用法，只要按照要求自定义数据集，就可以用标准的 dataloader 加载数据
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = MyLSTM().to(device)

    loss_fn = nn.CrossEntropyLoss()  # 课上我们说过，loss 类型是可以选择的
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 这里的优化器也是可以选择的

    epochs = 5  # 这个训练的轮数也可以设置

    # 下面这个训练和测试的过程也是标准形式，我们用自己的数据也还是这样去写
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), "model/TrainModel")  # 模型可以保存下来，这里 model 文件夹要和当前 py 文件在同一个目录下
    print("Saved PyTorch Model State to the project root folder!")

    classes = [
        "positive",
        "negative"
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
