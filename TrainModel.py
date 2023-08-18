import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from IMDBDataset import IMDBDataset
from MyLSTM import MyLSTM


def train(dataloader, model, loss_fn, optimizer, device):  # 模型训练过程的定义；这个可以看作是模板，以后写 pytorch 程序基本都这样
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


if __name__ == "__main__":
    # Define hyperparameters
    batch_size = 64
    epochs = 5

    # Load training data
    training_data = IMDBDataset(file='data/vectors_train.csv')
    # Create data loaders.
    # 这个也是标准用法，只要按照要求自定义数据集，就可以用标准的 dataloader 加载数据
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Define model, loss function, and optimizer
    model = MyLSTM(100, 128).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Training loop
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, device)
    print("Done!")

    torch.save(model.state_dict(), "model/TCModel")  # 模型可以保存下来，这里 model 文件夹要和当前 py 文件在同一个目录下
    print("Saved PyTorch Model State to the project root folder!")
