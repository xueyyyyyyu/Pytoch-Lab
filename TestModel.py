import torch
from torch import nn
from torch.utils.data import DataLoader
from IMDBDataset import IMDBDataset
from MyLSTM import MyLSTM


def test(dataloader, model, loss_fn):  # 模型测试过程的定义，这个也是模板，以后可以借鉴
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    # Define hyperparameters
    batch_size = 64
    epochs = 5

    # Load test data
    test_data = IMDBDataset(file='data/vectors_test.csv')
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Load pre-trained model
    model = MyLSTM()
    model.load_state_dict(torch.load("model/TrainModel"))

    loss_fn = nn.CrossEntropyLoss()  # 课上我们说过，loss 类型是可以选择的
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 这里的优化器也是可以选择的

    # 下面这个训练和测试的过程也是标准形式，我们用自己的数据也还是这样去写
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), "model/TestModel")  # 模型可以保存下来，这里 model 文件夹要和当前 py 文件在同一个目录下
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
