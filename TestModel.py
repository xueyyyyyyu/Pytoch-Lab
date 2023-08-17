import torch
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
    # epochs = 5

    # Load test data
    test_data = IMDBDataset(file='vectors_test.csv')
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

    # Define loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Test loop
    test(test_dataloader, model, loss_fn)
