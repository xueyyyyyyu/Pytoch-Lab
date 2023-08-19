import pandas as pd
from torch import tensor, float32
from torch.utils.data import Dataset


class IMDBDataset(Dataset):
    def __init__(self, file, transform=None, target_transform=None):
        self.vectors_labels = pd.read_csv(file)

    def __len__(self):
        return len(self.vectors_labels)

    def __getitem__(self, idx):
        vector_str = self.vectors_labels.iloc[idx, 0]
        label = self.vectors_labels.iloc[idx, 1]
        vectors_tensor = tensor(eval(vector_str), dtype=float32)
        return vectors_tensor, label

