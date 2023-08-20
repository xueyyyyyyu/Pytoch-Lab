import csv

import pandas as pd
from torch import tensor, float32, zeros, cat
from torch.utils.data import Dataset

zero = [0.0] * 100


class IMDBDataset(Dataset):
    def __init__(self, file, transform=None, target_transform=None):
        self.vectors_labels = pd.read_csv(file)
        self.max_length = self.calculate_max_length()

    def __len__(self):
        return len(self.vectors_labels)

    def calculate_max_length(self):
        max_length = 0
        for vector_str in self.vectors_labels['review']:
            vector = eval(vector_str)
            max_length = max(max_length, len(vector))
        return max_length

    def pad_vector(self, vector):
        padding_size = self.max_length - vector.size(0)
        padding = zeros((padding_size, 100), dtype=float32)
        return cat((vector, padding), dim=0)  # Pad at the end

    def __getitem__(self, idx):
        vector_str = self.vectors_labels.iloc[idx, 0]
        label = self.vectors_labels.iloc[idx, 1]
        # with open('data/new/out.csv', 'w', newline='') as csvfile:
        #     csv_writer = csv.writer(csvfile)
        #     csv_writer.writerow(['Vector', 'Label'])  # Write header
        #     csv_writer.writerow([vector_str, label])

        # vector_list = eval(vector_str)
        #
        vectors_tensor = tensor(vector_str, dtype=float32)
        # padded_vector = self.pad_vector(vectors_tensor)
        # print(vectors_tensor)
        return vectors_tensor, label

