import torch
import numpy as np
from torch.utils.data import Dataset
from load_data import lire_alpha_digit

class AlphaDigitsDataset(Dataset):
    def __init__(self, path, chars):
        self.data = lire_alpha_digit(path, chars)[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx])
        return data
