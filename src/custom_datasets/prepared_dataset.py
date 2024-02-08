import pandas as pd
from torch.utils.data import Dataset


class PreparedDataset(Dataset):
    def __init__(self, pickle_file, transform=None):
        self.df = pd.read_pickle(pickle_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(self.df['image'][idx]), self.df['image'][idx]
        else :
            return self.df['image'][idx], self.df['image'][idx]
