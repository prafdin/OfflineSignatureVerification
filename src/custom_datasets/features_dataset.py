import pandas as pd
from torch.utils.data import Dataset


class FeaturesDataset(Dataset):
    def __init__(self, pickle_file):
        self.df = pd.read_pickle(pickle_file)

    def __len__(self):
        return len(self.df)

    def values(self):
        return self.df['feature']

    def targets(self):
        return self.df['class']

    def __getitem__(self, idx):
        return self.df['feature'][idx], self.df['class'][idx]
