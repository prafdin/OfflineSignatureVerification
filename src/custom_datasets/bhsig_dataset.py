import os

from torchvision.datasets import ImageFolder


class OriginalSignsBHSigDataset(ImageFolder):
    def __init__(self, root: str, transform=None):
        def valid_file(path: str):
            return "G" in os.path.basename(path)

        super().__init__(root, transform, is_valid_file=valid_file)
