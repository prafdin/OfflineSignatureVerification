import os

from torchvision.datasets import ImageFolder


class OriginalSignsCedarDataset(ImageFolder):
    def __init__(self, root: str, transform=None):
        def valid_file(path: str):
            return "original" in os.path.basename(path)

        super().__init__(root, transform, is_valid_file=valid_file)
