import pickle
import sys

import numpy as np
import torch
import yaml
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("python train.py train_dataset_path output_model_path\n")
        sys.stderr.write("Where: 'train_dataset_path' path to train dataset file\n")
        sys.stderr.write("and 'output_model_path' path to output model file\n")
        sys.exit(1)

    train_params = yaml.safe_load(open("params.yaml"))["train"]

    train_dataset_path = sys.argv[1]
    output_model_path = sys.argv[2]

    train_dataset = torch.load(train_dataset_path)

    neigh = KNeighborsClassifier(n_neighbors=3)

    X = []
    y = []
    dataloader = DataLoader(train_dataset, batch_size=32, num_workers=2)
    for i in dataloader:
        X.extend(np.asarray(i[0]))
        y.extend(np.asarray(i[1]))

    neigh.fit(X, y)

    with open(output_model_path, "wb") as fd:
        pickle.dump(neigh, fd)



if __name__ == '__main__':
    main()