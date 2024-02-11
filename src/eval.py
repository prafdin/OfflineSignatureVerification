import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
from dvclive import Live

def eval(model, X, y, split, live):
    predictions_by_class = model.predict_proba(X)
    avg_prec = metrics.average_precision_score(y, predictions_by_class)
    live.log_metric("avg_prec", avg_prec)


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("python eval.py test_dataset_path model_path\n")
        sys.stderr.write("Where: 'test_dataset_path' path to test dataset file\n")
        sys.stderr.write("and 'model_path' path to model file\n")
        sys.exit(1)


    test_dataset_path = sys.argv[1]
    model_path = sys.argv[2]


    with open(model_path, "rb") as fd:
        model = pickle.load(fd)

    test_dataset = torch.load(test_dataset_path)

    X = []
    y = []
    dataloader = DataLoader(test_dataset, batch_size=32, num_workers=2)
    for i in dataloader:
        X.extend(np.asarray(i[0]))
        y.extend(np.asarray(i[1]))

    X = np.asarray(X)
    y = np.asarray(y)

    with Live("eval", save_dvc_exp=True) as live:
        eval(model, X, y, "test", live)


if __name__ == '__main__':
    main()