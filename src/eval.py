import pickle
import sys
from enum import Enum

import numpy as np
import torch
import yaml
import cv2 as cv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import distance_metrics
from torch.utils.data import DataLoader
from sklearn import metrics
from dvclive import Live

class Metric(Enum):
    EUCLIDEAN = lambda x, y: distance_metrics()['euclidean'](x.reshape(1, -1), y.reshape(1, -1))
    CITYBLOCK = lambda x, y: distance_metrics()['cityblock'](x.reshape(1, -1), y.reshape(1, -1))
    MANHATTAN = lambda x, y: distance_metrics()['manhattan'](x.reshape(1, -1), y.reshape(1, -1))
    BHATTACHARYYA = lambda x, y: cv.compareHist(x.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32), cv.HISTCMP_BHATTACHARYYA)
    CORRELATION = lambda x, y: cv.compareHist(x.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32), cv.HISTCMP_CORREL)
    KV_DIVERGENCE = lambda x, y: cv.compareHist(x.reshape(-1, 1).astype(np.float32), y.reshape(-1, 1).astype(np.float32), cv.HISTCMP_KL_DIV)


def eval(model, X, y, split, live):
    predictions_by_class = model.predict_proba(X)
    avg_prec = metrics.average_precision_score(y, predictions_by_class)
    live.log_metric("avg_prec", avg_prec)


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("python eval.py train_dataset_path test_dataset_path\n")
        sys.stderr.write("Where: 'train_dataset_path' path to train dataset file\n")
        sys.stderr.write("and 'test_dataset_path' path to test dataset file\n")
        sys.exit(1)

    train_dataset_path = sys.argv[1]
    test_dataset_path = sys.argv[2]

    eval_params = yaml.safe_load(open("params.yaml"))["eval"]

    use_classifier = eval_params["use_cls"]

    if use_classifier == "knn":
        k = eval_params["knn"]["k"]
        metric = eval_params["knn"]["metric"]
        classifier = KNeighborsClassifier(n_neighbors=k, metric=Metric.__dict__[metric])

    X_train = []
    y_train = []
    train_dataset = torch.load(train_dataset_path)
    dataloader = DataLoader(train_dataset, batch_size=32, num_workers=2)
    for i in dataloader:
        X_train.extend(np.asarray(i[0]))
        y_train.extend(np.asarray(i[1]))
    classifier.fit(X_train, y_train)

    test_dataset = torch.load(test_dataset_path)

    X = []
    y = []
    dataloader = DataLoader(test_dataset, batch_size=32, num_workers=2)
    for i in dataloader:
        X.extend(np.asarray(i[0]))
        y.extend(np.asarray(i[1]))

    X = np.asarray(X)
    y = np.asarray(y)

    with Live("eval", dvcyaml=False) as live:
        eval(classifier, X, y, "test", live)


if __name__ == '__main__':
    main()