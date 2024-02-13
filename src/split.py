import sys

import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from custom_datasets import FeaturesDataset


def class_to_idx(feature_dataset):
    return {cls_name: i for i, cls_name in enumerate(feature_dataset.targets())}

def main():
    if len(sys.argv) != 4:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("python split.py feature_dataset_path train_output_dir test_output_path\n")
        sys.stderr.write("Where: 'feature_dataset_path' path to featured dataset file\n")
        sys.stderr.write("and 'train_output_dir' path to dir where train dataset file will be saved\n")
        sys.stderr.write("and 'test_output_path' path to dir where test dataset file will be saved\n")
        sys.exit(1)

    split_params = yaml.safe_load(open("params.yaml"))["split"]
    train_samples_per_user = int(split_params["train_samples_per_user"])

    feature_dataset_path = sys.argv[1]
    train_output_path = sys.argv[2]
    test_output_path = sys.argv[3]

    features_dataset = FeaturesDataset(feature_dataset_path)

    X = range(len(features_dataset))
    y = features_dataset.targets()

    train_size = train_samples_per_user * len(set(y)) / len(y)

    X_train_ids, X_test_ids, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=split_params["salt"], stratify=y)

    train_dataset = Subset(features_dataset, X_train_ids)
    test_dataset = Subset(features_dataset, X_test_ids)

    torch.save(train_dataset, train_output_path)
    torch.save(test_dataset, test_output_path)



if __name__ == '__main__':
    main()



