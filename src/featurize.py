import os
import sys
import yaml

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from custom_datasets import PreparedDataset
from feature_extraction import calc_hog
from feature_extraction.lbp_method import calc_lbp


def featurize(transform, dataset_path, output_path):
    prepared_dataset = PreparedDataset(dataset_path, transform)
    dataloader = DataLoader(prepared_dataset, batch_size=32, num_workers=2)
    dataframes = []
    for i in dataloader:
        df = pd.DataFrame(zip(np.asarray(i[0]), np.asarray(i[1])), columns=['feature', 'class'])
        dataframes.append(df)
    result = pd.concat(dataframes, ignore_index=True)

    result.to_pickle(output_path)
    sys.stdout.write(f"Successfully saved by path: {output_path}")

def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("python prepare.py prepared_path output_path\n")
        sys.stderr.write("Where 'prepared_path' path to pickle file with prepared images\n")
        sys.stderr.write("and 'output_path' path to result pickle file\n")
        sys.exit(1)

    featurization_params = yaml.safe_load(open("params.yaml"))["featurization"]

    prepared_path = sys.argv[1]
    output_path = sys.argv[2]
    method = featurization_params["method"]

    if method == "hog":
        transform = transforms.Compose([
            calc_hog
        ])
    elif method == "lbp":
        transform = transforms.Compose([
            calc_lbp
        ])
    else:
        sys.stderr.write("Not recognized method\n")
        sys.exit(1)

    featurize(transform, prepared_path, output_path)

if __name__ == "__main__":
    main()

