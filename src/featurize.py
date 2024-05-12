import itertools
import sys

import torch
import yaml

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from custom_datasets import PreparedDataset
from feature_extraction import calc_hog
from feature_extraction import calc_lbp
from feature_extraction.patterns_hist import PatternsHist
from feature_extraction.polinom_coefficients_hist import PolinomCoefficientsHist


def featurize(transform, dataset_path, output_path):
    prepared_dataset = PreparedDataset(dataset_path, transform)
    dataloader = DataLoader(prepared_dataset, batch_size=1)
    dataframes = []
    for i in dataloader:
        df = pd.DataFrame(zip(np.asarray(i[0]), np.asarray(i[1])), columns=['feature', 'class'])
        dataframes.append(df)
    result = pd.concat(dataframes, ignore_index=True)

    result.to_pickle(output_path)
    sys.stdout.write(f"Successfully saved by path: {output_path}")

def to_rgb(image):
    return image.convert("RGB")

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
        featurize(transform, prepared_path, output_path)
    elif method == "lbp":
        transform = transforms.Compose([
            calc_lbp
        ])
        featurize(transform, prepared_path, output_path)
    elif method == "patterns_hist":
        transform = transforms.Compose([
            PatternsHist(featurization_params["patterns_hist"])
        ])
        featurize(transform, prepared_path, output_path)
    elif method == "polinom_coefficients_hist":
        transform = transforms.Compose([
            PolinomCoefficientsHist(featurization_params["polinom_coefficients_hist"])
        ])
        featurize(transform, prepared_path, output_path)
    elif method == "inceptionv3":
        transform = transforms.Compose([
            Image.fromarray,
            to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        prepared_dataset = PreparedDataset(prepared_path, transform)
        dataloader = DataLoader(prepared_dataset, batch_size=32, num_workers=2)
        model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)

        layer = model._modules.get('avgpool')

        embedings = []

        def copy_embedings(m, i, o):
            o = o[:, :, 0, 0].detach().numpy().tolist()
            embedings.append(o)

        _ = layer.register_forward_hook(copy_embedings)
        model.eval()
        Y = []
        for X, y in dataloader:
            model(X)
            Y.append(np.asarray(y))

        result = pd.DataFrame(zip(np.asarray(list(itertools.chain(*embedings))), np.asarray(list(itertools.chain(*Y)))), columns=['feature', 'class'])
        result.to_pickle(output_path)
    else:
        sys.stderr.write("Not recognized method\n")
        sys.exit(1)

if __name__ == "__main__":
    main()

