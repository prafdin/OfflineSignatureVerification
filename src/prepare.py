import os
import sys

import numpy as np
import pandas as pd
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from custom_datasets import OriginalSignsCedarDataset
from image_processor.image_processor import ImageProcessor

general_params = yaml.safe_load(open("params.yaml"))["general"]
CEDAR_DATASET_ROOT_FOLDER = general_params["cedar_path"]


def prepare_images(transform, output_path):
    original_dataset = OriginalSignsCedarDataset(CEDAR_DATASET_ROOT_FOLDER, transform)

    dataloader = DataLoader(original_dataset, batch_size=32, num_workers=2)
    dataframes = []
    for i in dataloader:
        df = pd.DataFrame(zip(np.asarray(i[0]), np.asarray(i[1])), columns=['image', 'class'])
        dataframes.append(df)

    result_path = os.path.join(output_path, "dataset.pkl")

    result = pd.concat(dataframes, ignore_index=True)
    result.to_pickle(result_path)
    sys.stdout.write(f"Successfully saved by path: {result_path}")


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("python prepare.py mode output_path\n")
        sys.stderr.write("Where: 'mode' one of: 'default', 'thinned'\n")
        sys.stderr.write("and 'output_path' path to directory \n")
        sys.exit(1)

    mode = sys.argv[1]
    output_path = sys.argv[2]

    if mode == "default":
        transform = transforms.Compose([
            ImageProcessor.img_to_gray,
            ImageProcessor.img_to_bin,
            ImageProcessor.morph_open,
            ImageProcessor.dilate_img,
            ImageProcessor.erode_img,
            ImageProcessor.thin_img,
            transforms.Resize((400, 600)),  # TODO: Need to customize target size
            np.array
        ])
    elif mode == "thinned":
        transform = transforms.Compose([
            ImageProcessor.img_to_gray,
            ImageProcessor.img_to_bin,
            ImageProcessor.morph_open,
            ImageProcessor.dilate_img,
            ImageProcessor.erode_img,
            ImageProcessor.thin_img,
            transforms.Resize((400, 600)),  # TODO: Need to customize target size
            #  TODO: Add skeletonization in pipeline image
            np.array
        ])
    else:
        sys.stderr.write("Not recognized mode option\n")
        sys.exit(1)

    prepare_images(transform, output_path)


if __name__ == '__main__':
    main()
