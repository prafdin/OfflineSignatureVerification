import os
import sys

import numpy as np
import pandas as pd
import yaml
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader
from torchvision import transforms

from custom_datasets import OriginalSignsCedarDataset
from image_processor.image_processor import ImageProcessor



def prepare_images(transform, cedar_dataset_root_dir, output_path):
    original_dataset = OriginalSignsCedarDataset(cedar_dataset_root_dir, transform)

    dataloader = DataLoader(original_dataset, batch_size=32, num_workers=2)
    dataframes = []
    for i in dataloader:
        df = pd.DataFrame(zip(np.asarray(i[0]), np.asarray(i[1])), columns=['image', 'class'])
        dataframes.append(df)

    result = pd.concat(dataframes, ignore_index=True)
    result.to_pickle(output_path)
    sys.stdout.write(f"Successfully saved by path: {output_path}")


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("python prepare.py cedar_dataset_root_dir output_path\n")
        sys.stderr.write("Where: 'cedar_dataset_root_dir' path to CEDAR dataset dir\n")
        sys.stderr.write("and 'output_path' path to result pickle file\n")
        sys.exit(1)

    prepare_params = yaml.safe_load(open("params.yaml"))["prepare"]

    cedar_dataset_root_dir = sys.argv[1]
    output_path = sys.argv[2]
    mode = prepare_params["mode"]
    fixed_size = list(map(int, prepare_params["fixed_size"].split(",")))

    if mode == "default":
        transform = transforms.Compose([
            ImageProcessor.fix_slope,
            ImageProcessor.crop_roi,
            ImageProcessor.img_to_gray,
            ImageProcessor.img_to_bin,
            ImageProcessor.morph_open,
            ImageProcessor.dilate_img,
            ImageProcessor.erode_img,
            transforms.Resize(fixed_size),
            np.array
        ])
    elif mode == "thinned":
        transform = transforms.Compose([
            ImageProcessor.fix_slope,
            ImageProcessor.crop_roi,
            transforms.Resize(fixed_size),
            ImageProcessor.img_to_gray,
            ImageProcessor.img_to_bin,
            ImageProcessor.morph_open,
            ImageProcessor.dilate_img,
            ImageProcessor.erode_img,
            ImageProcessor.thin_img,
            np.array
        ])
    else:
        sys.stderr.write("Not recognized mode option\n")
        sys.exit(1)

    prepare_images(transform, cedar_dataset_root_dir, output_path)


if __name__ == '__main__':
    main()
