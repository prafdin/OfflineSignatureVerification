import os
import sys

import numpy as np
import pandas as pd
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from custom_datasets import OriginalSignsCedarDataset
from custom_datasets.bhsig_dataset import OriginalSignsBHSigDataset
from image_processor.image_processor import ImageProcessor
from image_processor import image_processor


def prepare_images(original_dataset, output_path):
    dataloader = DataLoader(original_dataset, batch_size=32, num_workers=2)
    dataframes = []
    for i in dataloader:
        df = pd.DataFrame(zip(np.asarray(i[0]), np.asarray(i[1])), columns=['image', 'class'])
        dataframes.append(df)

    result = pd.concat(dataframes, ignore_index=True)
    result.to_pickle(output_path)
    sys.stdout.write(f"Successfully saved by path: {output_path}")


def main():
    if not (2 < len(sys.argv) < 5):
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("python prepare.py cedar_dataset_root_dir output_path --dry-run=NUM\n")
        sys.stderr.write("Where: 'cedar_dataset_root_dir' path to CEDAR dataset dir\n")
        sys.stderr.write("'output_path' path to result pickle file\n")
        sys.stderr.write("'--dry-run=NUM' flag for run script in dry-run mode, NUM is id of image from dataset. Optional.\n")
        sys.exit(1)

    prepare_params = yaml.safe_load(open("params.yaml"))["prepare"]

    cedar_dataset_root_dir = sys.argv[1]
    output_path = sys.argv[2]
    mode = prepare_params["mode"]
    fixed_size = list(map(int, prepare_params["fixed_size"].split(",")))

    if mode == "default":
        transform = transforms.Compose([
            ImageProcessor.img_to_gray,
            ImageProcessor.img_to_bin,
            ImageProcessor.morph_open,
            ImageProcessor.dilate_img,
            ImageProcessor.fix_slope,
            ImageProcessor.crop_roi,
            transforms.Resize(fixed_size),
            ImageProcessor.erode_img,
            np.array
        ])
    elif mode == "thinned":
        transform = transforms.Compose([
            ImageProcessor.img_to_gray,
            ImageProcessor.img_to_bin,
            ImageProcessor.morph_open,
            ImageProcessor.dilate_img,
            ImageProcessor.erode_img,
            ImageProcessor.fix_slope,
            ImageProcessor.crop_roi,
            transforms.Resize(fixed_size),
            ImageProcessor.thin_img,
            np.array
        ])
    elif mode == "as_is":
        transform = transforms.Compose([
            transforms.Resize(fixed_size),
            np.array
        ])
    else:
        sys.stderr.write("Not recognized mode option\n")
        sys.exit(1)

    if "CEDAR" in cedar_dataset_root_dir:
        original_dataset = OriginalSignsCedarDataset(cedar_dataset_root_dir, transform)
    else:
        original_dataset = OriginalSignsBHSigDataset(cedar_dataset_root_dir, transform)

    if len(sys.argv) > 3 and "--dry-run" in sys.argv[3]:
        id = int(sys.argv[3].split("=")[1])
        os.environ["DEBUG"] = "true"
        image, _ = original_dataset[id]
        sys.stdout.write(f"Script successfully executed in dry-run mode\n")
        sys.stdout.write(f"Output dir: {image_processor.DEBUG_OUTPUT_DIR}\n")
        exit(0)
    else:
        prepare_images(original_dataset, output_path)


if __name__ == '__main__':
    main()
