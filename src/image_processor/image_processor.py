import datetime
import os
from pathlib import Path

import PIL
import numpy as np
from PIL import ImageFilter, ImageEnhance
from PIL.Image import Image
import cv2 as cv
import skimage
from sklearn.linear_model import LinearRegression

TIMESTAMP_STRF = "%S-%f"
DEBUG_OUTPUT_DIR = f"debug/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

class ImageProcessorSettings:
    bin_threshold = -1
    morph_open_kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))  # MORPH_ELLIPSE
    contrast_factor = 20
    dilate_iterations = 1
    erode_iterations = 1


DefaultImageProcessorSettings = ImageProcessorSettings()


def calc_img_cog(image):
    M = cv.moments(image, binaryImage=True)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy


def is_debug():
    return "DEBUG" in os.environ and os.environ["DEBUG"].lower() == 'true'


def save_img(path, image: Image):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    image.save(path)

class ImageProcessor:
    """
    We can't use debug function as decorator over process functions because it fails in multiprocessing operations.
    https://stackoverflow.com/q/56533827
    """

    @staticmethod
    def blur_img(image: Image) -> Image:
        return image.filter(filter=ImageFilter.BLUR)

    @staticmethod
    def img_to_gray(image: Image) -> Image:
        image_copy = image.copy()
        gray_img = image_copy.convert("L")
        if is_debug():
            save_img(
                Path(DEBUG_OUTPUT_DIR).joinpath(f"{datetime.datetime.now().strftime(TIMESTAMP_STRF)}_img_to_gray.png"),
                gray_img)
        return gray_img

    @staticmethod
    def img_to_bin(image: Image) -> Image:
        if image.mode == "1":
            return image
        threshold = -1
        if threshold < 0:
            _, bin_img = cv.threshold(np.array(image, dtype=np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            # bin_img = cv.adaptiveThreshold(np.array(image, dtype=np.uint8), 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
            bin_img = np.invert(bin_img)
            bin_img = PIL.Image.fromarray(bin_img).convert("1")
        else:
            bin_img = image.point(lambda x: 255 * (x < threshold), mode='1')
        if is_debug():
            save_img(
                Path(DEBUG_OUTPUT_DIR).joinpath(f"{datetime.datetime.now().strftime(TIMESTAMP_STRF)}_img_to_bin.png"),
                bin_img)
        return bin_img

    @staticmethod
    def thin_img(image: Image) -> Image:
        image = image.convert("1")
        thinned = skimage.morphology.skeletonize(np.array(image, dtype=np.uint8), method="zhang")
        thinned = PIL.Image.fromarray(thinned.astype(np.uint8) * 255, "L")
        if is_debug():
            save_img(
                Path(DEBUG_OUTPUT_DIR).joinpath(f"{datetime.datetime.now().strftime(TIMESTAMP_STRF)}_thin_img.png"),
                thinned
            )
        return thinned

    @staticmethod
    def morph_open(image: Image) -> Image:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        image = cv.morphologyEx(np.array(image).astype(np.uint8), cv.MORPH_OPEN, kernel)
        image = PIL.Image.fromarray(image.astype(bool))
        if is_debug():
            save_img(
                Path(DEBUG_OUTPUT_DIR).joinpath(f"{datetime.datetime.now().strftime(TIMESTAMP_STRF)}_morph_open.png"),
                image)
        return image

    @staticmethod
    def dilate_img(image: Image) -> Image:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        dilate_iterations = 1
        img = cv.dilate(np.array(image).astype(np.uint8), kernel, iterations=dilate_iterations)
        img = PIL.Image.fromarray(img.astype(bool))
        if is_debug():
            save_img(
                Path(DEBUG_OUTPUT_DIR).joinpath(f"{datetime.datetime.now().strftime(TIMESTAMP_STRF)}_dilate_img.png"),
                img)
        return img

    @staticmethod
    def erode_img(image: Image) -> Image:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))

        erode_iterations = 1
        img = cv.erode(np.array(image).astype(np.uint8), kernel, iterations=erode_iterations)
        img = PIL.Image.fromarray(img.astype(bool))
        if is_debug():
            save_img(
                Path(DEBUG_OUTPUT_DIR).joinpath(f"{datetime.datetime.now().strftime(TIMESTAMP_STRF)}_erode_img.png"),
                img)
        return img

    @staticmethod
    def contrast_img(image: Image) -> Image:
        contrast_factor = 20
        img = ImageEnhance.Contrast(image).enhance(contrast_factor)
        if is_debug():
            save_img(
                Path(DEBUG_OUTPUT_DIR).joinpath(f"{datetime.datetime.now().strftime(TIMESTAMP_STRF)}_contrast_img.png"),
                img)
        return img

    @staticmethod
    def crop_roi(image: Image) -> Image:
        if image.mode != "1":
            raise RuntimeError("Image in binary mode is expected for crop_roi func")

        image = np.array(image).astype(np.uint8)
        cnts = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[-2]

        nh, nw = image.shape[:2]
        min_x, min_y, max_x, max_y = 999999999, 999999999, -1, -1
        for cnt in cnts:
            x, y, w, h = cv.boundingRect(cnt)
            if h >= 0.1 * nh:
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x + w > max_x:
                    max_x = x + w
                if y + h > max_y:
                    max_y = y + h

        cropped_img = PIL.Image.fromarray(image.astype(bool)).crop((min_x, min_y, max_x, max_y))
        if is_debug():
            save_img(
                Path(DEBUG_OUTPUT_DIR).joinpath(f"{datetime.datetime.now().strftime(TIMESTAMP_STRF)}_crop_roi.png"),
                cropped_img)
        return cropped_img

    @staticmethod
    def fix_slope(image: Image) -> Image:
        """
        Expect image in binary mode.
        """
        if image.mode != "1":
            raise RuntimeError("Image in binary mode is expected for fix_slope func")
        image = np.array(image).astype(int)
        image = np.rot90(image, 3)

        indices = np.argwhere(np.apply_along_axis(lambda x: x == 1, axis=0, arr=image))
        X, y = np.split(indices, 2, axis=1)
        reg = LinearRegression().fit(X, y)
        angle = np.degrees(np.arctan(reg.coef_[0][0])).astype(int)
        center = calc_img_cog(image)
        rotate_matrix = cv.getRotationMatrix2D(center=center, angle=-angle, scale=1)
        rotated_image = cv.warpAffine(src=image.astype(float), M=rotate_matrix, dsize=(image.shape[1], image.shape[0]))
        rotated_image = PIL.Image.fromarray(np.rot90(np.array(rotated_image).astype(bool)))
        if is_debug():
            save_img(
                Path(DEBUG_OUTPUT_DIR).joinpath(f"{datetime.datetime.now().strftime(TIMESTAMP_STRF)}_fix_slope.png"),
                rotated_image)
        return rotated_image
