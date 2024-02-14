import PIL
import numpy as np
from PIL import ImageFilter, ImageEnhance
from PIL.Image import Image
import cv2 as cv
import skimage

class ImageProcessorSettings:
    bin_threshold = -1
    morph_open_kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2)) # MORPH_ELLIPSE
    contrast_factor = 20
    dilate_iterations = 1
    erode_iterations = 1


DefaultImageProcessorSettings = ImageProcessorSettings()


class ImageProcessor:
    @staticmethod
    def blur_img(image: Image) -> Image:
        return image.filter(filter=ImageFilter.BLUR)

    @staticmethod
    def img_to_gray(image: Image) -> Image:
        image_copy = image.copy()
        return image_copy.convert("L")

    @staticmethod
    def img_to_bin(image: Image) -> Image:
        threshold = -1
        if threshold < 0:
            _, bin_img = cv.threshold(np.array(image, dtype=np.uint8), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            # bin_img = cv.adaptiveThreshold(np.array(image, dtype=np.uint8), 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
            bin_img = np.invert(bin_img)
            return PIL.Image.fromarray(bin_img).convert("1")
        else:
            return image.point(lambda x: 255 * (x < threshold), mode='1')

    @staticmethod
    def thin_img(image: Image) -> Image:
        image = image.convert("1")
        thinned = skimage.morphology.skeletonize(np.array(image, dtype=np.uint8), method="zhang")
        return PIL.Image.fromarray(thinned.astype(np.uint8) * 255, "L")

    @staticmethod
    def morph_open(image: Image) -> Image:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        img = np.array(image).astype(np.uint8)
        img = cv.morphologyEx(np.array(image).astype(np.uint8), cv.MORPH_OPEN, kernel)
        return PIL.Image.fromarray(img.astype(np.uint8) * 255, "L")

    @staticmethod
    def dilate_img(image: Image) -> Image:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        img = np.array(image).astype(np.uint8)

        dilate_iterations = 1
        img = cv.dilate(img, kernel, iterations=dilate_iterations)
        return PIL.Image.fromarray(img.astype(np.uint8)).convert("L")

    @staticmethod
    def erode_img(image: Image) -> Image:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        img = np.array(image).astype(np.uint8)

        erode_iterations = 1
        img = cv.erode(img, kernel, iterations=erode_iterations)
        return PIL.Image.fromarray(img.astype(np.uint8)).convert("L")

    @staticmethod
    def contrast_img(image: Image) -> Image:
        contrast_factor = 20
        return ImageEnhance.Contrast(image).enhance(contrast_factor)

    @staticmethod
    def crop_roi(image: Image) -> Image:
        orig_image = image.copy()
        image = ImageProcessor.img_to_gray(image)
        image = ImageProcessor.img_to_bin(image)

        image = np.array(image.convert('L'))

        kernel = np.ones((15, 15), np.uint8)
        image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)
        image = cv.morphologyEx(image, cv.MORPH_DILATE, kernel)

        cnts = cv.findContours(image, cv.RETR_LIST,
                                cv.CHAIN_APPROX_SIMPLE)[-2]

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
        return orig_image.crop((min_x, min_y, max_x, max_y))