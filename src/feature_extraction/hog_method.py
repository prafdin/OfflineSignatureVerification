from skimage.feature import hog

def calc_hog(image):
    return hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), feature_vector=True)