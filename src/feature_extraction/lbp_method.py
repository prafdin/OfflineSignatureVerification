import numpy as np
from skimage import feature

def calc_lbp(image):
    LPB_patterns = feature.local_binary_pattern(image, 8, 1, method="uniform").ravel()
    LPB_patterns = list(map(lambda x: int(x), LPB_patterns))
    n_bins = int(np.max(LPB_patterns) + 1)
    hist1, _ = np.histogram(LPB_patterns, density=True, bins=n_bins, range=(0, n_bins))
    return hist1