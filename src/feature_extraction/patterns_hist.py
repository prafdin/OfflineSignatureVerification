import numpy as np

def get_shifted_window(data, n_c_x, n_c_y, w_size):
    shifted_window = np.zeros((w_size, w_size))

    c_x, c_y = 1, 1
    offset_x = n_c_x - c_x
    offset_y = n_c_y - c_y

    for i in range(w_size):
        offset_i = i + offset_x
        if offset_i < 0 or offset_i > data.shape[0] - 1:
            continue
        for j in range(w_size):
            offset_j = j + offset_y
            if offset_j < 0 or offset_j > data.shape[1] - 1:
                continue
            shifted_window[i][j] = data[offset_i][offset_j]
    return shifted_window

class PatternsHist:
    def __init__(self, config):
        self.config = config

    def __call__(self, *args, **kwargs):
        img = args[0]
        w_size = self.config["w_size"]
        if np.max(img) == 255:
            img = img / 255
        indices = np.argwhere(np.apply_along_axis(lambda x: x == 1, axis=0, arr=img))
        window_array = [get_shifted_window(img, *coords, w_size) for coords in indices]

        feature_v = [w.ravel().astype(int) for w in window_array]
        feature_v = [int("".join(map(str, v)), 2) for v in feature_v]

        bins = 2 ** (w_size * w_size)
        feature_v, bins = np.histogram(feature_v, density=True, bins=bins, range=(0, bins))

        return feature_v