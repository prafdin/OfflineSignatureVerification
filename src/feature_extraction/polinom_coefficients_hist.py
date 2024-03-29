import numpy as np
from sklearn.linear_model import LinearRegression

_configs = {
    "w_size": 3
}

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

class PolinomCoefficientsHist:
    def __init__(self, configs):
        self.configs = configs

    def __call__(self, *args, **kwargs):
        img = args[0]
        w_size = self.configs["w_size"]
        if np.max(img) == 255:
            img = img / 255
        indices = np.argwhere(np.apply_along_axis(lambda x: x == 1, axis=0, arr=img))
        window_array = [get_shifted_window(img, *coords, w_size) for coords in indices]
        f_vector = []
        for some_window in window_array:
            indices = np.argwhere(np.apply_along_axis(lambda x: x == 1, axis=0, arr=some_window))
            X, y = np.split(indices, 2, axis=1)
            reg = LinearRegression(fit_intercept=False).fit(X, y)
            f_vector.append(reg.coef_)

        feature_v, bins = np.histogram(f_vector, density=True)
        return feature_v



