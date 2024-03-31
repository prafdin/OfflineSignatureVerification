import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

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
        bins = list(map(int, self.configs["bins"].split(",")))
        if np.max(img) == 255:
            img = img / 255
        indices = np.argwhere(np.apply_along_axis(lambda x: x == 1, axis=0, arr=img))
        window_array = [get_shifted_window(img, *coords, w_size) for coords in indices]
        f_vector = []
        for some_window in window_array:
            indices = np.argwhere(np.apply_along_axis(lambda x: x == 1, axis=0, arr=some_window))
            X, y = np.split(indices, 2, axis=1)
            reg = make_pipeline(
                PolynomialFeatures(1),
                LinearRegression(fit_intercept=False)
            ).fit(X, y)
            f_vector.append([*reg.steps[-1][1].coef_[0]])

        x, y = np.array_split(np.array(f_vector), 2, axis=1)
        x = x.reshape(-1)
        y = y.reshape(-1)
        H, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)

        return np.ravel(H)



