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

        window_array = create_2d_square_rolling_window(img, w_size).reshape((-1, w_size, w_size))
        window_array = [w for w in window_array if w.any()]
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



# https://habr.com/ru/articles/489734/#2d
def rolling_window_2d(a, window_shape, dx=3, dy=3):
    if (len(window_shape) > 2):
        raise Exception("Function supports only 2d window")

    shape = a.shape[:-2] + \
            ((a.shape[-2] - window_shape[0]) // dy + 1,) + \
            ((a.shape[-1] - window_shape[1]) // dx + 1,) + \
            (window_shape[0], window_shape[1])  # sausage-like shape with 2D cross-section
    strides = a.strides[:-2] + \
              (a.strides[-2] * dy,) + \
              (a.strides[-1] * dx,) + \
              a.strides[-2:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def create_2d_square_rolling_window(a, square_window_size):
    if (a.shape[0] % square_window_size or a.shape[1] % square_window_size):
        raise Exception("""\
            Some elements of the matrix will not get into the rolling window. 
            Expand the original matrix for the following conditions are met:
            a.shape[0] % square_window_size == 0 and a.shape[1] % square_window_size == 0""".strip())

    return rolling_window_2d(a, (square_window_size, square_window_size), square_window_size, square_window_size)
