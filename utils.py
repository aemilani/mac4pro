import numpy as np


def real_(x):
    x_real = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x_real[i, j, k] = x[i, j, k]['real']
    return x_real


def imag_(x):
    x_imag = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x_imag[i, j, k] = x[i, j, k]['imag']
    return x_imag


def abs_(x):
    x_abs = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x_abs[i, j, k] = np.sqrt(np.square(x[i, j, k]['real']) + np.square(x[i, j, k]['imag']))
    return x_abs


def norm_row(x):
    min_ = np.min(x, axis=1)
    max_ = np.max(x, axis=1)
    return (x - np.tile(min_, (x.shape[1], 1)).T) / (
            np.tile(max_, (x.shape[1], 1)).T - np.tile(min_, (x.shape[1], 1)).T)


def norm_pixel(x):
    min_ = np.min(x, axis=0)
    min_ = np.tile(min_, (x.shape[0], 1, 1))
    max_ = np.max(x, axis=0)
    max_ = np.tile(max_, (x.shape[0], 1, 1))
    return (x - min_) / (max_ - min_)
