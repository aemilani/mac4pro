import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.manifold import TSNE


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


def norm_row(x):
    min_ = np.min(x, axis=1)
    max_ = np.max(x, axis=1)
    return (x - np.tile(min_, (x.shape[1], 1)).T) / (
            np.tile(max_, (x.shape[1], 1)).T - np.tile(min_, (x.shape[1], 1)).T)


f_idx = [10, 11, 12, 16, 17, 18, 34, 35, 36, 40, 41, 42, 58, 59, 60]

data_path = 'data/raw data/MATLAB'
with h5py.File(os.path.join(data_path, 'DICAM_TOWER_spectra_H.mat'), 'r') as f:
    healthy = real_(np.array(f['FFT_X2dotfl']))[:, :15000, f_idx].transpose(0, 2, 1)
with h5py.File(os.path.join(data_path, 'DICAM_TOWER_spectra_A5_D40.mat'), 'r') as f:
    a5 = real_(np.array(f['FFT_X2dotfl']))[:, :15000, f_idx].transpose(0, 2, 1)

all_data = np.concatenate((healthy, a5), axis=2)

for i in range(all_data.shape[0]):
    all_data[i, :, :] = norm_row(all_data[i, :, :])

healthy_norm = all_data[:, :, :15000]
a5_norm = all_data[:, :, 15000:]

healthy_norm = np.expand_dims(healthy_norm, axis=-1)
a5_norm = np.expand_dims(a5_norm, axis=-1)

healthy_resized = tf.image.resize(healthy_norm, (15, 15))
a5_resized = tf.image.resize(a5_norm, (15, 15))

tsne = TSNE(random_state=None, init='random')

emb = tsne.fit_transform(
    np.concatenate((np.reshape(healthy_resized, (100, 225)), np.reshape(a5_resized, (100, 225))), axis=0))

plt.figure()
plt.scatter(emb[:100, 0], emb[:100, 1], label='Healthy')
plt.scatter(emb[100:, 0], emb[100:, 1], label='A5')
plt.legend()
plt.show()
