import os
import h5py
import pickle
import numpy as np
from utils import real_, norm_row


def dataset(num_features=6):
    data = {}
    healthy_filepath = 'data/healthy.csv'
    healthy = np.genfromtxt(healthy_filepath, delimiter=',')[:, :num_features]
    data['healthy'] = healthy

    beams = ['a1', 'a6', 'a17', 'a18']
    damages = [f'd{i}' for i in range(10, 91, 10)]

    for beam in beams:
        beam_dic = {}
        for damage in damages:
            defective_filepath = f'data/{beam}/{beam}_{damage}.csv'
            defective = np.genfromtxt(defective_filepath, delimiter=',')[:, :num_features]
            beam_dic[int(damage[1:])] = defective
        data[beam] = beam_dic

    return data


def raw_data(signal_idx, n_cls=11, write_to_disk=False):
    n_signal = len(signal_idx)
    dic = {}
    data_path = 'data/raw data/MATLAB'
    write_path = 'data/raw data/csv'

    with h5py.File(os.path.join(data_path, 'DICAM_TOWER_spectra_H.mat'), 'r') as f:
        healthy = real_(np.array(f['FFT_X2dotfl']))[:, :, signal_idx].transpose(0, 2, 1)
    healthy_small = np.zeros((healthy.shape[0], n_signal, n_signal))
    for row in range(n_signal):
        arg = np.argmax(healthy[0, row, :])
        healthy_small[:, row, :] = healthy[:, row, (arg - int(n_signal / 2)):(arg + int(n_signal / 2) + 1)]
    for i in range(healthy_small.shape[0]):
        healthy_small[i, :, :] = norm_row(healthy_small[i, :, :])
    healthy_small = np.expand_dims(healthy_small, axis=-1)
    dic[0] = healthy_small
    if write_to_disk:
        with open(os.path.join(write_path, '0.pickle'), 'wb') as f:
            pickle.dump(healthy_small, f)
    del healthy

    defective_beams = [5, 6, 13, 17, 18, 22, 51, 57, 66, 67]
    for c, b in enumerate(defective_beams):
        with h5py.File(os.path.join(data_path, f'DICAM_TOWER_spectra_A{b}_D40.mat'), 'r') as f:
            defective = real_(np.array(f['FFT_X2dotfl']))[:, :, signal_idx].transpose(0, 2, 1)
        defective_small = np.zeros((defective.shape[0], n_signal, n_signal))
        for row in range(n_signal):
            arg = np.argmax(defective[0, row, :])
            defective_small[:, row, :] = defective[:, row, (arg - int(n_signal / 2)):(arg + int(n_signal / 2) + 1)]
        for i in range(defective_small.shape[0]):
            defective_small[i, :, :] = norm_row(defective_small[i, :, :])
        defective_small = np.expand_dims(defective_small, axis=-1)
        dic[b] = defective_small
        if write_to_disk:
            with open(os.path.join(write_path, f'{b}.pickle'), 'wb') as f:
                pickle.dump(defective_small, f)
        del defective
        if c + 2 == n_cls:
            break

    return dic


def load_raw_data():
    dic = {}
    data_path = 'data/raw data/csv'
    files = os.listdir(data_path)
    files = sorted(files, key=lambda x: int(x[:-7]))
    for file in files:
        with open(os.path.join(data_path, file), 'rb') as f:
            data = pickle.load(f)
            dic[int(file[:-7])] = data
    return dic
