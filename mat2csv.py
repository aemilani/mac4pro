import os
import numpy as np
from scipy.io import loadmat


data_path = 'data/MATLAB files/'
np.savetxt(f'data/healthy.csv', loadmat(os.path.join(data_path, os.listdir(data_path)[0]))['Freq'].T, delimiter=',')
for file in os.listdir(data_path)[1:]:
    name = file[12:-4].lower()
    data = loadmat(os.path.join(data_path, file))['Freq'].T
    np.savetxt(f'data/{name}.csv', data, delimiter=',')
