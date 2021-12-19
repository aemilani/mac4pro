import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def normalize(data):
    """Normalize the data so that mean = 0 and std = 1"""
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


def denormalize(data, mean, std):
    """Denormalize the data to the specified mean and std"""
    return data * std + mean


def plot_corr(arr, name='healthy'):
    df = pd.DataFrame(arr)
    plt.figure()
    plt.matshow(df.corr())
    plt.colorbar()
    plt.title('Correlation - {}'.format(name.capitalize()))
    plt.savefig('corr_{}.png'.format(name))


healthy = np.genfromtxt('data/healthy.csv', delimiter=',').T
defective = np.genfromtxt('data/a5_d50.csv', delimiter=',').T

# plotting the correlations
plot_corr(healthy, name='healthy')
plot_corr(defective, name='defective')

healthy_n = normalize(healthy)
defective_n = normalize(defective)

print(mean_squared_error(healthy_n, defective_n, squared=False))

for i in range(64):
    plt.figure()
    plt.hist(healthy[:, i], 50, label='Healthy')
    plt.hist(defective[:, i], 50, label='Defective')
    plt.legend()
    plt.title('Data Distribution - Frequency {}'.format(i + 1))
    plt.savefig('data_dist/data_dist_{}.png'.format(i + 1))
    plt.close()
