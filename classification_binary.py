import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import real_, norm_row, norm_pixel


signal_idx = [10, 11, 12, 16, 17, 18, 34, 35, 36, 40, 41, 42, 58, 59, 60]

data_path = 'data/raw data/MATLAB'
with h5py.File(os.path.join(data_path, 'DICAM_TOWER_spectra_H.mat'), 'r') as f:
    healthy = real_(np.array(f['FFT_X2dotfl']))[:, :15000, signal_idx].transpose(0, 2, 1)
with h5py.File(os.path.join(data_path, 'DICAM_TOWER_spectra_A17_D40.mat'), 'r') as f:
    defective = real_(np.array(f['FFT_X2dotfl']))[:, :15000, signal_idx].transpose(0, 2, 1)

healthy_small = np.zeros((100, 15, 15))
defective_small = np.zeros((100, 15, 15))

for row in range(healthy.shape[1]):
    arg = np.argmax(healthy[0, row, :])
    healthy_small[:, row, :] = healthy[:, row, (arg - 7):(arg + 8)]
for row in range(defective.shape[1]):
    arg = np.argmax(defective[0, row, :])
    defective_small[:, row, :] = defective[:, row, (arg - 7):(arg + 8)]

for i in range(healthy_small.shape[0]):
    healthy_small[i, :, :] = norm_row(healthy_small[i, :, :])
for i in range(defective_small.shape[0]):
    defective_small[i, :, :] = norm_row(defective_small[i, :, :])

# all_small = np.concatenate((healthy_small, defective_small), axis=0)
# all_small = norm_pixel(all_small)
# healthy_small = all_small[:healthy.shape[0], :, :]
# defective_small = all_small[healthy.shape[0]:, :, :]

healthy_small = np.expand_dims(healthy_small, axis=-1)
defective_small = np.expand_dims(defective_small, axis=-1)

n_times = 1
n_folds = 10
idxs = list(range(100))
accs = []
for t in range(n_times):
    print(f'*** Iteration: {t + 1} / {n_times} ***')
    for i in range(n_folds):
        print(f'*** Fold: {i + 1} / {n_folds} ***')
        idxs_test = list(range(int(len(idxs) / n_folds) * i, int(len(idxs) / n_folds) * (i + 1)))
        idxs_train = [idx for idx in idxs if idx not in idxs_test]

        healthy_train = healthy_small[idxs_train, :, :, :]
        defective_train = defective_small[idxs_train, :, :, :]
        healthy_test = healthy_small[idxs_test, :, :, :]
        defective_test = defective_small[idxs_test, :, :, :]
        x_train = np.concatenate((healthy_train, defective_train), axis=0)
        x_test = np.concatenate((healthy_test, defective_test), axis=0)
        y_train = np.concatenate((np.zeros(100 - int(len(idxs) / n_folds)), np.ones(100 - int(len(idxs) / n_folds))))
        y_test = np.concatenate((np.zeros(int(len(idxs) / n_folds)), np.ones(int(len(idxs) / n_folds))))
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        model = Sequential()
        model.add(Conv2D(4, (3, 3), padding='same', activation='relu', input_shape=(15, 15, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        ea = EarlyStopping(patience=100)
        cp = ModelCheckpoint('checkpoints/cnn.h5', save_best_only=True)
        cb = [ea, cp]

        model.compile('nadam', 'binary_crossentropy', ['accuracy'])
        history = model.fit(x_train, y_train, batch_size=2, verbose=0, epochs=1000, validation_split=0.2, callbacks=cb)

        model = load_model('checkpoints/cnn.h5')
        test_accuracy = model.evaluate(x_test, y_test)[1]
        accs.append(test_accuracy)

        backend.clear_session()

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

print(f'Accuracy Mean: {np.mean(accs)}')
print(f'Accuracy Std: {np.std(accs)}')
