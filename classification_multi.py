import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dataset import load_raw_data


signal_idx = [10, 11, 12, 16, 17, 18, 34, 35, 36, 40, 41, 42, 58, 59, 60]

# multiple binary classifiers
n_classes = 2
data = load_raw_data()
healthy_data = data[0]
healthy_labels = to_categorical(np.zeros(healthy_data.shape[0]), num_classes=n_classes)

accs_dic = {}
for b in list(data.keys())[1:]:
    print(f'*** Beam: {b} ***')
    defective_data = data[b]
    defective_labels = to_categorical(np.ones(healthy_data.shape[0]), num_classes=n_classes)

    n_times = 5
    n_folds = 10
    idxs = list(range(healthy_data.shape[0]))
    accs = []
    for t in range(n_times):
        print(f'*** Iteration: {t + 1} / {n_times} ***')
        for i in range(n_folds):
            print(f'*** Fold: {i + 1} / {n_folds} ***')
            idxs_test = list(range(int(len(idxs) / n_folds) * i, int(len(idxs) / n_folds) * (i + 1)))
            idxs_train = [idx for idx in idxs if idx not in idxs_test]

            x_train = np.concatenate((healthy_data[idxs_train, :, :, :], defective_data[idxs_train, :, :, :]), axis=0)
            y_train = np.concatenate((healthy_labels[idxs_train, :], defective_labels[idxs_train, :]), axis=0)
            x_test = np.concatenate((healthy_data[idxs_test, :, :, :], defective_data[idxs_test, :, :, :]), axis=0)
            y_test = np.concatenate((healthy_labels[idxs_test, :], defective_labels[idxs_test, :]), axis=0)

            model = Sequential()
            model.add(Conv2D(4, (3, 3), padding='same', activation='relu', input_shape=(15, 15, 1)))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dropout(0.5))
            model.add(Dense(n_classes, activation='softmax'))

            ea = EarlyStopping(patience=100)
            cp = ModelCheckpoint('checkpoints/cnn.h5', save_best_only=True)
            cb = [ea, cp]

            model.compile('nadam', 'binary_crossentropy', ['accuracy'])
            history = model.fit(x_train, y_train, batch_size=2, verbose=0, epochs=1000, validation_split=0.2,
                                callbacks=cb)

            model = load_model('checkpoints/cnn.h5')
            test_accuracy = model.evaluate(x_test, y_test)[1]
            accs.append(test_accuracy)

            backend.clear_session()

    accs_dic[b] = np.mean(accs), np.std(accs)

fig, ax = plt.subplots()
ax.bar(np.arange(len(accs_dic)), [m[0] for m in accs_dic.values()], yerr=[m[1] for m in accs_dic.values()],
       align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_xlabel('Beams')
ax.set_ylabel('Accuracy')
ax.set_xticks(np.arange(len(accs_dic)))
ax.set_xticklabels(accs_dic.keys())
ax.yaxis.grid(True)
ax.set_title('Healthy/Defective Classification Accuracy for each Beam')
fig.show()

# multi-class classification
n_classes = 11
data = load_raw_data()

labels = {}
for i, key in enumerate(data.keys()):
    labels[key] = to_categorical(np.ones(data[key].shape[0]) * i, num_classes=len(data.keys()))

n_times = 5
n_folds = 10
idxs = list(range(data[0].shape[0]))
accs = []
for t in range(n_times):
    print(f'*** Iteration: {t + 1} / {n_times} ***')
    for i in range(n_folds):
        print(f'*** Fold: {i + 1} / {n_folds} ***')
        idxs_test = list(range(int(len(idxs) / n_folds) * i, int(len(idxs) / n_folds) * (i + 1)))
        idxs_train = [idx for idx in idxs if idx not in idxs_test]

        train_data = {}
        train_labels = {}
        test_data = {}
        test_labels = {}
        for key in data.keys():
            train_data[key] = data[key][idxs_train, :, :, :]
            test_data[key] = data[key][idxs_test, :, :, :]
            train_labels[key] = labels[key][idxs_train, :]
            test_labels[key] = labels[key][idxs_test, :]

        x_train = np.concatenate(list(train_data.values()))
        y_train = np.concatenate(list(train_labels.values()))
        x_test = np.concatenate(list(test_data.values()))
        y_test = np.concatenate(list(test_labels.values()))

        model = Sequential()
        model.add(Conv2D(4, (3, 3), padding='same', activation='relu', input_shape=(15, 15, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(n_classes, activation='softmax'))

        ea = EarlyStopping(patience=100)
        cp = ModelCheckpoint('checkpoints/cnn.h5', save_best_only=True)
        cb = [ea, cp]

        model.compile('nadam', 'binary_crossentropy', ['accuracy'])
        history = model.fit(x_train, y_train, batch_size=2, epochs=1000, validation_split=0.2, callbacks=cb)

        model = load_model('checkpoints/cnn.h5')
        test_accuracy = model.evaluate(x_test, y_test)[1]
        accs.append(test_accuracy)

        backend.clear_session()

print(f'Number of classes: {n_classes}')
print(f'Accuracy Mean: {np.mean(accs)}')
print(f'Accuracy Std: {np.std(accs)}')
