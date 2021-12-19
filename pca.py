import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def q_stat(r):
    q = []
    for i in range(r.shape[0]):
        q.append(np.matmul(r[i], r[i].T))
    return np.array(q)


def get_pca_components(num_features=6):
    healthy_filepath = 'data/healthy.csv'
    healthy = np.genfromtxt(healthy_filepath, delimiter=',')[:, :num_features]
    mean = np.mean(healthy, axis=0)
    std = np.std(healthy, axis=0)
    healthy_n = (healthy - mean) / std

    pca = PCA(n_components=num_features, svd_solver='full')
    pca.fit(healthy_n)

    print(pca.components_)
    return pca.components_


def plot_feature_1_dist():
    if not os.path.exists('results/f_1'):
        os.mkdir('results/f_1')
    healthy_filepath = 'data/healthy.csv'
    healthy = np.genfromtxt(healthy_filepath, delimiter=',')[:, 0]

    beams = ['a1', 'a6', 'a17', 'a18']
    damages = [f'd{i}' for i in range(10, 91, 10)]

    defectives = []
    for beam in beams:
        data = np.array([])
        for damage in damages:
            defective_filepath = f'data/{beam}/{beam}_{damage}.csv'
            defective = np.genfromtxt(defective_filepath, delimiter=',')[:, 0]
            data = np.concatenate((data, defective))
        defectives.append(data)
    defectives = np.array(defectives)

    labels = beams.copy()
    labels = [l.capitalize() for l in labels]
    for d in range(9):
        plt.figure()
        plt.scatter([0] * healthy.shape[0], healthy, label='Healthy')
        for i, label in enumerate(labels):
            defective = defectives[i, d * 100: (d + 1) * 100]
            plt.scatter([i + 1] * defective.shape[0], defective, label=label)
        plt.title(f'Feature 1: Healthy vs. {(d + 1) * 10}% Cross-section Reduction for Each Beam')
        plt.legend()
        plt.savefig(f'results/f_1/{(d + 1) * 10}.png')


def roc_damage_20_only_feature_1():
    if not os.path.exists('results/f_1'):
        os.mkdir('results/f_1')
    healthy_filepath = 'data/healthy.csv'
    healthy = np.genfromtxt(healthy_filepath, delimiter=',')[:, 0]

    beams = ['a6', 'a17', 'a18']

    val_ratio = 0.5
    val_idx = int(val_ratio * healthy.shape[0])

    val_healthy = healthy[:val_idx]
    test_healthy = healthy[val_idx:]

    val_healthy_mean = np.mean(val_healthy)
    val_healthy_std = np.std(val_healthy)

    val_defective  = []
    for beam in beams:
        damage = 'd20'
        defective_filepath = f'data/{beam}/{beam}_{damage}.csv'
        defective = np.genfromtxt(defective_filepath, delimiter=',')[:, 0]
        val_defective.append(defective[:val_idx])

    val_defective = np.concatenate(val_defective, axis=0)

    fpr, fnr = [], []
    threshold_ratios = [i / 4 for i in range(1, 41)]
    for thr in threshold_ratios:
        higher_limit = val_healthy_mean + thr * val_healthy_std
        lower_limit = val_healthy_mean - thr * val_healthy_std

        y_val_healthy = np.zeros(val_healthy.shape)  # 0 = Healthy (negative), 1 = Failed (positive)
        y_val_defective = np.zeros(val_defective.shape)
        for i in range(val_healthy.shape[0]):
            if not lower_limit <= val_healthy[i] <= higher_limit:
                y_val_healthy[i] = 1
        for i in range(val_defective.shape[0]):
            if not lower_limit <= val_defective[i] <= higher_limit:
                y_val_defective[i] = 1
        fpr.append(np.count_nonzero(y_val_healthy) / y_val_healthy.shape[0])
        fnr.append(1 - np.count_nonzero(y_val_defective) / y_val_defective.shape[0])

    print(f'Threshold ratios: {threshold_ratios}')
    print(f'FPR: {fpr}')
    print(f'FNR: {fnr}')

    plt.figure()
    plt.scatter(fpr, fnr)
    plt.xlabel('False positive rate')
    plt.ylabel('False negative rate')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.title(f'Only Considering Feature 1: ROC Curve for Validation Set')
    plt.savefig(f'results/roc/only_feature_1_damage_20.png')
    

def roc_damage_20_m3(num_features=6):
    if not os.path.exists('results/roc'):
        os.mkdir('results/roc')
    if not os.path.exists(f'results/roc/{num_features}'):
        os.mkdir(f'results/roc/{num_features}')

    healthy_filepath = 'data/healthy.csv'
    healthy = np.genfromtxt(healthy_filepath, delimiter=',')[:, :num_features]

    beams = ['a6', 'a17', 'a18']

    train_ratio = 0.6
    val_ratio = 0.2

    train_idx = int(train_ratio * healthy.shape[0])
    val_idx = int((train_ratio + val_ratio) * healthy.shape[0])

    train = healthy[:train_idx, :]
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    train_n = (train - mean) / std

    val_healthy = healthy[train_idx:val_idx, :]
    test_healthy = healthy[val_idx:, :]

    val_healthy_n = (val_healthy - mean) / std
    test_healthy_n = (test_healthy - mean) / std

    pca = PCA(n_components=1, svd_solver='full')
    pca.fit(train_n)

    val_healthy_recons_n = pca.inverse_transform(pca.transform(val_healthy_n))
    q_val_healthy = q_stat(val_healthy_recons_n - val_healthy_n)

    val_defective = []
    for beam in beams:
        damage = 'd20'
        defective_filepath = f'data/{beam}/{beam}_{damage}.csv'
        defective = np.genfromtxt(defective_filepath, delimiter=',')[:, :num_features]
        val_defective.append(defective[:val_idx, :])

    val_defective = np.concatenate(val_defective, axis=0)
    val_defective_n = (val_defective - mean) / std
    val_defective_recons_n = pca.inverse_transform(pca.transform(val_defective_n))
    q_val_defective = q_stat(val_defective_recons_n - val_defective_n)

    conf_range = range(80, 100)
    fpr, fnr = [], []
    for conf in conf_range:
        # Q-statistic threshold
        c_alpha = stats.norm.ppf(conf / 100)  # normal deviate corresponding to 95 percentile
        cov = np.cov(train_n.T)
        d, _ = np.linalg.eig(cov)
        d = np.sort(d)[::-1]
        theta = []
        for i in range(3):
            theta.append(np.sum(np.array(d[1:]) ** (i + 1)))
        h_0 = 1 - (2 * theta[0] * theta[2]) / (3 * theta[1] ** 2)
        q_alpha = theta[0] * np.power((c_alpha * h_0 * np.sqrt(
            2 * theta[1]) / theta[0] + 1 + theta[1] * h_0 * (h_0 - 1) / theta[0] ** 2), 1 / h_0)

        y_val_healthy = np.zeros(q_val_healthy.shape)  # 0 = Healthy (negative), 1 = Failed (positive)
        y_val_defective = np.zeros(q_val_defective.shape)

        for i in range(q_val_healthy.shape[0]):
            if not q_val_healthy[i] <= q_alpha:
                y_val_healthy[i] = 1
        for i in range(q_val_defective.shape[0]):
            if not q_val_defective[i] <= q_alpha:
                y_val_defective[i] = 1

        fpr.append(np.count_nonzero(y_val_healthy) / y_val_healthy.shape[0])
        fnr.append(1 - np.count_nonzero(y_val_defective) / y_val_defective.shape[0])

    print(f'Confidence intervals: {list(conf_range)}')
    print(f'FPR: {fpr}')
    print(f'FNR: {fnr}')

    plt.figure()
    plt.scatter(fpr, fnr)
    plt.xlabel('False positive ratio')
    plt.ylabel('False negative ratio')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.title(f'Model 3: ROC Curve for Validation Set')
    plt.savefig(f'results/roc/m_3_damage_20.png')


def only_feature_1():
    if not os.path.exists('results/f_1'):
        os.mkdir('results/f_1')
    healthy_filepath = 'data/healthy.csv'
    healthy = np.genfromtxt(healthy_filepath, delimiter=',')[:, 0]

    beams = ['a6', 'a17', 'a18']
    damages = [f'd{i}' for i in range(10, 91, 10)]

    val_ratio = 0.5
    val_idx = int(val_ratio * healthy.shape[0])

    val_healthy = healthy[:val_idx]
    test_healthy = healthy[val_idx:]

    val_healthy_mean = np.mean(val_healthy)
    val_healthy_std = np.std(val_healthy)

    val_defective, test_defective = [], []
    for beam in beams:
        for i, damage in enumerate(damages):
            defective_filepath = f'data/{beam}/{beam}_{damage}.csv'
            defective = np.genfromtxt(defective_filepath, delimiter=',')[:, 0]
            if i % 2 == 0:
                val_defective.append(defective)
            else:
                test_defective.append(defective)
    val_defective = np.concatenate(val_defective, axis=0)
    test_defective = np.concatenate(test_defective, axis=0)

    fpr, fnr = [], []
    threshold_ratios = [i / 40 for i in range(1, 401)]
    for thr in threshold_ratios:
        higher_limit = val_healthy_mean + thr * val_healthy_std
        lower_limit = val_healthy_mean - thr * val_healthy_std

        y_val_healthy = np.zeros(val_healthy.shape)  # 0 = Healthy (negative), 1 = Failed (positive)
        y_val_defective = np.zeros(val_defective.shape)
        for i in range(val_healthy.shape[0]):
            if not lower_limit <= val_healthy[i] <= higher_limit:
                y_val_healthy[i] = 1
        for i in range(val_defective.shape[0]):
            if not lower_limit <= val_defective[i] <= higher_limit:
                y_val_defective[i] = 1
        fpr.append(np.count_nonzero(y_val_healthy) / y_val_healthy.shape[0])
        fnr.append(1 - np.count_nonzero(y_val_defective) / y_val_defective.shape[0])

    print(f'Threshold ratios: {threshold_ratios}')
    print(f'FPR: {fpr}')
    print(f'FNR: {fnr}')

    plt.figure()
    plt.scatter(fpr, fnr)
    plt.xlabel('False positive rate')
    plt.ylabel('False negative rate')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.title(f'Only Considering Feature 1: ROC Curve for Validation Set')
    plt.savefig(f'results/roc/only_feature_1.png')

    thr = 1.725
    higher_limit = val_healthy_mean + thr * val_healthy_std
    lower_limit = val_healthy_mean - thr * val_healthy_std

    y_test_healthy = np.zeros(test_healthy.shape)  # 0 = Healthy (negative), 1 = Failed (positive)
    y_test_defective = np.zeros(test_defective.shape)
    for i in range(test_healthy.shape[0]):
        if not lower_limit <= test_healthy[i] <= higher_limit:
            y_test_healthy[i] = 1
    for i in range(test_defective.shape[0]):
        if not lower_limit <= test_defective[i] <= higher_limit:
            y_test_defective[i] = 1
    fpr_test = np.count_nonzero(y_test_healthy) / y_test_healthy.shape[0]
    fnr_test = 1 - np.count_nonzero(y_test_defective) / y_test_defective.shape[0]

    print(f'Test set FPR: {fpr_test}')
    print(f'Test set FNR: {fnr_test}')

    for beam in beams:
        mar = []
        for damage in damages:
            defective_filepath = f'data/{beam}/{beam}_{damage}.csv'
            defective = np.genfromtxt(defective_filepath, delimiter=',')[:, 0]

            y = np.zeros(defective.shape)
            for i in range(defective.shape[0]):
                if not lower_limit <= defective[i] <= higher_limit:
                    y[i] = 1
            mar.append(1 - np.count_nonzero(y) / y.shape[0])

        plt.figure()
        plt.plot(range(10, 91, 10), mar)
        plt.scatter(range(10, 91, 10), mar)
        plt.xlabel('Cross-section Reduction')
        plt.ylabel('Missed Alarm Ratio')
        plt.ylim([-0.1, 1.1])
        plt.title(f'{beam.upper()} Missed Alarm Ratio vs. Cross-section Reduction - Feature 1')
        plt.savefig(f'results/f_1/mar_{beam}.png')


def far_analysis(num_features=10):
    if not os.path.exists('results/far'):
        os.mkdir('results/far')
    healthy_filepath = 'data/healthy.csv'
    healthy = np.genfromtxt(healthy_filepath, delimiter=',')[:, :num_features]

    train_ratio = 0.6
    val_ratio = 0.2

    train_idx = int(train_ratio * healthy.shape[0])
    val_idx = int((train_ratio + val_ratio) * healthy.shape[0])

    train = healthy[:train_idx, :]
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    train_n = (train - mean) / std

    val_healthy = healthy[train_idx:val_idx, :]
    test_healthy = healthy[val_idx:, :]

    val_healthy_n = (val_healthy - mean) / std
    test_healthy_n = (test_healthy - mean) / std

    m1, m2, m3 = [], [], []
    for n_pc in range(1, num_features):
        pca = PCA(n_components=n_pc, svd_solver='full')
        pca.fit(train_n)

        val_healthy_recons_n = pca.inverse_transform(pca.transform(val_healthy_n))
        val_healthy_recons = val_healthy_recons_n * std + mean

        test_healthy_recons_n = pca.inverse_transform(pca.transform(test_healthy_n))
        test_healthy_recons = test_healthy_recons_n * std + mean

        residual_val_healthy = val_healthy_recons - val_healthy
        residual_test_healthy = test_healthy_recons - test_healthy

        residual_val_healthy_means = np.mean(residual_val_healthy, axis=0)
        residual_val_healthy_stds = np.std(residual_val_healthy, axis=0)

        q_test_healthy = q_stat(test_healthy_recons_n - test_healthy_n)

        # Q-statistic threshold
        c_alpha = stats.norm.ppf(0.95)  # normal deviate corresponding to 95 percentile
        cov = np.cov(train_n.T)
        d, _ = np.linalg.eig(cov)
        d = np.sort(d)[::-1]
        theta = []
        for i in range(3):
            theta.append(np.sum(np.array(d[n_pc:]) ** (i + 1)))
        h_0 = 1 - (2 * theta[0] * theta[2]) / (3 * theta[1] ** 2)
        q_alpha = theta[0] * np.power((c_alpha * h_0 * np.sqrt(
            2 * theta[1]) / theta[0] + 1 + theta[1] * h_0 * (h_0 - 1) / theta[0] ** 2), 1 / h_0)

        # Classification based on residuals: Threshold = mean +- 3 * std
        residual_lower_limit = residual_val_healthy_means - 3 * residual_val_healthy_stds
        residual_higher_limit = residual_val_healthy_means + 3 * residual_val_healthy_stds

        y = np.zeros(residual_test_healthy.shape)  # 0 = Healthy (negative), 1 = Failed (positive)
        for j in range(residual_test_healthy.shape[1]):
            for i in range(residual_test_healthy.shape[0]):
                if not residual_lower_limit[j] <= residual_test_healthy[i, j] <= residual_higher_limit[j]:
                    y[i, j] = 1

        y = np.max(y, axis=1)
        far = np.count_nonzero(y) / y.shape[0]
        m1.append(far)

        # Classification based on residuals: Threshold fixed to the minimum with no FPs for healthy validation data
        residual_lower_limit = np.min(residual_val_healthy, axis=0)
        residual_higher_limit = np.max(residual_val_healthy, axis=0)

        y = np.zeros(residual_test_healthy.shape)  # 0 = Healthy (negative), 1 = Failed (positive)
        for j in range(residual_test_healthy.shape[1]):
            for i in range(residual_test_healthy.shape[0]):
                if not residual_lower_limit[j] <= residual_test_healthy[i, j] <= residual_higher_limit[j]:
                    y[i, j] = 1

        y = np.max(y, axis=1)
        far = np.count_nonzero(y) / y.shape[0]
        m2.append(far)

        # Classification based on Q-statistic
        y = np.zeros(q_test_healthy.shape)  # 0 = Healthy (negative), 1 = Failed (positive)
        for i in range(q_test_healthy.shape[0]):
            if not q_test_healthy[i] <= q_alpha:
                y[i] = 1

        far = np.count_nonzero(y) / y.shape[0]
        m3.append(far)

    print(f'Number of features: {num_features}')
    print(f'Method 1 false alarm rates: {m1}')
    print(f'Method 2 false alarm rates: {m2}')
    print(f'Method 3 false alarm rates: {m3}')

    plt.figure()
    plt.bar(np.array(range(1, num_features)) - 0.2, m1, width=0.2, label='Method 1')
    plt.bar(range(1, num_features), m2, width=0.2, label='Method 2')
    plt.bar(np.array(range(1, num_features)) + 0.2, m3, width=0.2, label='Method 3')
    plt.title(f'False Alarm Rates vs. Number of PCs - {num_features} Features')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(f'results/far/far_vs_n_pc_{num_features}.png')

    plt.figure()
    plt.bar(np.array(range(1, num_features)) - 0.1, m1, width=0.2, label='Method 1')
    plt.bar(np.array(range(1, num_features)) + 0.1, m3, width=0.2, label='Method 3')
    plt.title(f'False Alarm Rates vs. Number of PCs - {num_features} Features')
    plt.legend()
    plt.ylim([0, 1])
    plt.savefig(f'results/far/far_vs_n_pc_{num_features}_m1m3.png')

    # Model 1 FAR Analysis
    fars = []
    pcs_range = range(1, num_features)
    threshold_ratios = range(1, 5)
    for n_pc in pcs_range:
        pca = PCA(n_components=n_pc, svd_solver='full')
        pca.fit(train_n)

        val_healthy_recons_n = pca.inverse_transform(pca.transform(val_healthy_n))
        val_healthy_recons = val_healthy_recons_n * std + mean

        test_healthy_recons_n = pca.inverse_transform(pca.transform(test_healthy_n))
        test_healthy_recons = test_healthy_recons_n * std + mean

        residual_val_healthy = val_healthy_recons - val_healthy
        residual_test_healthy = test_healthy_recons - test_healthy

        residual_val_healthy_means = np.mean(residual_val_healthy, axis=0)
        residual_val_healthy_stds = np.std(residual_val_healthy, axis=0)

        far = []
        for thr in threshold_ratios:
            residual_lower_limit = residual_val_healthy_means - thr * residual_val_healthy_stds
            residual_higher_limit = residual_val_healthy_means + thr * residual_val_healthy_stds

            y = np.zeros(residual_test_healthy.shape)  # 0 = Healthy (negative), 1 = Failed (positive)
            for j in range(residual_test_healthy.shape[1]):
                for i in range(residual_test_healthy.shape[0]):
                    if not residual_lower_limit[j] <= residual_test_healthy[i, j] <= residual_higher_limit[j]:
                        y[i, j] = 1

            y = np.max(y, axis=1)
            far_test = np.count_nonzero(y) / y.shape[0]
            far.append(far_test)
        fars.append(far)

    plt.figure()
    for far, n in zip(fars, pcs_range):
        plt.plot(threshold_ratios, far, label=f'{n} PCs')
        plt.scatter(threshold_ratios, far)
    plt.xlabel('Threshold ratio')
    plt.ylabel('False Alarm Rate')
    plt.ylim([-0.1, 1.1])
    plt.legend()
    plt.title(f'Model 1 False Alarm Rate vs. Threshold Ratio - {num_features} Features')
    plt.savefig(f'results/far/m1_far_vs_threshold_{num_features}.png')

    fars = []
    for thr in threshold_ratios:
        far = []
        for n_pc in pcs_range:
            pca = PCA(n_components=n_pc, svd_solver='full')
            pca.fit(train_n)

            val_healthy_recons_n = pca.inverse_transform(pca.transform(val_healthy_n))
            val_healthy_recons = val_healthy_recons_n * std + mean

            test_healthy_recons_n = pca.inverse_transform(pca.transform(test_healthy_n))
            test_healthy_recons = test_healthy_recons_n * std + mean

            residual_val_healthy = val_healthy_recons - val_healthy
            residual_test_healthy = test_healthy_recons - test_healthy

            residual_val_healthy_means = np.mean(residual_val_healthy, axis=0)
            residual_val_healthy_stds = np.std(residual_val_healthy, axis=0)

            residual_lower_limit = residual_val_healthy_means - thr * residual_val_healthy_stds
            residual_higher_limit = residual_val_healthy_means + thr * residual_val_healthy_stds

            y = np.zeros(residual_test_healthy.shape)  # 0 = Healthy (negative), 1 = Failed (positive)
            for j in range(residual_test_healthy.shape[1]):
                for i in range(residual_test_healthy.shape[0]):
                    if not residual_lower_limit[j] <= residual_test_healthy[i, j] <= residual_higher_limit[j]:
                        y[i, j] = 1

            y = np.max(y, axis=1)
            far_test = np.count_nonzero(y) / y.shape[0]
            far.append(far_test)
        fars.append(far)

    plt.figure()
    for far, n in zip(fars, threshold_ratios):
        plt.plot(pcs_range, far, label=f'Threshold ratio = {n} ')
        plt.scatter(pcs_range, far)
    plt.xlabel('Number of PCs')
    plt.ylabel('False Alarm Rate')
    plt.ylim([-0.1, 1.1])
    plt.legend()
    plt.title(f'Model 1 False Alarm Rate vs. Number of PCs - {num_features} Features')
    plt.savefig(f'results/far/m1_far_vs_n_pcs_{num_features}.png')

    plt.figure()
    plt.plot(pcs_range, fars[2])
    plt.scatter(pcs_range, fars[2])
    plt.xlabel('Number of PCs')
    plt.ylabel('False Alarm Rate')
    plt.ylim([-0.1, 1.1])
    plt.title(f'Model 1 False Alarm Rate vs. Number of PCs - {num_features} Features')
    plt.savefig(f'results/far/m1_far_vs_n_pcs_{num_features}_no_thr.png')

    # Model 3 FAR Analysis
    fars = []
    pcs_range = range(1, num_features)
    conf_range = range(90, 100, 3)
    for n_pc in pcs_range:
        pca = PCA(n_components=n_pc, svd_solver='full')
        pca.fit(train_n)

        test_healthy_recons_n = pca.inverse_transform(pca.transform(test_healthy_n))
        q_test_healthy = q_stat(test_healthy_recons_n - test_healthy_n)

        far = []
        for conf in conf_range:
            # Q-statistic threshold
            c_alpha = stats.norm.ppf(conf / 100)  # normal deviate corresponding to 95 percentile
            cov = np.cov(train_n.T)
            d, _ = np.linalg.eig(cov)
            d = np.sort(d)[::-1]
            theta = []
            for i in range(3):
                theta.append(np.sum(np.array(d[n_pc:]) ** (i + 1)))
            h_0 = 1 - (2 * theta[0] * theta[2]) / (3 * theta[1] ** 2)
            q_alpha = theta[0] * np.power((c_alpha * h_0 * np.sqrt(
                2 * theta[1]) / theta[0] + 1 + theta[1] * h_0 * (h_0 - 1) / theta[0] ** 2), 1 / h_0)

            y = np.zeros(q_test_healthy.shape)  # 0 = Healthy (negative), 1 = Failed (positive)
            for i in range(q_test_healthy.shape[0]):
                if not q_test_healthy[i] <= q_alpha:
                    y[i] = 1

            far_test = np.count_nonzero(y) / y.shape[0]
            far.append(far_test)
        fars.append(far)

    plt.figure()
    for far, n in zip(fars, pcs_range):
        plt.plot(conf_range, far, label=f'{n} PCs')
        plt.scatter(conf_range, far)
    plt.xlabel('Normal Confidence Interval')
    plt.ylabel('False Alarm Rate')
    plt.ylim([-0.1, 1.1])
    plt.legend()
    plt.title(f'Model 3 False Alarm Rate vs. Q Threshold Confidence - {num_features} Features')
    plt.savefig(f'results/far/q_far_vs_interval_{num_features}.png')

    fars = []
    for conf in conf_range:
        far = []
        for n_pc in pcs_range:
            pca = PCA(n_components=n_pc, svd_solver='full')
            pca.fit(train_n)

            test_healthy_recons_n = pca.inverse_transform(pca.transform(test_healthy_n))
            q_test_healthy = q_stat(test_healthy_recons_n - test_healthy_n)

            # Q-statistic threshold
            c_alpha = stats.norm.ppf(conf / 100)  # normal deviate corresponding to 95 percentile
            cov = np.cov(train_n.T)
            d, _ = np.linalg.eig(cov)
            d = np.sort(d)[::-1]
            theta = []
            for i in range(3):
                theta.append(np.sum(np.array(d[n_pc:]) ** (i + 1)))
            h_0 = 1 - (2 * theta[0] * theta[2]) / (3 * theta[1] ** 2)
            q_alpha = theta[0] * np.power((c_alpha * h_0 * np.sqrt(
                2 * theta[1]) / theta[0] + 1 + theta[1] * h_0 * (h_0 - 1) / theta[0] ** 2), 1 / h_0)

            y = np.zeros(q_test_healthy.shape)  # 0 = Healthy (negative), 1 = Failed (positive)
            for i in range(q_test_healthy.shape[0]):
                if not q_test_healthy[i] <= q_alpha:
                    y[i] = 1

            far_test = np.count_nonzero(y) / y.shape[0]
            far.append(far_test)
        fars.append(far)

    plt.figure()
    for far, n in zip(fars, conf_range):
        plt.plot(pcs_range, far, label=f'{n}% confidence')
        plt.scatter(pcs_range, far)
    plt.xlabel('Number of PCs')
    plt.ylabel('False Alarm Rate')
    plt.ylim([-0.1, 1.1])
    plt.legend()
    plt.title(f'Model 3 False Alarm Rate vs. Number of PCs - {num_features} Features')
    plt.savefig(f'results/far/q_far_vs_n_pcs_{num_features}.png')

    plt.figure()
    plt.plot(pcs_range, fars[2])
    plt.scatter(pcs_range, fars[2])
    plt.xlabel('Number of PCs')
    plt.ylabel('False Alarm Rate')
    plt.ylim([-0.1, 1.1])
    plt.title(f'Model 3 False Alarm Rate vs. Number of PCs - {num_features} Features')
    plt.savefig(f'results/far/q_far_vs_n_pcs_{num_features}_no_thr.png')


def mar_analysis(num_features=10):
    if not os.path.exists('results/mar'):
        os.mkdir('results/mar')
    healthy_filepath = 'data/healthy.csv'
    healthy = np.genfromtxt(healthy_filepath, delimiter=',')[:, :num_features]

    beams = ['a1', 'a6', 'a17', 'a18']
    damages = [f'd{i}' for i in range(10, 91, 10)]

    train_ratio = 0.6
    val_ratio = 0.2

    train_idx = int(train_ratio * healthy.shape[0])
    val_idx = int((train_ratio + val_ratio) * healthy.shape[0])

    train = healthy[:train_idx, :]
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    train_n = (train - mean) / std

    val_healthy = healthy[train_idx:val_idx, :]
    test_healthy = healthy[val_idx:, :]

    val_healthy_n = (val_healthy - mean) / std
    test_healthy_n = (test_healthy - mean) / std

    for n_pc in range(1, num_features):
        if not os.path.exists(f'results/mar/{n_pc}'):
            os.mkdir(f'results/mar/{n_pc}')

        pca = PCA(n_components=n_pc, svd_solver='full')
        pca.fit(train_n)

        val_healthy_recons_n = pca.inverse_transform(pca.transform(val_healthy_n))
        val_healthy_recons = val_healthy_recons_n * std + mean

        residual_val_healthy = val_healthy_recons - val_healthy

        residual_val_healthy_means = np.mean(residual_val_healthy, axis=0)
        residual_val_healthy_stds = np.std(residual_val_healthy, axis=0)

        # Q-statistic threshold
        c_alpha = stats.norm.ppf(0.95)  # normal deviate corresponding to 95 percentile
        cov = np.cov(train_n.T)
        d, _ = np.linalg.eig(cov)
        d = np.sort(d)[::-1]
        theta = []
        for i in range(3):
            theta.append(np.sum(np.array(d[n_pc:]) ** (i + 1)))
        h_0 = 1 - (2 * theta[0] * theta[2]) / (3 * theta[1] ** 2)
        q_alpha = theta[0] * np.power((c_alpha * h_0 * np.sqrt(
            2 * theta[1]) / theta[0] + 1 + theta[1] * h_0 * (h_0 - 1) / theta[0] ** 2), 1 / h_0)

        for beam in beams:
            mar1, mar2, mar3 = [], [], []
            for damage in damages:
                defective_filepath = f'data/{beam}/{beam}_{damage}.csv'
                defective = np.genfromtxt(defective_filepath, delimiter=',')[:, :num_features]

                test_defective = defective[val_idx:, :]
                test_defective_n = (test_defective - mean) / std

                test_defective_recons_n = pca.inverse_transform(pca.transform(test_defective_n))
                test_defective_recons = test_defective_recons_n * std + mean

                residual_test_defective = test_defective_recons - test_defective

                q_test_defective = q_stat(test_defective_recons_n - test_defective_n)

                # Classification based on residuals
                # Threshold = mean +- 3 * std
                residual_lower_limit = residual_val_healthy_means - 3 * residual_val_healthy_stds
                residual_higher_limit = residual_val_healthy_means + 3 * residual_val_healthy_stds

                y_test_defective = np.zeros(residual_test_defective.shape)
                for j in range(residual_test_defective.shape[1]):
                    for i in range(residual_test_defective.shape[0]):
                        if not residual_lower_limit[j] <= residual_test_defective[i, j] <= residual_higher_limit[j]:
                            y_test_defective[i, j] = 1
                y_test_defective = np.max(y_test_defective, axis=1)

                mar_test = (y_test_defective.shape[0] - np.count_nonzero(y_test_defective)) / y_test_defective.shape[0]
                mar1.append(mar_test)

                # Threshold fixed to the minimum with no false positives for healthy validation data
                residual_lower_limit = np.min(residual_val_healthy, axis=0)
                residual_higher_limit = np.max(residual_val_healthy, axis=0)

                y_test_defective = np.zeros(residual_test_defective.shape)
                for j in range(residual_test_defective.shape[1]):
                    for i in range(residual_test_defective.shape[0]):
                        if not residual_lower_limit[j] <= residual_test_defective[i, j] <= residual_higher_limit[j]:
                            y_test_defective[i, j] = 1
                y_test_defective = np.max(y_test_defective, axis=1)

                mar_test = (y_test_defective.shape[0] - np.count_nonzero(y_test_defective)) / y_test_defective.shape[0]
                mar2.append(mar_test)

                # Classification based on Q-statistic
                y_test_defective = np.zeros(q_test_defective.shape)
                for i in range(q_test_defective.shape[0]):
                    if not q_test_defective[i] <= q_alpha:
                        y_test_defective[i] = 1

                mar_test = (y_test_defective.shape[0] - np.count_nonzero(y_test_defective)) / y_test_defective.shape[0]
                mar3.append(mar_test)

            plt.figure()
            # plt.plot(range(10, 91, 10), mar1, label='Method 1')
            # plt.plot(range(10, 91, 10), mar2, label='Method 2')
            plt.plot(range(10, 91, 10), mar3, label='Method 3')
            # plt.scatter(range(10, 91, 10), mar1)
            # plt.scatter(range(10, 91, 10), mar2)
            plt.scatter(range(10, 91, 10), mar3)
            plt.xlabel('Cross-section Reduction')
            plt.ylabel('Missed Alarm Rate')
            plt.ylim([-0.1, 1.1])
            plt.title(f'{beam.upper()} Missed Alarm Rate vs. Cross-section Reduction - {num_features} Features')
            # plt.legend()
            plt.savefig(f'results/mar/{n_pc}/mar_{beam}_{num_features}.png')


def mar_vs_damage(num_features=6):
    if not os.path.exists('results/mar'):
        os.mkdir('results/mar')
    healthy_filepath = 'data/healthy.csv'
    healthy = np.genfromtxt(healthy_filepath, delimiter=',')[:, :num_features]

    beams = ['a6', 'a17', 'a18']
    damages = [f'd{i}' for i in range(10, 91, 10)]

    train_ratio = 0.6
    val_ratio = 0.2

    train_idx = int(train_ratio * healthy.shape[0])
    val_idx = int((train_ratio + val_ratio) * healthy.shape[0])

    train = healthy[:train_idx, :]
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    train_n = (train - mean) / std

    val_healthy = healthy[train_idx:val_idx, :]
    test_healthy = healthy[val_idx:, :]

    val_healthy_n = (val_healthy - mean) / std
    test_healthy_n = (test_healthy - mean) / std

    val_defective = []
    for damage in damages:
        val_damage = []
        for beam in beams:
            defective_filepath = f'data/{beam}/{beam}_{damage}.csv'
            defective = np.genfromtxt(defective_filepath, delimiter=',')[:, :num_features]
            val_damage.append(defective)
        val_defective.append(np.concatenate(val_damage, axis=0))

    # Model 1
    n_pc = 2
    pca = PCA(n_components=n_pc, svd_solver='full')
    pca.fit(train_n)

    val_healthy_recons_n = pca.inverse_transform(pca.transform(val_healthy_n))
    val_healthy_recons = val_healthy_recons_n * std + mean

    residual_val_healthy = val_healthy_recons - val_healthy

    residual_val_healthy_means = np.mean(residual_val_healthy, axis=0)
    residual_val_healthy_stds = np.std(residual_val_healthy, axis=0)

    thr = 2.25
    residual_lower_limit = residual_val_healthy_means - thr * residual_val_healthy_stds
    residual_higher_limit = residual_val_healthy_means + thr * residual_val_healthy_stds

    mar1 = []
    for defective in val_defective:
        defective_n = (defective - mean) / std
        defective_recons_n = pca.inverse_transform(pca.transform(defective_n))
        defective_recons = defective_recons_n * std + mean
        residual_defective = defective_recons - defective

        y = np.zeros(residual_defective.shape)
        for j in range(residual_defective.shape[1]):
            for i in range(residual_defective.shape[0]):
                if not residual_lower_limit[j] <= residual_defective[i, j] <= residual_higher_limit[j]:
                    y[i, j] = 1
        y = np.max(y, axis=1)

        mar = (y.shape[0] - np.count_nonzero(y)) / y.shape[0]
        mar1.append(mar)

    # Model 3
    n_pc = 1
    pca = PCA(n_components=n_pc, svd_solver='full')
    pca.fit(train_n)

    conf = 92
    # Q-statistic threshold
    c_alpha = stats.norm.ppf(conf / 100)  # normal deviate corresponding to 95 percentile
    cov = np.cov(train_n.T)
    d, _ = np.linalg.eig(cov)
    d = np.sort(d)[::-1]
    theta = []
    for i in range(3):
        theta.append(np.sum(np.array(d[n_pc:]) ** (i + 1)))
    h_0 = 1 - (2 * theta[0] * theta[2]) / (3 * theta[1] ** 2)
    q_alpha = theta[0] * np.power((c_alpha * h_0 * np.sqrt(
        2 * theta[1]) / theta[0] + 1 + theta[1] * h_0 * (h_0 - 1) / theta[0] ** 2), 1 / h_0)

    mar3 = []
    for defective in val_defective:
        defective_n = (defective - mean) / std
        defective_recons_n = pca.inverse_transform(pca.transform(defective_n))
        q_defective = q_stat(defective_recons_n - defective_n)

        y = np.zeros(q_defective.shape)
        for i in range(q_defective.shape[0]):
            if not q_defective[i] <= q_alpha:
                y[i] = 1

        mar = (y.shape[0] - np.count_nonzero(y)) / y.shape[0]
        mar3.append(mar)

    # Only feature 1
    healthy = healthy[:, 0]
    val_ratio = 0.5
    val_idx = int(val_ratio * healthy.shape[0])

    val_healthy = healthy[:val_idx]
    test_healthy = healthy[val_idx:]

    val_healthy_mean = np.mean(val_healthy)
    val_healthy_std = np.std(val_healthy)

    thr = 1.75
    higher_limit = val_healthy_mean + thr * val_healthy_std
    lower_limit = val_healthy_mean - thr * val_healthy_std

    mar4 = []
    for defective in val_defective:
        defective = defective[:, 0]

        y = np.zeros(defective.shape)
        for i in range(defective.shape[0]):
            if not lower_limit <= defective[i] <= higher_limit:
                y[i] = 1
        mar4.append(1 - np.count_nonzero(y) / y.shape[0])

    plt.figure()
    # plt.plot(range(10, 91, 10), mar1, label='Model 1')
    # plt.scatter(range(10, 91, 10), mar1)
    plt.plot(range(10, 91, 10), mar3, label='Model 3')
    plt.scatter(range(10, 91, 10), mar3)
    plt.plot(range(10, 91, 10), mar4, label='Only Feature 1')
    plt.scatter(range(10, 91, 10), mar4)
    plt.xlabel('Cross-section Reduction')
    plt.ylabel('Missed Alarm Ratio')
    plt.ylim([-0.1, 1.1])
    plt.title('Missed Alarm Ratio vs. Damage Level')
    plt.legend()
    plt.savefig('results/mar/mar_vs_damage.png')


def main(num_features=10):
    healthy_filepath = 'data/healthy.csv'
    healthy = np.genfromtxt(healthy_filepath, delimiter=',')[:, :num_features]

    beams = ['a6', 'a17', 'a18']
    damages = [f'd{i}' for i in range(10, 91, 10)]

    train_ratio = 0.6
    val_ratio = 0.2

    train_idx = int(train_ratio * healthy.shape[0])
    val_idx = int((train_ratio + val_ratio) * healthy.shape[0])

    train = healthy[:train_idx, :]
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    train_n = (train - mean) / std

    val_healthy = healthy[train_idx:val_idx, :]
    test_healthy = healthy[val_idx:, :]

    val_healthy_n = (val_healthy - mean) / std
    test_healthy_n = (test_healthy - mean) / std

    variance_ratio = 0.95
    pca = PCA(n_components=variance_ratio, svd_solver='full')
    train_trans_n = pca.fit_transform(train_n)
    n_pc = train_trans_n.shape[1]
    print(f'Number of PCs required to maintain 95% of the variance: {n_pc}')

    val_healthy_recons_n = pca.inverse_transform(pca.transform(val_healthy_n))
    val_healthy_recons = val_healthy_recons_n * std + mean

    test_healthy_recons_n = pca.inverse_transform(pca.transform(test_healthy_n))
    test_healthy_recons = test_healthy_recons_n * std + mean

    residual_val_healthy = val_healthy_recons - val_healthy
    residual_test_healthy = test_healthy_recons - test_healthy

    residual_val_healthy_means = np.mean(residual_val_healthy, axis=0)
    residual_val_healthy_stds = np.std(residual_val_healthy, axis=0)

    q_test_healthy = q_stat(test_healthy_recons_n - test_healthy_n)

    # Q-statistic threshold
    c_alpha = stats.norm.ppf(0.95)  # normal deviate corresponding to 95 percentile
    cov = np.cov(train_n.T)
    d, _ = np.linalg.eig(cov)
    d = np.sort(d)[::-1]
    theta = []
    for i in range(3):
        theta.append(np.sum(np.array(d[n_pc:]) ** (i + 1)))
    h_0 = 1 - (2 * theta[0] * theta[2]) / (3 * theta[1] ** 2)
    q_alpha = theta[0] * np.power((c_alpha * h_0 * np.sqrt(
        2 * theta[1]) / theta[0] + 1 + theta[1] * h_0 * (h_0 - 1) / theta[0] ** 2), 1 / h_0)

    for beam in beams:
        mar1, far1, mar2, far2, mar3, far3 = [], [], [], [], [], []
        for damage in damages:
            defective_filepath = f'data/{beam}/{beam}_{damage}.csv'
            defective = np.genfromtxt(defective_filepath, delimiter=',')[:, :num_features]

            test_defective = defective[val_idx:, :]
            test_defective_n = (test_defective - mean) / std

            test_defective_recons_n = pca.inverse_transform(pca.transform(test_defective_n))
            test_defective_recons = test_defective_recons_n * std + mean

            residual_test_defective = test_defective_recons - test_defective

            q_test_defective = q_stat(test_defective_recons_n - test_defective_n)

            # Classification based on residuals
            # Threshold = mean +- 3 * std

            residual_lower_limit = residual_val_healthy_means - 3 * residual_val_healthy_stds
            residual_higher_limit = residual_val_healthy_means + 3 * residual_val_healthy_stds

            y_test_healthy = np.zeros(residual_test_healthy.shape)  # 0 = Healthy (negative), 1 = Failed (positive)
            y_test_defective = np.zeros(residual_test_defective.shape)
            for j in range(residual_test_healthy.shape[1]):
                for i in range(residual_test_healthy.shape[0]):
                    if not residual_lower_limit[j] <= residual_test_healthy[i, j] <= residual_higher_limit[j]:
                        y_test_healthy[i, j] = 1
            for j in range(residual_test_defective.shape[1]):
                for i in range(residual_test_defective.shape[0]):
                    if not residual_lower_limit[j] <= residual_test_defective[i, j] <= residual_higher_limit[j]:
                        y_test_defective[i, j] = 1

            n_signal_fa_1 = np.count_nonzero(y_test_healthy, axis=1)

            y_test_healthy = np.max(y_test_healthy, axis=1)
            y_test_defective = np.max(y_test_defective, axis=1)

            mar_test = (y_test_defective.shape[0] - np.count_nonzero(y_test_defective)) / y_test_defective.shape[0]
            far_test = np.count_nonzero(y_test_healthy) / y_test_healthy.shape[0]
            mar1.append(mar_test)
            far1.append(far_test)

            # Threshold fixed to the minimum with no false positives for healthy validation data
            residual_lower_limit = np.min(residual_val_healthy, axis=0)
            residual_higher_limit = np.max(residual_val_healthy, axis=0)

            y_test_healthy = np.zeros(residual_test_healthy.shape)  # 0 = Healthy (negative), 1 = Failed (positive)
            y_test_defective = np.zeros(residual_test_defective.shape)
            for j in range(residual_test_healthy.shape[1]):
                for i in range(residual_test_healthy.shape[0]):
                    if not residual_lower_limit[j] <= residual_test_healthy[i, j] <= residual_higher_limit[j]:
                        y_test_healthy[i, j] = 1
            for j in range(residual_test_defective.shape[1]):
                for i in range(residual_test_defective.shape[0]):
                    if not residual_lower_limit[j] <= residual_test_defective[i, j] <= residual_higher_limit[j]:
                        y_test_defective[i, j] = 1

            n_signal_fa_2 = np.count_nonzero(y_test_healthy, axis=1)

            y_test_healthy = np.max(y_test_healthy, axis=1)
            y_test_defective = np.max(y_test_defective, axis=1)

            mar_test = (y_test_defective.shape[0] - np.count_nonzero(y_test_defective)) / y_test_defective.shape[0]
            far_test = np.count_nonzero(y_test_healthy) / y_test_healthy.shape[0]
            mar2.append(mar_test)
            far2.append(far_test)

            # Classification based on Q-statistic
            y_test_healthy = np.zeros(q_test_healthy.shape)  # 0 = Healthy (negative), 1 = Failed (positive)
            y_test_defective = np.zeros(q_test_defective.shape)

            for i in range(q_test_healthy.shape[0]):
                if not q_test_healthy[i] <= q_alpha:
                    y_test_healthy[i] = 1

            for i in range(q_test_defective.shape[0]):
                if not q_test_defective[i] <= q_alpha:
                    y_test_defective[i] = 1

            mar_test = (y_test_defective.shape[0] - np.count_nonzero(y_test_defective)) / y_test_defective.shape[0]
            far_test = np.count_nonzero(y_test_healthy) / y_test_healthy.shape[0]
            mar3.append(mar_test)
            far3.append(far_test)

        plt.figure()
        plt.plot(range(10, 91, 10), mar1, label='Method 1')
        plt.plot(range(10, 91, 10), mar2, label='Method 2')
        plt.plot(range(10, 91, 10), mar3, label='Method 3')
        plt.scatter(range(10, 91, 10), mar1)
        plt.scatter(range(10, 91, 10), mar2)
        plt.scatter(range(10, 91, 10), mar3)
        plt.xlabel('Cross-section Reduction')
        plt.ylabel('Missed Alarm Rate')
        plt.ylim([-0.1, 1.1])
        plt.title('Missed Alarm Rate vs. Cross-section Reduction')
        plt.legend()
        plt.savefig(f'results/mar_{beam}_{num_features}.png')

        plt.figure()
        plt.plot(range(10, 91, 10), far1, label='Method 1')
        plt.plot(range(10, 91, 10), far2, label='Method 2')
        plt.plot(range(10, 91, 10), far3, label='Method 3')
        plt.scatter(range(10, 91, 10), far1)
        plt.scatter(range(10, 91, 10), far2)
        plt.scatter(range(10, 91, 10), far3)
        plt.xlabel('Cross-section Reduction')
        plt.ylabel('False Alarm Rate')
        plt.ylim([-0.1, 1.1])
        plt.title('False Alarm Rate vs. Cross-section Reduction')
        plt.legend()
        plt.savefig(f'results/far_{beam}_{num_features}.png')

    plt.figure()
    plt.plot(range(test_healthy.shape[0]), n_signal_fa_1, label='Method 1')
    plt.plot(range(test_healthy.shape[0]), n_signal_fa_2, label='Method 2')
    plt.scatter(range(test_healthy.shape[0]), n_signal_fa_1)
    plt.scatter(range(test_healthy.shape[0]), n_signal_fa_2)
    plt.xlabel('Samples')
    plt.ylabel('Signals')
    plt.title(f'Number of Signals out of {num_features} Falsely Classified')
    plt.legend()
    plt.savefig(f'results/n_signal_fa_{num_features}.png')


def roc(num_features=10):
    print(f'*** {num_features} Features ***')
    if not os.path.exists('results/roc'):
        os.mkdir('results/roc')
    if not os.path.exists(f'results/roc/{num_features}'):
        os.mkdir(f'results/roc/{num_features}')

    healthy_filepath = 'data/healthy.csv'
    healthy = np.genfromtxt(healthy_filepath, delimiter=',')[:, :num_features]

    beams = ['a6', 'a17', 'a18']
    damages = [f'd{i}' for i in range(10, 91, 10)]

    train_ratio = 0.6
    val_ratio = 0.2

    train_idx = int(train_ratio * healthy.shape[0])
    val_idx = int((train_ratio + val_ratio) * healthy.shape[0])

    train = healthy[:train_idx, :]
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    train_n = (train - mean) / std

    val_healthy = healthy[train_idx:val_idx, :]
    test_healthy = healthy[val_idx:, :]

    val_healthy_n = (val_healthy - mean) / std
    test_healthy_n = (test_healthy - mean) / std

    n_pc_range = range(1, num_features)
    fprs1, fnrs1, fprs3, fnrs3 = [], [], [], []
    for n_pc in n_pc_range:
        print(f'{n_pc} PCs:')
        pca = PCA(n_components=n_pc, svd_solver='full')
        pca.fit(train_n)

        val_healthy_recons_n = pca.inverse_transform(pca.transform(val_healthy_n))
        val_healthy_recons = val_healthy_recons_n * std + mean

        residual_val_healthy = val_healthy_recons - val_healthy

        residual_val_healthy_means = np.mean(residual_val_healthy, axis=0)
        residual_val_healthy_stds = np.std(residual_val_healthy, axis=0)

        q_val_healthy = q_stat(val_healthy_recons_n - val_healthy_n)

        val_defective = []
        for beam in beams:
            for i, damage in enumerate(damages):
                defective_filepath = f'data/{beam}/{beam}_{damage}.csv'
                defective = np.genfromtxt(defective_filepath, delimiter=',')[:, :num_features]
                if i % 2 == 0:
                    val_defective.append(defective)

        val_defective = np.concatenate(val_defective, axis=0)
        val_defective_n = (val_defective - mean) / std
        val_defective_recons_n = pca.inverse_transform(pca.transform(val_defective_n))
        val_defective_recons = val_defective_recons_n * std + mean
        residual_val_defective = val_defective_recons - val_defective
        q_val_defective = q_stat(val_defective_recons_n - val_defective_n)

        # Method 1:
        print('Method 1:')
        threshold_ratios = [i / 40 for i in range(1, 401)]
        print(f'Threshold ratios: {threshold_ratios}')
        fpr, fnr = [], []
        for thr in threshold_ratios:
            residual_lower_limit = residual_val_healthy_means - thr * residual_val_healthy_stds
            residual_higher_limit = residual_val_healthy_means + thr * residual_val_healthy_stds
            y_val_healthy = np.zeros(residual_val_healthy.shape)  # 0 = Healthy (negative), 1 = Failed (positive)
            y_val_defective = np.zeros(residual_val_defective.shape)
            for j in range(residual_val_healthy.shape[1]):
                for i in range(residual_val_healthy.shape[0]):
                    if not residual_lower_limit[j] <= residual_val_healthy[i, j] <= residual_higher_limit[j]:
                        y_val_healthy[i, j] = 1
            for j in range(residual_val_defective.shape[1]):
                for i in range(residual_val_defective.shape[0]):
                    if not residual_lower_limit[j] <= residual_val_defective[i, j] <= residual_higher_limit[j]:
                        y_val_defective[i, j] = 1
            y_val_healthy = np.max(y_val_healthy, axis=1)
            y_val_defective = np.max(y_val_defective, axis=1)
            fpr.append(np.count_nonzero(y_val_healthy) / y_val_healthy.shape[0])
            fnr.append(1 - np.count_nonzero(y_val_defective) / y_val_defective.shape[0])
        fprs1.append(fpr)
        fnrs1.append(fnr)
        print(f'fpr1: {fpr}')
        print(f'fnr1: {fnr}')

        # Method 3:
        print('Method 3:')
        conf_range = range(800, 1000)
        print(f'Confidence intervals: {list(conf_range)}')
        fpr, fnr = [], []
        for conf in conf_range:
            # Q-statistic threshold
            c_alpha = stats.norm.ppf(conf / 1000)  # normal deviate corresponding to "conf / 1000" percentile
            cov = np.cov(train_n.T)
            d, _ = np.linalg.eig(cov)
            d = np.sort(d)[::-1]
            theta = []
            for i in range(3):
                theta.append(np.sum(np.array(d[n_pc:]) ** (i + 1)))
            h_0 = 1 - (2 * theta[0] * theta[2]) / (3 * theta[1] ** 2)
            q_alpha = theta[0] * np.power((c_alpha * h_0 * np.sqrt(
                2 * theta[1]) / theta[0] + 1 + theta[1] * h_0 * (h_0 - 1) / theta[0] ** 2), 1 / h_0)

            y_val_healthy = np.zeros(q_val_healthy.shape)  # 0 = Healthy (negative), 1 = Failed (positive)
            y_val_defective = np.zeros(q_val_defective.shape)

            for i in range(q_val_healthy.shape[0]):
                if not q_val_healthy[i] <= q_alpha:
                    y_val_healthy[i] = 1
            for i in range(q_val_defective.shape[0]):
                if not q_val_defective[i] <= q_alpha:
                    y_val_defective[i] = 1

            fpr.append(np.count_nonzero(y_val_healthy) / y_val_healthy.shape[0])
            fnr.append(1 - np.count_nonzero(y_val_defective) / y_val_defective.shape[0])
        fprs3.append(fpr)
        fnrs3.append(fnr)
        print(f'fpr3: {fpr}')
        print(f'fnr3: {fnr}')

    for fpr, fnr, n in zip(fprs1, fnrs1, n_pc_range):
        plt.figure()
        plt.scatter(fpr, fnr)
        plt.xlabel('False positive rate')
        plt.ylabel('False negative rate')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.title(f'Method 1: ROC Curve for Validation Set ({num_features} Features) - {n} PCs')
        plt.savefig(f'results/roc/{num_features}/m1_{n}_pcs.png')

    for fpr, fnr, n in zip(fprs3, fnrs3, n_pc_range):
        plt.figure()
        plt.scatter(fpr, fnr)
        plt.xlabel('False positive rate')
        plt.ylabel('False negative rate')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.title(f'Method 3: ROC Curve for Validation Set ({num_features} Features) - {n} PCs')
        plt.savefig(f'results/roc/{num_features}/m3_{n}_pcs.png')


def test(num_features=6):
    healthy_filepath = 'data/healthy.csv'
    healthy = np.genfromtxt(healthy_filepath, delimiter=',')[:, :num_features]

    beams = ['a6', 'a17', 'a18']
    damages = [f'd{i}' for i in range(10, 91, 10)]

    train_ratio = 0.6
    val_ratio = 0.2

    train_idx = int(train_ratio * healthy.shape[0])
    val_idx = int((train_ratio + val_ratio) * healthy.shape[0])

    train = healthy[:train_idx, :]
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    train_n = (train - mean) / std

    val_healthy = healthy[train_idx:val_idx, :]
    test_healthy = healthy[val_idx:, :]

    val_healthy_n = (val_healthy - mean) / std
    test_healthy_n = (test_healthy - mean) / std

    val_defective, test_defective = [], []
    for beam in beams:
        for i, damage in enumerate(damages):
            defective_filepath = f'data/{beam}/{beam}_{damage}.csv'
            defective = np.genfromtxt(defective_filepath, delimiter=',')[:, :num_features]
            if i % 2 == 0:
                val_defective.append(defective)
            else:
                test_defective.append(defective)

    test_defective = np.concatenate(test_defective, axis=0)
    test_defective_n = (test_defective - mean) / std

    # Model 1
    n_pc = 2
    pca = PCA(n_components=n_pc, svd_solver='full')
    pca.fit(train_n)

    val_healthy_recons_n = pca.inverse_transform(pca.transform(val_healthy_n))
    val_healthy_recons = val_healthy_recons_n * std + mean

    residual_val_healthy = val_healthy_recons - val_healthy

    residual_val_healthy_means = np.mean(residual_val_healthy, axis=0)
    residual_val_healthy_stds = np.std(residual_val_healthy, axis=0)

    test_healthy_recons_n = pca.inverse_transform(pca.transform(test_healthy_n))
    test_healthy_recons = test_healthy_recons_n * std + mean

    residual_test_healthy = test_healthy_recons - test_healthy

    test_defective_recons_n = pca.inverse_transform(pca.transform(test_defective_n))
    test_defective_recons = test_defective_recons_n * std + mean
    residual_test_defective = test_defective_recons - test_defective

    thr = 2.15
    residual_lower_limit = residual_val_healthy_means - thr * residual_val_healthy_stds
    residual_higher_limit = residual_val_healthy_means + thr * residual_val_healthy_stds
    y_test_healthy = np.zeros(residual_test_healthy.shape)  # 0 = Healthy (negative), 1 = Failed (positive)
    y_test_defective = np.zeros(residual_test_defective.shape)
    for j in range(residual_test_healthy.shape[1]):
        for i in range(residual_test_healthy.shape[0]):
            if not residual_lower_limit[j] <= residual_test_healthy[i, j] <= residual_higher_limit[j]:
                y_test_healthy[i, j] = 1
    for j in range(residual_test_defective.shape[1]):
        for i in range(residual_test_defective.shape[0]):
            if not residual_lower_limit[j] <= residual_test_defective[i, j] <= residual_higher_limit[j]:
                y_test_defective[i, j] = 1
    y_test_healthy = np.max(y_test_healthy, axis=1)
    y_test_defective = np.max(y_test_defective, axis=1)
    fpr1 = np.count_nonzero(y_test_healthy) / y_test_healthy.shape[0]
    fnr1 = 1 - np.count_nonzero(y_test_defective) / y_test_defective.shape[0]
    print(f'Method 1: FPR = {fpr1}, FNR = {fnr1}')

    # Model 3
    n_pc = 1
    pca = PCA(n_components=n_pc, svd_solver='full')
    pca.fit(train_n)

    test_healthy_recons_n = pca.inverse_transform(pca.transform(test_healthy_n))
    q_test_healthy = q_stat(test_healthy_recons_n - test_healthy_n)

    test_defective_recons_n = pca.inverse_transform(pca.transform(test_defective_n))
    q_test_defective = q_stat(test_defective_recons_n - test_defective_n)

    conf = 914
    # Q-statistic threshold
    c_alpha = stats.norm.ppf(conf / 1000)  # normal deviate corresponding to 95 percentile
    cov = np.cov(train_n.T)
    d, _ = np.linalg.eig(cov)
    d = np.sort(d)[::-1]
    theta = []
    for i in range(3):
        theta.append(np.sum(np.array(d[n_pc:]) ** (i + 1)))
    h_0 = 1 - (2 * theta[0] * theta[2]) / (3 * theta[1] ** 2)
    q_alpha = theta[0] * np.power((c_alpha * h_0 * np.sqrt(
        2 * theta[1]) / theta[0] + 1 + theta[1] * h_0 * (h_0 - 1) / theta[0] ** 2), 1 / h_0)

    y_test_healthy = np.zeros(q_test_healthy.shape)  # 0 = Healthy (negative), 1 = Failed (positive)
    y_test_defective = np.zeros(q_test_defective.shape)

    for i in range(q_test_healthy.shape[0]):
        if not q_test_healthy[i] <= q_alpha:
            y_test_healthy[i] = 1
    for i in range(q_test_defective.shape[0]):
        if not q_test_defective[i] <= q_alpha:
            y_test_defective[i] = 1

    fpr3 = np.count_nonzero(y_test_healthy) / y_test_healthy.shape[0]
    fnr3 = 1 - np.count_nonzero(y_test_defective) / y_test_defective.shape[0]
    print(f'Method 3: FPR = {fpr3}, FNR = {fnr3}')


def dim_analysis(num_features=6):
    healthy_filepath = 'data/healthy.csv'
    healthy = np.genfromtxt(healthy_filepath, delimiter=',')[:, :num_features]

    beams = ['a6', 'a17', 'a18']
    damages = [f'd{i}' for i in range(10, 91, 10)]

    train_ratio = 0.6
    val_ratio = 0.2

    train_idx = int(train_ratio * healthy.shape[0])
    val_idx = int((train_ratio + val_ratio) * healthy.shape[0])

    train = healthy[:train_idx, :]
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    train_n = (train - mean) / std

    n_pc = 1
    pca = PCA(n_components=n_pc, svd_solver='full')
    pca.fit(train_n)

    dims_mean, dims_std = [], []
    for beam in beams:
        # defective data for "beam"
        q_val_defective, val_defective_n = [], []
        for damage in damages:
            defective_filepath = f'data/{beam}/{beam}_{damage}.csv'
            defective = np.genfromtxt(defective_filepath, delimiter=',')[:, :num_features]

            val_defective = defective[train_idx:val_idx, :]
            val_defective_n.append((val_defective - mean) / std)

            val_defective_recons_n = pca.inverse_transform(pca.transform((val_defective - mean) / std))

            q_val_defective.append(q_stat(val_defective_recons_n - (val_defective - mean) / std))

        q_val_defective = np.concatenate(q_val_defective, axis=0)
        val_defective_n = np.concatenate(val_defective_n, axis=0)

        # DIM analysis
        q_0 = q_val_defective

        delta_qs = []
        for f in range(val_defective_n.shape[1]):
            min_r = np.min(val_defective_n[:, f])
            max_r = np.max(val_defective_n[:, f])
            increment = (max_r - min_r) / 100
            val_defective_plus_n = val_defective_n.copy()
            val_defective_minus_n = val_defective_n.copy()
            for i in range(val_defective_n.shape[0]):
                val_defective_plus_n[i, f] += increment
                val_defective_minus_n[i, f] -= increment
            val_defective_plus_n_recons = pca.inverse_transform(pca.transform(val_defective_plus_n))
            val_defective_minus_n_recons = pca.inverse_transform(pca.transform(val_defective_minus_n))
            residual_val_defective_plus_n = val_defective_plus_n_recons - val_defective_plus_n
            residual_val_defective_minus_n = val_defective_minus_n_recons - val_defective_minus_n
            q_plus = q_stat(residual_val_defective_plus_n)
            q_minus = q_stat(residual_val_defective_minus_n)
            q = np.maximum(q_plus, q_minus)
            delta_q = q - q_0
            delta_qs.append(delta_q)
        delta_qs = np.array(delta_qs).T
        delta_qs_sum = np.sum(delta_qs, axis=1)
        dim = []
        for i in range(delta_qs.shape[0]):
            dim.append(delta_qs[i, :] / delta_qs_sum[i])
        dim = np.array(dim)

        dims_mean.append(np.mean(dim, axis=0))
        dims_std.append(np.std(dim, axis=0))

    for dim_mean, dim_std, beam in zip(dims_mean, dims_std, beams):
        plt.figure()
        plt.errorbar(range(dim_mean.shape[0]), dim_mean, dim_std)
        plt.xlabel('Feature')
        plt.ylabel('DIM')
        plt.title(f'Mean and Std of DIM for each Feature - Beam {beam.upper()}')
        plt.savefig(f'results/dim_{beam}.png')


def dim_healthy(num_features=6):
    healthy_filepath = 'data/healthy.csv'
    healthy = np.genfromtxt(healthy_filepath, delimiter=',')[:, :num_features]

    train_ratio = 0.6
    val_ratio = 0.2

    train_idx = int(train_ratio * healthy.shape[0])

    train = healthy[:train_idx, :]
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    train_n = (train - mean) / std

    val = healthy[train_idx:, :]
    val_n = (val - mean) / std

    for n_pc in range(1, num_features):
        pca = PCA(n_components=n_pc, svd_solver='full')
        pca.fit(train_n)

        val_recons_n = pca.inverse_transform(pca.transform(val_n))
        q_val = q_stat(val_recons_n - val_n)

        # DIM analysis
        q_0 = q_val
        delta_qs = []
        for f in range(val_n.shape[1]):
            min_r = np.min(val_n[:, f])
            max_r = np.max(val_n[:, f])
            increment = (max_r - min_r) / 100
            val_plus_n = val_n.copy()
            val_minus_n = val_n.copy()
            for i in range(val_n.shape[0]):
                val_plus_n[i, f] += increment
                val_minus_n[i, f] -= increment
            val_plus_n_recons = pca.inverse_transform(pca.transform(val_plus_n))
            val_minus_n_recons = pca.inverse_transform(pca.transform(val_minus_n))
            residual_val_plus_n = val_plus_n_recons - val_plus_n
            residual_val_minus_n = val_minus_n_recons - val_minus_n
            q_plus = q_stat(residual_val_plus_n)
            q_minus = q_stat(residual_val_minus_n)
            q = np.maximum(q_plus, q_minus)
            delta_q = q - q_0
            delta_qs.append(delta_q)
        delta_qs = np.array(delta_qs).T
        delta_qs_sum = np.sum(delta_qs, axis=1)
        dim = []
        for i in range(delta_qs.shape[0]):
            dim.append(delta_qs[i, :] / delta_qs_sum[i])
        dim = np.array(dim)

        dims_mean = np.mean(dim, axis=0)
        dims_std = np.std(dim, axis=0)

        plt.figure()
        plt.errorbar(range(dims_mean.shape[0]), dims_mean, dims_std)
        plt.xlabel('Feature')
        plt.ylabel('DIM')
        plt.title(f'Mean and Std of DIM for each Feature - {n_pc} PCs')
        plt.savefig(f'results/dim_healthy_{n_pc}_pcs.png')


if __name__ == '__main__':
    # plot_feature_1_dist()
    # roc_damage_20_only_feature_1()
    # only_feature_1()
    # far_analysis(num_features=10)
    # far_analysis(num_features=6)
    # mar_analysis(num_features=10)
    # mar_analysis(num_features=6)
    # main(num_features=10)
    # main(num_features=6)
    # roc(num_features=10)
    # roc(num_features=6)
    # dim_analysis(num_features=6)
    # test()
    # mar_vs_damage(num_features=6)
    # get_pca_components(num_features=6)
    # roc_damage_20_m3(num_features=6)
    dim_healthy()
