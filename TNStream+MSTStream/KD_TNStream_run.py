import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import adjusted_rand_score
from KD_TNStream import TNStream  # import class MCMSTStream from the same directory
import time
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score, calinski_harabasz_score
from scipy.spatial.distance import euclidean
from scipy.stats import ks_2samp
import random

# 使用你提供的正确文件路径
dataset = np.loadtxt("/Users/tsiphenzeng/Desktop/TNStream/dataset/dataset2/kdd_converted.txt", dtype=float,
                     delimiter=' ')
X = dataset[:, :-1]
labels_true = dataset[:, -1]

# 数据标准化移到循环外面，避免重复标准化
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

T = 0
while T < 1000:
    T += 1
    dataset_name = "kdd-converted"
    W = random.randint(3000, 5000)  # 4000
    N = random.randint(2, 3)
    r = random.randint(400, 800) / 10000  # 0.6642
    n_micro = random.randint(6, 10)
    mk = random.randint(3, 5)

    plotFigure = 0  # 设为 1 则每加入100个点绘制一次图像
    start = time.time()
    kds = TNStream(X_normalized, labels_true, N, W, r, n_micro, X_normalized.shape[1], plotFigure, 4, mk, mk - 1)
    end = time.time()

    labels = np.hstack((kds.deleted_data[:, 2], kds.buffered_data[:, 2]))
    ARI = adjusted_rand_score(labels_true.reshape(-1), labels)
    Purity = kds.purity_score(labels_true.reshape(-1), labels)
    NMI = normalized_mutual_info_score(labels_true.reshape(-1), labels)

    if ARI >= 0.9:
        print("Current Dataset:", dataset_name)
        print("Current Iteration:", T)
        print("当前参数：W=", W, "N=", N, "r=", r, "n_micro=", n_micro, "mk=", mk)
        print("\n##### The Best Results #############")
        print("Purity=", Purity)
        print("ARI=", ARI)
        print("NMI =", NMI)
        print("运行时间:", end - start, "秒")
        time.sleep(2)

        # 可视化部分
        if X_normalized.shape[1] == 2:
            kds.plotGraph(str("KD-TNStream"))
        elif X_normalized.shape[1] == 3:
            kds.plotGraph3D_normal(str("KD-TNStream"))