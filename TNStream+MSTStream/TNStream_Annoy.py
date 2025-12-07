"""
TNStream with Annoy - 基于原始LSH-TNStream改进
完全兼容原始接口，用Annoy替代LSH
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from sklearn import metrics
from tqdm import tqdm
import time
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import os
from datetime import datetime

try:
    from annoy import AnnoyIndex
    ANNOY_AVAILABLE = True
except ImportError:
    print("警告: Annoy未安装，使用KDTree")
    ANNOY_AVAILABLE = False


class AnnoyIndexWrapper:
    """Annoy索引封装 - 兼容LSH接口"""
    def __init__(self, X, num_hashes=5):
        self.X = X
        self.N = X.shape[0]
        self.dim = X.shape[1]
        self.num_hashes = num_hashes

        if ANNOY_AVAILABLE:
            self.index = AnnoyIndex(self.dim, metric='angular')
            for i, x in enumerate(X):
                self.index.add_item(i, x.astype('float32'))
            self.index.build(10)
            self.use_annoy = True
        else:
            self.tree = KDTree(X)
            self.use_annoy = False

    def query(self, x, k=-1):
        """兼容LSH的query接口"""
        if self.use_annoy:
            indices, distances = self.index.get_nns_by_vector(
                x.astype('float32'), min(k+1 if k > 0 else 10, self.N),
                search_k=-1, include_distances=True
            )
            # 移除自身（第一个）
            if len(indices) > 1:
                candidates = [(indices[i], distances[i]) for i in range(1, len(indices))]
            else:
                candidates = [(indices[0], distances[0])] if indices else []
        else:
            distances, indices = self.tree.query(x, k=min(k+1 if k > 0 else 10, self.N))
            if k == -1:
                candidates = [(indices[i], distances[i]) for i in range(1, len(indices))]
            else:
                candidates = [(indices[i], distances[i]) for i in range(len(indices))]

        if k == -1:
            if len(candidates) < 2:
                return "Not Found"
            else:
                return candidates[0]  # 返回最近的一个
        else:
            return candidates[:k]

    def query_radius(self, x, r=100):
        """范围查询"""
        if self.use_annoy:
            k = min(200, max(50, self.N // 10))
            indices, distances = self.index.get_nns_by_vector(
                x.astype('float32'), k, search_k=-1, include_distances=True
            )
            result = [idx for idx, dist in zip(indices, distances) if dist < r]
            return result
        else:
            return self.tree.query_ball_point(x, r)


class TNStream:
    """TNStream with Annoy - 完全兼容原始接口"""

    def __init__(self, X, labels_true, N, W, r, n_micro, d, plotFigure, k, mk):
        """
        Args:
            X: 数据
            labels_true: 真实标签
            N: 微簇最小点数
            W: 滑动窗口大小
            r: 微簇半径
            n_micro: 宏簇最小微簇数
            d: 数据维度
            plotFigure: 是否绘图
            k: k近邻数
            mk: 共享近邻数
        """
        self.X = X
        self.labels_true = labels_true
        self.k = k
        self.N = N
        self.W = W
        self.r = r
        self.n_micro = n_micro
        self.plotFigure = plotFigure
        self.mk = mk
        self.d = d

        self.MC_Num = 0
        self.MacroC_Num = 0
        self.buffered_data = np.empty((0, d+3), float)
        self.MCs = np.empty((0, d+3), float)
        self.MCr = []
        self.MacroClusters = np.empty((0, 4), dtype=object)
        self.deleted_data = np.empty((0, d+3), float)

        self.colors = np.empty((0, 4), int)

        # 日志
        self.log_file = f"tnstream_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self._write_log(f"参数: N={N}, W={W}, r={r:.4f}, mk={mk}, k={k}, n_micro={n_micro}")

        start = time.time()
        for i in tqdm(range(len(self.X))):
            self.AddNode(self.X[i, :])
            self.DefineMC()
            self.AddtoMC()
            self.DefineMacroC()
            self.AddMCtoMacroC()
            self.UpdateInfo()
            self.UpdateMacroCs()
            self.KillMCs()
            self.KillMacroC()

            if i % 2000 == 0 and i != 0:
                print(f"No: {i}")
                self.evaluate()
                if self.plotFigure == 1 and self.d == 2:
                    self.plotGraph("Annoy-TNStream")

            end = time.time()
            if end - start > 60 * 25:
                print("Time Limit Exceeded")
                break

    def AddNode(self, X):
        """添加新数据点"""
        if self.buffered_data.shape[0] == 0:
            self.buffered_data = np.vstack((self.buffered_data,
                                           np.hstack((np.array([1, 0, 0]), X))))
        else:
            idx = self.buffered_data[self.buffered_data.shape[0]-1, 0] + 1
            self.buffered_data = np.vstack((self.buffered_data,
                                           np.hstack((np.array([idx, 0, 0]), X))))

        # 滑动窗口
        if self.buffered_data.shape[0] > self.W:
            split_point = self.buffered_data.shape[0] - self.W
            self.deleted_data = np.vstack((self.deleted_data, self.buffered_data[:split_point]))
            self.buffered_data = self.buffered_data[split_point:]

    def DefineMC(self):
        """定义微簇"""
        X = self.buffered_data[self.buffered_data[:, 1] == 0, :]
        tot = X.shape[0]

        if tot < self.N:
            return

        datatmp = X[:, 3:]
        tree = AnnoyIndexWrapper(datatmp)

        for i in range(X.shape[0]):
            knn_indx = tree.query_radius(X[i, 3:], r=self.r)

            if len(knn_indx) < self.N:
                continue

            points = []
            maxr = 0

            for j in knn_indx:
                if j >= tot:
                    break

                knn_indy = tree.query_radius(X[j, 3:], r=self.r)
                snn = len(set(knn_indx) & set(knn_indy))

                if snn >= self.mk:
                    maxr = max(maxr, np.linalg.norm(X[j, 3:] - X[i, 3:]))

            maxr = min(maxr, self.r)
            ind1 = tree.query_radius(X[i, 3:], r=maxr)
            points = [datatmp[l][:] for l in ind1]

            if len(points) >= self.N:
                center = np.mean(np.array(points), axis=0)
                self.MC_Num += 1
                self.MCr.append(maxr)
                self.MCs = np.vstack((self.MCs,
                                     np.hstack((np.array([self.MC_Num, len(points), 0]), center))))

                for j in range(len(points)):
                    mask = (self.buffered_data[:, 3:] == points[j]).all(axis=1)
                    self.buffered_data[mask, 1] = self.MC_Num
                return

    def AddtoMC(self):
        """将孤立点分配到微簇"""
        if self.MCs.shape[0] <= 1:
            return

        for i in range(self.buffered_data.shape[0]):
            if self.buffered_data[i, 1] == 0:
                tree = KDTree(self.MCs[:, 3:])
                d, ind = tree.query(self.buffered_data[i, 3:])
                if d <= self.MCr[ind]:
                    self.buffered_data[i, 1] = self.MCs[ind, 0]

    def UpdateInfo(self):
        """更新微簇信息"""
        for i in range(self.MCs.shape[0]):
            self.MCs[i, 1] = self.buffered_data[self.buffered_data[:, 1] == self.MCs[i, 0], :].shape[0]
            if self.MCs[i, 1] > 0:
                self.MCs[i, 3:] = np.mean(
                    self.buffered_data[self.buffered_data[:, 1] == self.MCs[i, 0], 3:],
                    axis=0)
            if self.MCs[i, 2] >= 0:
                self.buffered_data[self.buffered_data[:, 1] == self.MCs[i, 0], 2] = self.MCs[i, 2]

    def KillMCs(self):
        """删除空微簇"""
        for i in range(self.MCs.shape[0]):
            if int(self.MCs[i, 1]) == 0:
                if int(self.MCs[i, 2]) != 0:
                    MacroCluster = self.MCs[i, 2]
                    self.buffered_data[self.buffered_data[:, 1] == self.MCs[i, 0], 1] = 0
                    self.MCs = np.delete(self.MCs, i, axis=0)
                    self.MCr.pop(i)
                    self.UpdateMacroC(MacroCluster)
                else:
                    self.buffered_data[self.buffered_data[:, 1] == self.MCs[i, 0], 1] = 0
                    self.MCs = np.delete(self.MCs, i, axis=0)
                    self.MCr.pop(i)
                return

    def UpdateMacroCs(self):
        """更新宏簇信息"""
        for i in range(self.MacroClusters.shape[0]):
            self.UpdateMacroC(self.MacroClusters[i, 0])
            self.MacroClusters[i, 1] = len(np.unique(self.MacroClusters[i, 2]))

    def UpdateMacroC(self, macroCluster):
        """更新单个宏簇"""
        if macroCluster == 0:
            return

        P = self.MCs[self.MCs[:, 2] == macroCluster, :]
        if P.shape[0] == 0:
            return

        self.MCs[self.MCs[:, 2] == macroCluster, 2] = 0
        k = self.k
        TN = self.search_TN(P[:, 3:], k)
        components = self.clustering(P[:, 3:], k, TN)

        if len(components) > 0:
            largest_component = max(components, key=len)
            mc_ids = P[list(largest_component), 0].astype(int)

            self.MacroClusters[self.MacroClusters[:, 0] == macroCluster, 2] = [mc_ids]
            self.MacroClusters[self.MacroClusters[:, 0] == macroCluster, 1] = len(mc_ids)

            for x in list(largest_component):
                j = int(P[x, 0])
                self.MCs[self.MCs[:, 0] == j, 2] = macroCluster

    def DefineMacroC(self):
        """定义宏簇"""
        unassigned_mcs = self.MCs[self.MCs[:, 2] == 0, :]
        num_mcs = len(unassigned_mcs)

        if num_mcs >= self.n_micro:
            mc_centers = unassigned_mcs[:, 3:]
            k = self.k
            TN = self.search_TN(mc_centers, k)
            components = self.clustering(mc_centers, k, TN)

            for component in components:
                if len(component) >= self.n_micro:
                    self.MacroC_Num += 1
                    self.colors = np.array([plt.cm.Spectral(each)
                                           for each in np.linspace(0, 1, self.MacroC_Num + 1)])
                    color = self.colors[-2, :]

                    mc_ids = unassigned_mcs[list(component), 0].astype(int)

                    self.MacroClusters = np.vstack((self.MacroClusters,
                                                   np.array([self.MacroC_Num, len(mc_ids), mc_ids, color],
                                                           dtype=object)))
                    print(f"----------Macro Cluster #{self.MacroC_Num} is defined----------")

                    for mc_id in mc_ids:
                        self.MCs[self.MCs[:, 0] == mc_id, 2] = self.MacroC_Num

    def search_TN(self, X, k):
        """搜索TN-graph"""
        N = X.shape[0]
        XIndex = [None] * N
        KNN = [[] for _ in range(N)]
        RKNN = [[] for _ in range(N)]
        TN = [[[] for _ in range(k)] for _ in range(N)]

        tree = KDTree(X)
        for i in range(N):
            d1, ind1 = tree.query(X[i, :], k=k+1)
            XIndex[i] = ind1

        for r in range(k):
            for j in range(N):
                if r + 1 < len(XIndex[j]) and XIndex[j][r+1] < N:
                    y = XIndex[j][r + 1]
                    KNN[j] = KNN[j] + [y] if KNN[j] is not None else [y]
                    RKNN[y] = RKNN[y] + [j] if RKNN[y] is not None else [j]

            for i in range(N):
                if len(KNN[i]) > 0 and len(RKNN[i]) > 0:
                    TN[i][r] = list(set(KNN[i]) & set(RKNN[i]))

        return TN

    def clustering(self, X, k, TN):
        """基于TN-graph的聚类"""
        N = X.shape[0]
        s = [N - 1]
        t = [N - 1]

        for i in range(N):
            if TN[i] is None or len(TN[i]) <= 1 or TN[i][k-1] is None:
                continue

            tn = TN[i][k-1]
            if len(tn) > 0:
                idx = np.where(np.array(tn) > i)[0]
                tn = np.array(tn)[idx].tolist()
                p = len(tn)
                if p >= 1:
                    t.extend(tn)
                    s.extend([i] * p)

        G0 = nx.Graph()
        G0.add_edges_from(zip(s, t))
        G0.remove_edge(N - 1, N - 1)

        connected_components = list(nx.connected_components(G0))
        return connected_components

    def AddMCtoMacroC(self):
        """将微簇分配到宏簇"""
        if self.MacroClusters.shape[0] != 0:
            for i in range(self.MCs.shape[0]):
                if self.MCs[i, 2] == 0 and self.MCs[i, 1] >= self.N:
                    A = self.MCs[self.MCs[:, 2] != 0, :]
                    if A.shape[0] > 0:
                        tree = AnnoyIndexWrapper(A[:, 3:])
                        res = tree.query(self.MCs[i, 3:])
                        if res == "Not Found":
                            continue

                        ind = res[0]
                        self.MCs[i, 2] = A[ind, 2]
                        MacroC = int(A[ind, 2])
                        self.UpdateMacroC(self.MCs[i, 2])
                        return

    def KillMacroC(self):
        """删除不合格的宏簇"""
        for i in range(self.MacroClusters.shape[0]):
            summ = 0
            edges = self.MacroClusters[i, 2]
            for e in edges:
                summ = summ + self.MCs[self.MCs[:, 0] == e, 1]

            if summ < self.n_micro * self.N and len(edges) < self.n_micro:
                self.MCs[self.MCs[:, 2] == self.MacroClusters[i, 0], 2] = 0
                print(f"----------Macro Cluster #{self.MacroClusters[i, 0]} is killed----------")
                self.MacroClusters = np.delete(self.MacroClusters, i, axis=0)
                return

    def plotGraph(self, title, dpi=70):
        """绘制2D图"""
        ax = plt.gca()
        ax.cla()
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams["figure.figsize"] = (4, 4)

        for i in range(len(self.buffered_data)):
            plt.plot(self.buffered_data[i, 3], self.buffered_data[i, 4], 'bo',
                    markeredgecolor='k', alpha=.15, markersize=5)

        for i in range(len(self.MCs)):
            if self.MCs[i, 2] == 0:
                col = (0, 0, 1, 1)
            else:
                col = self.MacroClusters[self.MacroClusters[:, 0] == self.MCs[i, 2], 3][0].tolist()

            plt.plot(self.MCs[i, 3], self.MCs[i, 4], 'rd', markeredgecolor='k', markersize=1)
            circle1 = plt.Circle((self.MCs[i, 3], self.MCs[i, 4]), self.MCr[i],
                                color=col, clip_on=False, fill=False)
            ax.add_patch(circle1)

        plt.title(title)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.show()

    def purity_score(self, y_true, y_pred):
        """计算Purity"""
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    def evaluate(self):
        """评估聚类结果"""
        labels = np.hstack((self.deleted_data[:, 2], self.buffered_data[:, 2]))
        N = labels.shape[0]
        t_labels = self.labels_true.reshape(-1)[:N]

        ARI = adjusted_rand_score(t_labels, labels)
        Purity = self.purity_score(t_labels, labels)
        NMI = normalized_mutual_info_score(t_labels, labels)

        print("****** Temp Results ******")
        print(f"Purity = {Purity:.4f}")
        print(f"ARI = {ARI:.4f}")
        print(f"NMI = {NMI:.4f}")

        self._write_log(f"Purity={Purity:.4f}, ARI={ARI:.4f}, NMI={NMI:.4f}")

        return {'ARI': ARI, 'Purity': Purity, 'NMI': NMI}

    def _write_log(self, msg):
        """写入日志"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')


# 使用示例
if __name__ == "__main__":
    # model = TNStream(X, labels_true, N=5, W=1000, r=0.1, n_micro=3, d=X.shape[1],
    #                  plotFigure=0, k=5, mk=3)
    pass