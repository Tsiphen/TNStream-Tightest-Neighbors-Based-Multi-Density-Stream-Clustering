import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from sklearn import metrics
from scipy.spatial.distance import cdist
from tqdm import tqdm


# Defining class KDTNStream
class TNStream:
    N = int()
    W = int()
    r = float()
    d = int()
    MC_Num = int
    MacroC_Num = int
    colors = np.empty((0, 4), int)

    def __init__(self, X, N, W, r, n_micro, d, plotFigure, k, tk, mk):
        # algorithm parameters###########################################
        self.X = X
        self.k = k
        self.N = N  # minimum number of data to define a MC
        self.W = W  # sliding window size
        self.r = r  # radius of MC
        self.tk = tk
        self.mk = mk
        self.n_micro = n_micro  # minimum number of MC to define a Macro Cluster
        self.plotFigure = plotFigure
        ##################################################################
        self.d = d
        self.MC_Num = 0
        self.MacroC_Num = 0
        self.buffered_data = np.empty((0, d + 3), float)  # [index | MC No | isActive | features={d1,d2,d3...}]
        self.MCs = np.empty((0, d + 3), float)  # [MC No | #of data it has | centerCoordinates={d1,d2,d3,...}]
        self.MCr = []
        self.MacroClusters = np.empty((0, 4), dtype=object)  # [MacroClusterNo | #of data it has | isActive ]
        self.deleted_data = np.empty((0, d + 3), float)  # [index | features={d1,d2,d3...} | predictedclusterLabel]
        self.color_dict = {}
        self.buildcolorDict()
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
            if (i % 2000 == 0):
                print("No:", i)
                if self.plotFigure == 1 and self.d == 2:
                    self.plotGraph("KD-TNstream")
    def buildcolorDict(self):
        colors = plt.cm.get_cmap('Set2', 8)  # 使用 HSV 色图
        colors2 = plt.cm.get_cmap('tab10', 10)
        colors3 = plt.cm.get_cmap('Set1', 10)
        colors4 = plt.cm.get_cmap('Set3', 10)
        # 创建颜色字典
        for i in range(9):
            if i == 0:  # 离群点标签为0，设置为深蓝色
                self.color_dict[i] = 'darkblue'
            else:
                self.color_dict[i] = colors(i-1)
        for i in range(10):
            self.color_dict[i+9] = colors2(i)
            self.color_dict[i+19] = colors3(i)
            self.color_dict[i+29] = colors4(i)
    def AddNode(self, X):  # add new data to buffered_data, and delete old ones
        if (self.buffered_data.shape[0] == 0):
            self.buffered_data = np.vstack((self.buffered_data, np.hstack((np.array([1, 0, 0]), X))))
        else:
            self.buffered_data = np.vstack((self.buffered_data, np.hstack((np.array([self.buffered_data[self.buffered_data.shape[0] - 1, 0] + 1, 0, 0]), X))))
        if (self.buffered_data.shape[0] > self.W):
            self.deleted_data = np.vstack((self.deleted_data, np.split(self.buffered_data, [self.buffered_data.shape[0] - self.W])[0]))
            self.buffered_data = np.split(self.buffered_data, [self.buffered_data.shape[0] - self.W])[1]

    def DefineMC(self):
        X = self.buffered_data[self.buffered_data[:, 1] == 0, :]  # data that do not belong to any cluster
        tot = X.shape[0]
        if (tot >= self.N):  # if # of data that do not belong to any cluster greater than N
            datatmp = X[:, 3:].tolist()
            tree = KDTree(datatmp)  # construct kdtree
            for i in range(tot):  # for each data of tree do reangeserach
                knn_dx, knn_indx = tree.query(X[i, 3:], k=self.tk)
                maxr = 0
                for j in knn_indx:
                    if j >= tot:
                        break
                    knn_dy, knn_indy = tree.query(X[j, 3:], k=self.tk)
                    snn = len(set(knn_indx) & set(knn_indy))
                    if snn >= self.mk:
                        maxr = max(maxr, np.linalg.norm(X[j, 3:] - X[i, 3:]))
                maxr = min(maxr, self.r)
                maxr = max(maxr, self.r / 2)#xiugai
                # print("******",maxr)
                ind1 = tree.query_ball_point(X[i, 3:], r=maxr)  # rangesearch
                points = [datatmp[l][:] for l in ind1]
                if (len(points) >= self.N):
                    center = np.mean(np.array(points), axis=0)  # calculate the center of candidate MC
                    # if self.flag<=100:
                    #     print(center)
                    #     self.flag+=1
                    kdtree2 = KDTree(self.MCs[self.MCs[:, 2] != 0, 3:])
                    d2, ind2 = kdtree2.query(center)
                    if (d2 > self.r * 0.75):
                        self.MC_Num = self.MC_Num + 1
                        self.MCr.append(maxr)
                        #print("MC number ",self.MC_Num," is defined")
                        self.MCs = np.vstack(
                            (self.MCs, np.hstack((np.array([self.MC_Num, len(points), 0]), center))))  # define new MC
                        for j in range(len(points)):
                            self.buffered_data[np.where((self.buffered_data[:, 3:] == points[j]).all(axis=1))[
                                0], 1] = self.MC_Num  # assign data to new MC
                        return

    def AddtoMC(self):
        if (self.MCs.shape[0] > 1):
            for i in range(self.buffered_data.shape[0]):
                if (self.buffered_data[i, 1] == 0):
                    kdtree = KDTree(self.MCs[:, 3:])
                    d, ind = kdtree.query(self.buffered_data[i, 3:])
                    if (d <= self.MCr[ind]):
                        self.buffered_data[i, 1] = self.MCs[ind, 0]

    def UpdateInfo(self):
        for i in range(self.MCs.shape[0]):
            self.MCs[i, 1] = self.buffered_data[self.buffered_data[:, 1] == self.MCs[i, 0], :].shape[0]
            if (self.MCs[i, 1] > 0):
                self.MCs[i, 3:] = np.mean(self.buffered_data[self.buffered_data[:, 1] == self.MCs[i, 0], 3:],
                                          axis=0)  # calculate the center of MC
            if (self.MCs[i, 2] >= 0):
                self.buffered_data[self.buffered_data[:, 1] == self.MCs[i, 0], 2] = self.MCs[i, 2]

    def KillMCs(self):
        for i in range(self.MCs.shape[0]):  # 优化
            if (int(self.MCs[i, 1]) == 0):
                print("MC number ",int(self.MCs[i,0])," is killed")
                if (int(self.MCs[i, 2]) != 0):
                    MacroCluster = self.MCs[i, 2]
                    self.buffered_data[self.buffered_data[:, 1] == self.MCs[i, 0], 1] = 0
                    self.MCs = np.delete(self.MCs, i, axis=0)
                    self.UpdateMacroC(MacroCluster)
                else:
                    self.buffered_data[self.buffered_data[:, 1] == self.MCs[i, 0], 1] = 0
                    self.MCs = np.delete(self.MCs, i, axis=0)

                return

    def UpdateMacroCs(self):
        for i in range(self.MacroClusters.shape[0]):
            self.UpdateMacroC(self.MacroClusters[i, 0])
            # print("总UP")
            self.MacroClusters[i, 1] = len(np.unique(self.MacroClusters[i, 2]))

    def UpdateMacroC(self, macroCluster):
        if macroCluster == 0:
            return
        P = self.MCs[self.MCs[:, 2] == macroCluster, :]
        if P.shape[0] == 0:
            return
        self.MCs[self.MCs[:, 2] == macroCluster, 2] = 0
        k = self.k
        TN = self.search_TN(P[:, 3:], k)
        # TNOF
        #N = len(P[:, 3:])  # 样本总数
        A = cdist(P[:, 3:], P[:, 3:])  # 计算距离矩阵

        # 获取每个点的k个紧邻
        # TN = {i: TN0[i][:k] for i in range(N)}
        # print("define 调用")
        components = self.TNOF(P[:, 3:], k, TN, A)
        if not components:  # 如果组件列表为空
            print(f"Warning: No components found for cluster ")
            return  # 或者根据需求采取其他处理方式
        #components = self.clustering(P[:, 3:], k, TN)
        largest_component = max(components, key=len)
        mc_ids = P[list(largest_component), 0].astype(int)

        # edge_list = []
        # for idx in P[:, 3:].shape[0]:
        #     for j in TN[idx][k-1]:
        #         edge_list.append((int(P[idx, 0]), int(P[j, 0])))

        self.MacroClusters[self.MacroClusters[:, 0] == macroCluster, 2] = [mc_ids]
        self.MacroClusters[self.MacroClusters[:, 0] == macroCluster, 1] = len(mc_ids)
        for x in list(largest_component):
            j = int(P[x, 0])

            self.MCs[self.MCs[:, 0] == j, 2] = macroCluster
        return

    def DefineMacroC(self):
        # 获取未分配宏簇的微簇
        unassigned_mcs = self.MCs[self.MCs[:, 2] == 0, :]
        num_mcs = len(unassigned_mcs)
        if num_mcs >= self.n_micro:
            # 提取微簇的中心坐标
            mc_centers = unassigned_mcs[:, 3:]
            mc_all = self.MCs[:, 3:]
            k = self.k
            TN_part = self.search_TN(mc_centers, k)
            TN_all = self.search_TN(mc_all, k)
            #TNOF
            #N = len(mc_centers)  # 样本总数
            Dis_part = cdist(mc_centers, mc_centers)  # 计算距离矩阵
            Dis_all=cdist(mc_all, mc_all)

            # 获取每个点的k个紧邻
            #TN = {i: TN0[i][:k] for i in range(N)}
            # print("define 调用")
            components =self.TNOF_all(mc_centers,mc_all,k,TN_part,Dis_part,TN_all,Dis_all)
            #components = self.clustering(mc_centers, k, TN)
            for component in components:
                #print(component)
                if len(component) >= self.n_micro:  # 至少包含 n_micro 个微簇
                    self.MacroC_Num += 1  # 新的宏簇编号
                    # print(self.MacroC_Num)
                    self.colors = np.array([plt.cm.Spectral(each) for each in np.linspace(0, 1, self.MacroC_Num + 1)])
                    color = self.colors[-2, :]  # 为该宏簇分配颜色

                    # 获取连通分量中微簇的 ID
                    mc_ids = unassigned_mcs[list(component), 0].astype(int)

                    # 保存宏簇信息
                    self.MacroClusters = np.vstack((self.MacroClusters,
                                                    np.array([self.MacroC_Num, len(mc_ids), mc_ids, color],
                                                             dtype=object)))  # 修改
                    # print(self.MacroClusters)
                    print("----------Macro Cluster #", self.MacroC_Num, " is defined----------")
                    # 更新微簇的宏簇编号
                    for mc_id in mc_ids:
                        self.MCs[self.MCs[:, 0] == mc_id, 2] = self.MacroC_Num

    def search_TN(self, X, k):
        N = X.shape[0]
        XIndex = [None] * N
        KNN = [[] for _ in range(N)]
        RKNN = [[] for _ in range(N)]
        TN = [[[] for _ in range(k)] for _ in range(N)]
        tree = KDTree(X)  # construct kdtree
        for i in range(N):
            d1, ind1 = tree.query(X[i, :], k + 1)
            XIndex[i] = ind1
        for r in range(k):
            for j in range(N):
                if XIndex[j][r + 1] < N:
                    y = XIndex[j][r + 1]
                    KNN[j] = KNN[j] + [y] if KNN[j] is not None else [y]
                    RKNN[y] = RKNN[y] + [j] if RKNN[y] is not None else [j]
            for i in range(N):
                if len(KNN[i]) > 0 and len(RKNN[i]) > 0:
                    TN[i][r] = list(set(KNN[i]) & set(RKNN[i]))
        return TN

    def clustering(self, X, k, TN):
        import numpy as np
        import networkx as nx

        N = X.shape[0]
        num_cl_point = N
        # num_noise = len(noise)
        s = [N - 1]
        t = [N - 1]
        for i in range(num_cl_point):
            if TN[i] is None or len(TN[i]) <= 1 or TN[i][k - 1] is None:
                continue
            tn = TN[i][k - 1]
            # tn = list(set(tn) - set(noise))
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

    def TNOF(self, D, k, TN, Dis, figure=0):
        # global n
        n = D.shape[0]  # 数据集点的个数
        lines = []
        for i in range(n):
            last_neighbors = TN[i][k - 1]  # 取最后一轮 (r = k-1) 的可信邻居

            if len(last_neighbors) != 0:
                for j in last_neighbors:
                    lines.append((i, j))
            else:
                lines.append((i, i))

        if figure:
            self.draw(D, lines)

        # 计算逆密度
        re_density = np.zeros(n)
        for i in range(n):
            last_neighbors = TN[i][k - 1]  # 取最后一轮 (r = k-1) 的可信邻居

            if len(last_neighbors) != 0:
                re_density[i] = np.sum(Dis[i, last_neighbors]) / (len(last_neighbors) ** 2)
            else:
                re_density[i] = 0

        Cd = np.mean(re_density) +3.5* np.std(re_density)  # tuning coefficient alpha



        noise1 = np.where(re_density > Cd)[0]
        noise2 = np.where(re_density == 0)[0]
        noises = np.concatenate((noise1, noise2))

        # Performing the noise cutting
        for i in noises:
            lines = [line for line in lines if i not in line]

        if figure:
            self.draw(D, lines)

        # Assigning the other points
        G = nx.Graph(lines)
        connected_components = list(nx.connected_components(G))


        return connected_components

    def TNOF_all(self, D, D_all, k, TN_part, Dis_part, TN_all, Dis_all, figure=0):
        # global n
        n = D.shape[0]  # 数据集点的个数
        n_all = D_all.shape[0]
        lines = []
        for i in range(n):
            last_neighbors = TN_part[i][k - 1]  # 取最后一轮 (r = k-1) 的可信邻居

            if len(last_neighbors) != 0:
                for j in last_neighbors:
                    lines.append((i, j))
            else:
                lines.append((i, i))

        if figure:
            self.draw(D, lines)

        # 计算逆密度
        re_density = np.zeros(n)
        for i in range(n):
            lastpart_neighbors = TN_part[i][k - 1]  # 取最后一轮 (r = k-1) 的可信邻居

            if len(lastpart_neighbors) != 0:
                re_density[i] = np.sum(Dis_part[i, lastpart_neighbors]) / (len(lastpart_neighbors) ** 2)
            else:
                re_density[i] = 0
        re_denall = np.zeros(n_all)
        for i in range(n_all):
            lastall_neighbors = TN_all[i][k - 1]  # 取最后一轮 (r = k-1) 的可信邻居

            if len(lastall_neighbors) != 0:
                re_denall[i] = np.sum(Dis_all[i, lastall_neighbors]) / (len(lastall_neighbors) ** 2)
            else:
                re_denall[i] = 0
        Cd = np.mean(re_denall) + np.std(re_denall)  # tuning coefficient alpha 可以修改

        noise1 = np.where(re_density > Cd)[0]
        noise2 = np.where(re_density == 0)[0]
        noises = np.concatenate((noise1, noise2))

        # Performing the noise cutting
        for i in noises:
            lines = [line for line in lines if i not in line]

        if figure:
            self.draw(D, lines)

        # Assigning the other points
        G = nx.Graph(lines)
        connected_components = list(nx.connected_components(G))

        return connected_components
    def draw(self,D, lines):
        plt.figure()
        plt.plot(D[:, 0], D[:, 1], 'r.', markersize=10)
        for line in lines:
            x = [D[line[0], 0], D[line[1], 0]]
            y = [D[line[0], 1], D[line[1], 1]]
            plt.plot(x, y, 'b-')
        plt.show()


    def AddMCtoMacroC(self):  # Assign any MC that enoughly close to MC that was assigned MacroC
        if (self.MacroClusters.shape[0] != 0):
            for i in range(self.MCs.shape[0]):
                if (self.MCs[i, 2] == 0 and self.MCs[i, 1] >= self.N):
                    A = self.MCs[self.MCs[:, 2] != 0, :]
                    if (A.shape[0] > 0):
                        kdtree = KDTree(A[:, 3:])
                        d, ind = kdtree.query(self.MCs[i, 3:])
                        # if(d<=2*self.r):  优化
                        self.MCs[i, 2] = A[ind, 2]
                        MacroC = int(A[ind, 2])
                        # print("MC #",int(self.MCs[i,0])," is assigned to MacroC #",MacroC," over MC #",int(A[ind,0]))
                        # self.MacroClusters[self.MacroClusters[:,0]==MacroC,2]=[np.vstack((    self.MacroClusters[self.MacroClusters[:,0]==MacroC,2][0],[int(self.MCs[i,0]),int(A[ind,0])]   ))]
                        # print("add调用")
                        self.UpdateMacroC(self.MCs[i, 2])

    def KillMacroC(self):
        for i in range(self.MacroClusters.shape[0]):
            summ = 0
            edges = self.MacroClusters[i, 2]
            for e in edges:
                summ = summ + self.MCs[self.MCs[:, 0] == e, 1]
            # print(summ)
            # print(len(edges))
            if (summ < self.n_micro * self.N and len(edges) < self.n_micro):
                # if(self.MacroClusters[i,1]<self.n_micro):
                # for j in range(len(np.unique(self.MacroClusters[i,2]))):
                # print("Before ",self.MCs[self.MCs[:,0]==j,:])
                self.MCs[self.MCs[:, 2] == self.MacroClusters[i, 0], 2] = 0
                self.buffered_data[self.buffered_data[:,2] == self.MacroClusters[i, 0],2] = 0
                # print("After ",self.MCs[self.MCs[:,0]==j,:])
                print("----------Macro Cluster #", self.MacroClusters[i, 0], " is killed----------")
                self.MacroClusters = np.delete(self.MacroClusters, i, axis=0)
                return;

    def plotGraph(self,title,dpi=70,Flag=1):
        ax = plt.gca()
        ax.cla() # clear things for fresh plot 
        plt.rcParams['figure.dpi'] = dpi
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                            hspace=.01)
        plt.rcParams["figure.figsize"] = (10,10)
        plt.gca().set_aspect('equal', adjustable='box')

        if Flag:
            tmp = np.concatenate((self.deleted_data, self.buffered_data), axis=0)
        else:
            tmp = self.buffered_data
        for i in range(len(tmp)):
            if tmp[i, 2]==0:
                plt.plot(tmp[i, 3], tmp[i, 4],'o',color='purple',markeredgecolor='k',alpha=.55, markersize=5)
            else:
                plt.plot(tmp[i, 3], tmp[i, 4],'o',color=self.color_dict[tmp[i, 2]],alpha=.35, markersize=5)

        for i in range(len(self.MCs)):
            col=self.color_dict[self.MCs[i, 2]]
            plt.plot(self.MCs[i, 3], self.MCs[i, 4],'rd',markeredgecolor='k', markersize=1)
            circle1=plt.Circle((self.MCs[i,3],self.MCs[i,4]),self.MCr[i],color=col,linewidth=3, clip_on=False,fill=False)
            # plt.text(self.MCs[i, 3], self.MCs[i, 4],int(self.MCs[i, 0]),horizontalalignment='right')
            ax.add_patch(circle1)
        plt.title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.grid()
        # plt.xlim([0,1])
        # plt.ylim([0,1])
        plt.show()

    def purity_score(self, y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    def plotGraph3D(self, title, dpi=70):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.rcParams['figure.dpi'] = dpi
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                            hspace=.01)
        plt.rcParams["figure.figsize"] = (6, 6)

        # 绘制聚类数据点
        for i in range(len(self.buffered_data)):
            ax.scatter(self.buffered_data[i, 3], self.buffered_data[i, 4], self.buffered_data[i, 5],
                       c='b', marker='o', edgecolors='k', alpha=.15, s=50)

        # 绘制宏聚类中心
        for i in range(len(self.MCs)):
            if self.MCs[i, 2] == 0:
                col = (0, 0, 1, 1)  # 蓝色
            else:
                col = self.MacroClusters[self.MacroClusters[:, 0] == self.MCs[i, 2], 3][0].tolist()

            ax.scatter(self.MCs[i, 3], self.MCs[i, 4], self.MCs[i, 5], c='r', marker='d', edgecolors='k', s=50)
            self._plot_sphere(ax, self.MCs[i, 3], self.MCs[i, 4], self.MCs[i, 5], self.r, col)
            # 绘制圆形区域，作为聚类中心的可视化
            # circle1 = plt.Circle((self.MCs[i, 3], self.MCs[i, 4]), self.r, color=col, fill=False, clip_on=False)
            # ax.add_patch(circle1)  # 添加圆形区域

        # 设置标题、标签等
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True)

        # 设置坐标轴范围（可根据数据进行调整）
        ax.set_xlim([-0.25, 1.25])
        ax.set_ylim([-0.25, 1.25])
        ax.set_zlim([-0.25, 1.25])

        # 显示图形
        plt.show()

    # 修改后画图
    def plotGraph3D_normal(self, title, dpi=70):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.rcParams['figure.dpi'] = dpi
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                            hspace=.01)
        plt.rcParams["figure.figsize"] = (6, 6)

        # 检查 self.deleted_data 是否存在
        if hasattr(self, "deleted_data") and self.deleted_data is not None and len(self.deleted_data) > 0:
            scatter_deleted = ax.scatter(self.deleted_data[:, 3], self.deleted_data[:, 4], self.deleted_data[:, 5],
                                         c=self.deleted_data[:, 2], cmap='viridis', alpha=0.35)

        # 检查 self.buffered_data 是否存在
        if hasattr(self, "buffered_data") and self.buffered_data is not None and len(self.buffered_data) > 0:
            scatter_buffered = ax.scatter(self.buffered_data[:, 3], self.buffered_data[:, 4],
                                          self.buffered_data[:, 5],
                                          c=self.buffered_data[:, 2], cmap='viridis', alpha=0.35
                                          )
            plt.colorbar(scatter_buffered)  # 添加颜色条

        # 设置标题、标签
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True)

        # 设置坐标轴范围
        ax.set_xlim([-0.25, 1.25])
        ax.set_ylim([-0.25, 1.25])
        ax.set_zlim([-0.25, 1.25])

        # 添加图例
        ax.legend()

        # 显示图像
        plt.show()
    def purity_score(self, y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    def _plot_sphere(self, ax, x_center, y_center, z_center, radius, color):
        # 生成球体的数据
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = x_center + radius * np.outer(np.cos(u), np.sin(v))
        y = y_center + radius * np.outer(np.sin(u), np.sin(v))
        z = z_center + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # 绘制球体的表面
        ax.plot_surface(x, y, z, color=color, alpha=0.2, linewidth=0)