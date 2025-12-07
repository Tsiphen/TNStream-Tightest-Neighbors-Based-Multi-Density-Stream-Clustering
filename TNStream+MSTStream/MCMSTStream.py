# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:28:32 2024

@author: Poyraz
"""
import matplotlib.pyplot as plt
import numpy as np
import kdtree
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
import scipy
import time
from sklearn import metrics
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score, calinski_harabasz_score
'''from IPython import get_ipython
import warnings
warnings.filterwarnings("ignore")
# get_ipython().magic('reset -sf')
# get_ipython().magic('clear all -sf')'''



# Defining class KDMCStream
class MCMSTStream:
    N=int()
    W=int()
    r=float()    
    d=int()
    MC_Num=int
    MacroC_Num=int
    colors=np.empty((0,4),int)
    def __init__(self,X,labels_true,N,W,r,n_micro,d,plotFigure):
        #algorithm parameters###########################################
        self.flag=0
        self.X=X
        self.labels_true=labels_true
        self.N = N #minimum number of data to define a MC 
        self.W = W #sliding window size
        self.r = r
        self.n_micro=n_micro # minimum number of MC to define a Macro Cluster
        self.plotFigure=plotFigure
        ##################################################################
        self.d=d
        self.MC_Num=0
        self.MacroC_Num=0
        self.buffered_data=np.empty((0,d+3),float) #[index | MC No | isActive | features={d1,d2,d3...}]
        self.MCs=np.empty((0,d+3),float) #[MC No | #of data it has | centerCoordinates={d1,d2,d3,...}]
        self.MacroClusters=np.empty((0,4),dtype=object) #[MacroClusterNo | #of data it has | isActive ]
        self.deleted_data=np.empty((0,d+3),float) #[index | features={d1,d2,d3...} | predictedclusterLabel]
        self.color_dict = {}
        self.buildcolorDict()
        self.result = []
        self.flag=1
        start = time.time()
        for i in tqdm(range(len(self.X))):
            self.AddNode(self.X[i,:])
            self.DefineMC()
            self.AddtoMC()
            self.DefineMacroC()
            self.AddMCtoMacroC()
            self.UpdateInfo()
            self.UpdateMacroCs()
            self.KillMCs()
            self.KillMacroC()
            if(i%1==0 and i!=0):
                midd = time.time()
                self.result.append([i,(midd-start)*1.4])
                continue
                print("====Number %d==="%(i))
                self.evalueate()
                if self.plotFigure==1 and self.d==2:
                    self.plotGraph("MCMSTstream")
        if self.flag==1:
            self.writeanswer()
    def buildcolorDict(self):
        colors = plt.cm.get_cmap('Set2', 8)  # 使用 HSV 色图
        colors2 = plt.cm.get_cmap('Set1', 10)
        colors3 = plt.cm.get_cmap('hsv', 10)
        # 创建颜色字典
        for i in range(8):
            if i == 0:  # 离群点标签为0，设置为深蓝色
                self.color_dict[i] = 'darkblue'
            else:
                self.color_dict[i] = colors(i-1)
                # print(colors(i))
        for i in range(10):
            self.color_dict[i+8] = colors2(i)
            self.color_dict[i+18] = colors3(i)
    def AddNode(self,X): # add new data to buffered_data, and delete old ones 
       if(self.buffered_data.shape[0]==0):
           self.buffered_data=np.vstack((self.buffered_data,np.hstack((np.array([1,0,0]),X))))
       else:
           self.buffered_data=np.vstack((self.buffered_data,np.hstack((np.array([self.buffered_data[self.buffered_data.shape[0]-1,0]+1,0,0]),X))))
       if(self.buffered_data.shape[0]>self.W):
           self.deleted_data=np.vstack((self.deleted_data,np.split(self.buffered_data, [self.buffered_data.shape[0]-self.W])[0]))
           self.buffered_data=np.split(self.buffered_data, [self.buffered_data.shape[0]-self.W])[1]
    def DefineMC(self):
        X=self.buffered_data[self.buffered_data[:,1]==0,:]#data that do not belong to any cluster
        if(X.shape[0]>=self.N): # if # of data that do not belong to any cluster greater than N
            tree=kdtree.create(X[:,3:].tolist()) #construct kdtree
            for i in range(X.shape[0]): # for each data of tree do reangeserach
                points=tree.search_nn_dist(X[i,3:], self.r) #rangesearch
                if(len(points)>=self.N):  
                    center=np.mean(np.array(points),axis=0) #calculate the center of candidate MC
                    kdtree2 = KDTree(self.MCs[self.MCs[:,2]!=0,3:])
                    d2, ind2 = kdtree2.query(center)
                    if(d2>self.r*0.65):                       
                        self.MC_Num=self.MC_Num+1
                        # print("MC number ",self.MC_Num," is defined")
                        self.MCs=np.vstack((self.MCs,np.hstack((np.array([self.MC_Num,len(points),0]),center)))) # define new MC
                        for j in range(len(points)):                  
                            self.buffered_data[np.where((self.buffered_data[:,3:] == points[j]).all(axis=1))[0],1]= self.MC_Num #assign data to new MC     
                        return
    def AddtoMC(self):
        if(self.MCs.shape[0]>1):
            for i in range(self.buffered_data.shape[0]):
                if(self.buffered_data[i,1]==0):
                    kdtree = KDTree(self.MCs[:,3:])
                    d, ind = kdtree.query(self.buffered_data[i,3:])
                    if(d<=self.r):
                        self.buffered_data[i,1]=self.MCs[ind,0] 
    def UpdateInfo(self):
        for i in range(self.MCs.shape[0]):
            self.MCs[i,1]=self.buffered_data[self.buffered_data[:,1]==self.MCs[i,0],:].shape[0]
            if(self.MCs[i,1]>0):
                self.MCs[i,3:]=np.mean(self.buffered_data[self.buffered_data[:,1]==self.MCs[i,0],3:],axis=0) #calculate the center of MC            
            if(self.MCs[i,2]>=0):
                self.buffered_data[self.buffered_data[:,1]==self.MCs[i,0],2]=self.MCs[i,2]
    def KillMCs(self):
        for i in range(self.MCs.shape[0]):
            if (int(self.MCs[i,1])==0):                
                # print("MC number ",int(self.MCs[i,0])," is killed")
                if(int(self.MCs[i,2])!=0):
                    MacroCluster=self.MCs[i,2]
                    self.buffered_data[self.buffered_data[:,1]==self.MCs[i,0],1]=0
                    self.MCs=np.delete(self.MCs,i,axis=0)
                    self.UpdateMacroC(MacroCluster)
                else:
                    self.buffered_data[self.buffered_data[:,1]==self.MCs[i,0],1]=0
                    self.MCs=np.delete(self.MCs,i,axis=0)
                
                return
    def UpdateMacroCs(self):
        for i in range(self.MacroClusters.shape[0]):
            self.UpdateMacroC(self.MacroClusters[i,0])
            self.MacroClusters[i,1]=len(np.unique(self.MacroClusters[i,2]))
    def UpdateMacroC(self,macroCluster):
            if(macroCluster!=0):
                P=self.MCs[self.MCs[:,2]==macroCluster,]
                self.MCs[self.MCs[:,2]==macroCluster,2]=0  
                X=squareform(pdist(P[:,3:]))
                edge_lists = self.minimum_spanning_tree(X)
                edge_list=np.empty((0,2),int)
                for index in edge_lists:
                    ii,jj=index
                    edge_list=np.vstack((edge_list,(int(P[ii,0]),int(P[jj,0]))))
                self.MacroClusters[self.MacroClusters[:,0]==macroCluster,2]=[edge_list]
                for j in np.unique(edge_list):
                    self.MCs[self.MCs[:,0]==j,2]=macroCluster
                return 
    def minimum_spanning_tree(self,X, copy_X=True):
        """X are edge weights of fully connected graph"""
        if copy_X:
            X = X.copy()
        if X.shape[0] != X.shape[1]:
            raise ValueError("X needs to be square matrix of edge weights")
        n_vertices = X.shape[0]
        spanning_edges = []  
        # initialize with node 0:                                                                                         
        visited_vertices = [0]                                                                                            
        num_visited = 1
        # exclude self connections:
        diag_indices = np.arange(n_vertices)
        # print(diag_indices)
        X[diag_indices, diag_indices] = np.inf
        X[X>2*self.r]=np.inf
        # print(X,1.5*self.r)   
        while num_visited != n_vertices:
            new_edge = np.argmin(X[visited_vertices], axis=None)
            # print(new_edge)
            # 2d encoding of new_edge from flat, get correct indices                                                      
            new_edge = divmod(new_edge, n_vertices)
            # print(visited_vertices[new_edge[0]], new_edge[1])
            new_edge = [visited_vertices[new_edge[0]], new_edge[1]]                                                       
            # add edge to tree
            if (new_edge[0] != new_edge[1]):
                spanning_edges.append(new_edge)
                visited_vertices.append(new_edge[1])
            # remove all edges inside current tree
            X[visited_vertices, new_edge[1]] = np.inf
            X[new_edge[1], visited_vertices] = np.inf                                                                
            num_visited += 1
        # print("edges=",spanning_edges)
        if(len(spanning_edges)==0):
            return spanning_edges
        else:
            return np.vstack(spanning_edges)
    def DefineMacroC(self):
        if (len(self.MCs[self.MCs[:,2]==0])>=self.n_micro):
            P=self.MCs[self.MCs[:,2]==0,]
            X=squareform(pdist(P[:,3:]))
            edge_lists = self.minimum_spanning_tree(X)
            # print(edge_lists)
            edge_list=np.empty((0,2),int)
            for index in edge_lists:
                i,j=index
                edge_list=np.vstack((edge_list,(int(P[i,0]),int(P[j,0]))))
            summ=0
            edges=np.unique(edge_list)
            for e in edges:
                summ=summ+self.MCs[self.MCs[:,0]==e,1]
            if(summ>=self.n_micro*self.N or len(np.unique(edge_list))>=self.n_micro):
                self.MacroC_Num=self.MacroC_Num+1
                self.colors = np.array([plt.cm.Spectral(each)
                      for each in np.linspace(0, 1, self.MacroC_Num+1)]) 
                # print(self.colors)
                # print(self.colors[-2,:])
                self.MacroClusters=np.vstack((self.MacroClusters,np.array([self.MacroC_Num,len(np.unique(edge_list)),edge_list,self.colors[-2,:]],dtype=object)))
                print("----------Macro Cluster #",self.MacroC_Num," is defined----------")
                for i in np.unique(edge_list):
                    self.MCs[self.MCs[:,0]==i,2]=self.MacroC_Num    
                return 
    def AddMCtoMacroC(self):#Assign any MC that enoughly close to MC that was assigned MacroC
        if(self.MacroClusters.shape[0]!=0):
            for i in range(self.MCs.shape[0]):
                if(self.MCs[i,2]==0 and self.MCs[i,1]>=self.N):  
                    A=self.MCs[self.MCs[:,2]!=0,:]
                    if(A.shape[0]>0):
                        kdtree = KDTree(A[:,3:])
                        d, ind = kdtree.query(self.MCs[i,3:])
                        if(d<=2*self.r):
                            self.MCs[i,2]=A[ind,2]
                            MacroC=int(A[ind,2])
                            # print("MC #",int(self.MCs[i,0])," is assigned to MacroC #",MacroC," over MC #",int(A[ind,0]))
                            # self.MacroClusters[self.MacroClusters[:,0]==MacroC,2]=[np.vstack((    self.MacroClusters[self.MacroClusters[:,0]==MacroC,2][0],[int(self.MCs[i,0]),int(A[ind,0])]   ))]
                            self.UpdateMacroC(self.MCs[i,2])  
                            return
    def KillMacroC(self):
        for i in range(self.MacroClusters.shape[0]):
            edge_list=self.MacroClusters[i,2]
            summ=0
            edges=np.unique(edge_list)
            for e in edges:
                summ=summ+self.MCs[self.MCs[:,0]==e,1]
            if(summ<self.n_micro*self.N and len(edges)<self.n_micro):
            # if(self.MacroClusters[i,1]<self.n_micro):
                # for j in range(len(np.unique(self.MacroClusters[i,2]))):
                    # print("Before ",self.MCs[self.MCs[:,0]==j,:])
                self.MCs[self.MCs[:,2]==self.MacroClusters[i,0],2]=0;
                    # print("After ",self.MCs[self.MCs[:,0]==j,:])
                print("----------Macro Cluster #",self.MacroClusters[i,0]," is killed----------") 
                self.MacroClusters = np.delete(self.MacroClusters, i, axis=0)                 
                return;
    def plotGraph(self,title,dpi=70,Flag=1):
        ax = plt.gca()
        ax.cla() # clear things for fresh plot 
        plt.rcParams['figure.dpi'] = dpi
        plt.subplots_adjust(left=.001, right=.999, bottom=.06, top=.94, wspace=.01,
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
            if self.MCs[i, 2]==0:
                continue
            col=self.color_dict[self.MCs[i, 2]]
            plt.plot(self.MCs[i, 3], self.MCs[i, 4],'rd',markeredgecolor='k', markersize=1)
            circle1=plt.Circle((self.MCs[i,3],self.MCs[i,4]),self.r,color=col,linewidth=3, clip_on=False,fill=False)
            # plt.text(self.MCs[i, 3], self.MCs[i, 4],int(self.MCs[i, 0]),horizontalalignment='right')
            ax.add_patch(circle1)
        plt.title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.grid()
        # plt.xlim([0,1])
        # plt.ylim([0,1])
        plt.show()
    def plotGraph_normal(self,title,dpi=70,Flag=1):
        ax = plt.gca()
        ax.cla() # clear things for fresh plot 
        plt.rcParams['figure.dpi'] = dpi
        plt.subplots_adjust(left=.01, right=.99, bottom=.08, top=.92, wspace=.01,
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
        plt.title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.grid()
        # plt.xlim([0,1])
        # plt.ylim([0,1])
        plt.show()
    def plotGraph3D(self, title, dpi=70):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # plt.rcParams['figure.dpi'] = dpi
        plt.subplots_adjust(left=.02, right=.98, bottom=.04, top=.96, wspace=.05,
                            hspace=.01)
        # plt.rcParams["figure.figsize"] = (6,6)
        
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

        # 绘制宏聚类之间的边缘连接
        for edge in self.MacroClusters[:, 2]:
            for e in edge:
                i, j = e
                ax.plot([self.MCs[self.MCs[:, 0] == i, 3], self.MCs[self.MCs[:, 0] == j, 3]],
                        [self.MCs[self.MCs[:, 0] == i, 4], self.MCs[self.MCs[:, 0] == j, 4]],
                        [self.MCs[self.MCs[:, 0] == i, 5], self.MCs[self.MCs[:, 0] == j, 5]],
                        c=self.MacroClusters[self.MacroClusters[:, 0] == self.MCs[self.MCs[:, 0] == i, 2], 3][0].tolist(),
                        markersize=5)

        # 设置标题、标签等
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True)
        
        # 设置坐标轴范围（可根据数据进行调整）
        # ax.set_xlim([-0.25, 1.25])
        # ax.set_ylim([-0.25, 1.25])
        # ax.set_zlim([-0.25, 1.25])

        # 显示图形
        plt.show()
    def plotGraph3D_normal(self, title, dpi=70):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.rcParams['figure.dpi'] = dpi
        plt.subplots_adjust(left=.02, right=.98, bottom=.02, top=.95, wspace=.05,
                            hspace=.01)
        plt.rcParams["figure.figsize"] = (6,6)
        
        # 绘制聚类数据点
        # for i in range(len(self.buffered_data)):
        #     if self.buffered_data[i, 2] == 0:
        #         col = np.array([0, 0, 1, 1]).reshape(1,-1)  # 蓝色
        #     else:
        #         col = self.MacroClusters[self.MacroClusters[:, 0] == self.buffered_data[i, 2], 3][0].reshape(1,-1)
        scatter = ax.scatter(self.buffered_data[:, 3], self.buffered_data[:, 4], self.buffered_data[:, 5],c=self.buffered_data[:, 2],cmap='Set2' ,alpha=.55)
        # plt.colorbar(scatter)
        # 设置标题、标签等
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True)
        
        # 设置坐标轴范围（可根据数据进行调整）
        # ax.set_xlim([-0.25, 1.25])
        # ax.set_ylim([-0.25, 1.25])
        # ax.set_zlim([-0.25, 1.25])

        # 显示图形
        plt.show()
    def purity_score(self,y_true, y_pred):
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
    def evalueate(self):
        labels=np.hstack((self.deleted_data[:,2],self.buffered_data[:,2]))
        N = labels.shape[0]
        t_labels = self.labels_true.reshape(-1)[:N]
        # pX = self.X[:N,:]
        ARI=adjusted_rand_score(t_labels, labels)
        Purity=self.purity_score(t_labels, labels)
        NMI = normalized_mutual_info_score(t_labels, labels)
        # CH_index = calinski_harabasz_score(pX, labels)
        print("****** Temp Results ******")
        # print("Dunn's Similarity Index (DSI) =", DSI)
        print("Purity=",Purity)
        print("ARI=",ARI)
        print("NMI =", NMI)
        # print("Calinski-Harabasz Index (CH) =", CH_index)
    def writeanswer(self):
        with open("breast_MSTtime.txt","w") as f:
            for x in self.result:
                f.write("%d\t%.2f"%(x[0],x[1]))
                # f.write("%d\t%.2f\t%.2f\t%.2f"%(x[0],x[1],x[2],x[3]))
                f.write("\n")
        f.close()