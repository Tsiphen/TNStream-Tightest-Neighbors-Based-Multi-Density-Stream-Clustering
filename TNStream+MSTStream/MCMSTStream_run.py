# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:28:32 2024

@author: Poyraz
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import adjusted_rand_score
from MCMSTStream import MCMSTStream # import class MCMSTStream from the same directory
import time
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score, calinski_harabasz_score
from scipy.spatial.distance import euclidean
from scipy.stats import ks_2samp
import random
'''from IPython import get_ipython
import warnings
warnings.filterwarnings("ignore")
# get_ipython().magic('reset -sf')
# get_ipython().magic('clear all -sf')'''

dataset = np.loadtxt("Datasets/noise_new/kdd_converted.txt", dtype=float,delimiter=' ')
X=dataset[:,:-1]
labels_true=dataset[:,-1]

T = 0
while T < 1:
    T+=1
    # Obtained best parameters
    W=300
    N=2#random.randint(2,3)
    r=random.randint(450,950)/10000
    n_micro=random.randint(2,7)

    ####MinMaxNormalization#######################################################
    scaler = MinMaxScaler()
    scaler.fit(X)
    MinMaxScaler()
    X=scaler.transform(X[:,:])

    plotFigure=0 # 1 for plotting clusters 
    start = time.time()    
    kds=MCMSTStream(X,labels_true,N,W,r,n_micro,X.shape[1],plotFigure)
    end = time.time()
    print("Elapsed Time=",end - start)

    labels=np.hstack((kds.deleted_data[:,2],kds.buffered_data[:,2]))
    ARI=adjusted_rand_score(labels_true.reshape(-1), labels)
    Purity=kds.purity_score(labels_true.reshape(-1), labels)
    NMI = normalized_mutual_info_score(labels_true.reshape(-1), labels)

    if Purity>0.9:
        print("W=%d N=%d r=%f n_micro=%d"%(W,N,r,n_micro))
        print("\n\n##### The Best Results #############")
        # print("Dunn's Similarity Index (DSI) =", DSI)
        print("Purity=",Purity)
        print("ARI=",ARI)
        print("NMI =", NMI)
        # print("Calinski-Harabasz Index (CH) =", CH_index)
        if X.shape[1]==2:
            kds.plotGraph(str("MCMSTstream"))
        elif X.shape[1]==3:
            kds.plotGraph3D_normal(str("MCMSTStream"))