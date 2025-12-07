import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import adjusted_rand_score
from LSH_TNstream import TNStream # import class MCMSTStream from the same directory
import time
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score, calinski_harabasz_score
from scipy.spatial.distance import euclidean
from scipy.stats import ks_2samp
import random

# Test on ExclaStar dataset
dataset_name="covtype_10features"
dataset = np.loadtxt("Datasets/%s.txt"%(dataset_name), dtype=float,delimiter=' ')
X = dataset[:,:-1]
labels_true = dataset[:,-1]

T = 0
while T < 1000:
    T+=1
    # Obtained best parameters for ExclaStar dataset
    W = random.randint(900,1200) #4000
    N = random.randint(2,3)
    r = random.randint(250,7000)/10000 # 0.6642
    n_micro = random.randint(3,7)
    k=4

    ####MinMaxNormalization#######################################################
    scaler = MinMaxScaler()
    scaler.fit(X)
    MinMaxScaler()
    X=scaler.transform(X[:,:])


    plotFigure=0 # 设为 1 则每加入100个点绘制一次图像
    start = time.time()    
    kds=TNStream(X,labels_true,N,W,r,n_micro,X.shape[1],plotFigure,k,4)
    end = time.time()
    if kds.flag:
        metrics = kds.evalueate()
        if metrics["Purity"]>=0.00:
            print("=== writing results ===")
            kds.writetimetable(dataset_name)
            with open("%s_result_for_LSH_TN.txt"%(dataset_name), "a") as file:
                print("Current Dataset:",dataset_name, file=file)
                print("Current Iteration:",T, file=file)
                print("Elapsed Time:",end - start, file=file)
                print("Current Values: W=",W,"N=",N,"r=",r,"n_micro=",n_micro, file=file)
                print("\n##### The Best Results #############", file=file)
                for name,value in metrics.items():
                    print("%s ="%(name),value, file=file)
                print("\n",file=file)