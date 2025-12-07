# -*- coding: utf-8 -*-
"""
修改后的KD-ARStream_Demo.py - 每20个点记录一次累计时间到CSV
"""
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from KD_ARStream import KDARStream
import warnings
warnings.filterwarnings("ignore")

# 修改时间记录文件为CSV格式
time_log_file = "KDARStream_batch_runtime.csv"
cumulative_time = 0.0  # 累计运行时间

# 创建CSV文件并写入表头
with open(time_log_file, 'w') as f:
    f.write("batch_number,samples_processed,cumulative_time,batch_time\n")  # CSV表头

# [原有代码保持不变...]
#dataset = np.loadtxt("KDD.csv", dtype=float, delimiter=',')
X = np.loadtxt("data.txt", dtype=float, delimiter=',')
labels_true = np.loadtxt("labels_2.txt", dtype=float, delimiter=',')

# [原有预处理代码保持不变...]

kds = KDARStream(N=90, TN=160, r=1.4, r_threshold=4, r_max=6.55, d=X.shape[1])

# 记录程序开始时间
program_start = time.time()
batch_start = time.time()  # 记录批次开始时间

# 修改后的处理循环
for i in range(len(X)):
    # 记录单点处理开始时间
    point_start = time.time()
    
    # 处理单个数据点
    kds.addNode(X[i])
    kds.NewClusterAppear()
    kds.findandAddClosestCluster()
    kds.splitClusters()
    kds.mergeClusters()
    kds.updateRadius()
    kds.updateCenters()
    kds.flagActiveClusters()
    
    # 计算单点处理时间并累加
    point_time = time.time() - point_start
    cumulative_time += point_time
    

    with open(time_log_file, 'a') as f:
        f.write(f"{i+1},{cumulative_time:.6f},{point_time:.6f}\n")
    



# 计算总运行时间
total_time = time.time() - program_start

# [原有验证代码保持不变...]
print(f"\n所有样本处理完成，总运行时间: {total_time:.6f} 秒")
print(f"时间记录已保存至 {time_log_file} (CSV格式)")

