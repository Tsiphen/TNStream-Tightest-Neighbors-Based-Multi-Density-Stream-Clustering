import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
import pandas as pd
import time
def calculate_purity(y_true, y_pred):
    """计算 Purity"""
    contingency_matrix = np.zeros((len(np.unique(y_true)), len(np.unique(y_pred))))
    for true_label, pred_label in zip(y_true, y_pred):
        contingency_matrix[true_label, pred_label] += 1
    return np.sum(np.max(contingency_matrix, axis=0)) / len(y_true)

def calculate_ari(y_true, y_pred):
    """计算 Adjusted Rand Index (ARI)"""
    return adjusted_rand_score(y_true, y_pred)

def calculate_nmi(y_true, y_pred):
    """计算 Normalized Mutual Information (NMI)"""
    return normalized_mutual_info_score(y_true, y_pred)
def l2_normalize(features):
    """对特征矩阵进行 L2 归一化"""
    norms = np.linalg.norm(features, axis=1, keepdims=True)  # 计算每个样本的 L2 范数
    norms[norms == 0] = 1  # 避免除以零
    return features / norms  # 归一化
class DataPoint:
    def __init__(self, feature, segment=None):
        self.feature = feature  # D-dimensional feature vector
        self.segment = segment  # (start_time, end_time)

class Cluster:
    def __init__(self, name):
        self.name = name
        self.timeline = []
        self.points = []  # Data points in this cluster
        self.center = None  # Cluster centroid

class DRSCDM:
    def __init__(self, minpts=5, maxpts=15, alpha=60, beta=65, N_offline=100):
        self.minpts = minpts
        self.maxpts = maxpts
        self.theta_alpha = np.cos(np.deg2rad(alpha))
        self.theta_beta = np.cos(np.deg2rad(beta))
        self.N_offline = N_offline
        
        # Data structures
        self.pList = []
        self.micro_clusters = []
        self.central_clusters = []
        self.center_vectors = []
        
        # Counters and timing
        self.count = 0
        self.time_records = []
        self.last_time = time.time()
        self.time_csv = "processing_times.csv"
        
        # Initialize CSV file
        with open(self.time_csv, 'w') as f:
            f.write("Cumulative Processing Time (seconds)\n")

    def online_stage(self, file_path):
        """在线阶段：从文件加载数据流并处理"""
        df = pd.read_csv(file_path)
        features = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        features = l2_normalize(features)
        
        for feature, label in zip(features, labels):
            segment = (self.count, self.count + 1)
            new_point = DataPoint(feature, segment)
            self.pList.append(new_point)
            self.count += 1
            
            # Record time every 20 points
            
            current_time = time.time()
            elapsed = current_time - self.last_time
            self.time_records.append(elapsed)
                
                # Write to CSV (append mode)
            with open(self.time_csv, 'a') as f:
                f.write(f"{sum(self.time_records)}\n")
                
            self.last_time = current_time
            
            # Trigger offline clustering
        if self.count >= self.N_offline:
            self.offline_stage()

    def offline_stage(self):
        """离线阶段：执行聚类"""
    # 1. 计算与现有中心簇的相似度
        features = np.array([p.feature for p in self.pList])
        if len(self.center_vectors) > 0:
            sim_matrix = cosine_similarity(features, self.center_vectors)
            assigned_points = []
        
        # 将数据点分配到现有中心簇
            for i, p in enumerate(self.pList):
                max_sim = np.max(sim_matrix[i])
                if max_sim >= self.theta_beta:
                    cluster_idx = np.argmax(sim_matrix[i])
                    self.central_clusters[cluster_idx].points.append(p)
                    assigned_points.append(p)
        
        # 移除已分配的点
            self.pList = [p for p in self.pList if p not in assigned_points]

    # 2. 使用AngularNS和DPCS进行聚类
        remaining_points = self.pList.copy()
        processed_points = set()  # Track processed points to avoid removing twice
    
        while len(remaining_points) > 0:
        # 计算每个点的邻居数
            neighbor_counts = []
            for p in remaining_points:
                neighbors = self.find_angular_neighbors(p, remaining_points, self.theta_alpha)
                neighbor_counts.append(len(neighbors))
        
        # DPCS策略：选择密度最高的点
            max_idx = np.argmax(neighbor_counts)
            p_seed = remaining_points[max_idx]
            neighbors = self.find_angular_neighbors(p_seed, remaining_points, self.theta_alpha)
        
            if len(neighbors) >= self.minpts:
            # 创建微簇或中心簇
                new_cluster = Cluster(name=f"Cluster_{len(self.central_clusters)}")
                cluster_points = [p_seed] + neighbors
                new_cluster.points = cluster_points
                new_cluster.center = np.mean([p.feature for p in new_cluster.points], axis=0)
            
                if len(neighbors) >= self.maxpts:
                    self.central_clusters.append(new_cluster)
                    self.center_vectors.append(new_cluster.center)
            
            # Mark all cluster points as processed
                processed_points.update(cluster_points)
        
        # Remove the seed point from remaining_points
            if p_seed in remaining_points:
                remaining_points.remove(p_seed)
        
        # Remove all processed points from remaining_points
            remaining_points = [p for p in remaining_points if p not in processed_points]
    
    # 3. 异常点重新分配
        self.reassign_outliers()
    
    # 4. 清理过期数据
        self.data_expiration()

    def find_angular_neighbors(self, target_point, points, threshold):
        """基于角度选择邻居"""
        target_feature = target_point.feature.reshape(1, -1)
        features = np.array([p.feature for p in points])
        similarities = cosine_similarity(target_feature, features)[0]
        neighbors = [points[i] for i in np.where(similarities >= threshold)[0]]
        return neighbors

    def reassign_outliers(self):
        """重新分配异常点到最近的中心簇"""
        outliers = [p for p in self.pList if not any(p in c.points for c in self.central_clusters)]
        if len(outliers) == 0:
            return
        
        features = np.array([p.feature for p in outliers])
        sim_matrix = cosine_similarity(features, self.center_vectors)
        
        for i, p in enumerate(outliers):
            max_sim = np.max(sim_matrix[i])
            if max_sim >= self.theta_beta:
                cluster_idx = np.argmax(sim_matrix[i])
                self.central_clusters[cluster_idx].points.append(p)
        
        # 更新pList
        self.pList = [p for p in self.pList if p not in outliers]

    def data_expiration(self):
        """数据过期机制：删除稳定簇中的数据点"""
        expired_points = []
        for cluster in self.central_clusters:
            expired_points.extend(cluster.points)
        
        # 保留中心向量，删除原始数据
        self.pList = [p for p in self.pList if p not in expired_points]
        for cluster in self.central_clusters:
            cluster.points = []

    def extract_feature(self, data):
        """特征提取函数（需替换为实际模型）"""
        # 示例：随机生成特征向量
        return data  # 假设使用ECAPA-TDNN的192维特征

# 示例用法

if __name__ == "__main__":
    drscdm = DRSCDM(
        minpts=5,
        maxpts=15,
        alpha=60,
        beta=65,
        N_offline=100
    )
    dataset_file = "merged_output.csv"
    
    start = time.time()
    drscdm.online_stage(dataset_file)
    end = time.time()
    
    print(f"Total processing time: {end - start} seconds")