import numpy as np
import math
from collections import defaultdict
import random

class DStream:
    def __init__(self, DIMENSION, LAMBDA, GRID_LEN, UPPERBOUND, Cm=3.0, Cl=0.8, gap=10):
        """
        初始化D-Stream算法参数。
        
        :param DIMENSION: 数据维度
        :param LAMBDA: 衰减因子
        :param GRID_LEN: 网格长度
        :param UPPERBOUND: 密度阈值上限
        :param Cm: 稠密网格阈值参数
        :param Cl: 稀疏网格阈值参数
        :param gap: 时间间隔，若为None则自动计算
        """
        self.DIMENSION = DIMENSION
        self.LAMBDA = LAMBDA
        self.GRID_LEN = GRID_LEN
        self.UPPERBOUND = UPPERBOUND
        self.Cm = Cm
        self.Cl = Cl
        
        # 网格字典，键为网格坐标，值为特征向量
        self.grid_list = {}
        
        # 设置gap时间
        self.gap = gap
        
        # 当前时间
        self.current_time = 0
        
        # 调试信息
        self.debug = True
    
    def _calculate_gap(self):
        """
        根据命题4.1和4.2计算gap时间。
        """
        gap_dense = math.log((self.Cl * self.UPPERBOUND) / (self.Cm * self.UPPERBOUND)) / math.log(self.LAMBDA)
        gap_sparse = math.log((self.Cm * self.UPPERBOUND) / (self.Cl * self.UPPERBOUND)) / math.log(self.LAMBDA)
        return min(gap_dense, gap_sparse)
    
    def _map_to_grid(self, point):
        """
        将数据点映射到对应的网格。
        
        :param point: 数据点
        :return: 网格坐标
        """
        grid_coords = tuple(int(coord // self.GRID_LEN) for coord in point)
        return grid_coords
    
    def _update_grid_density(self, grid_coords, current_time):
        """
        更新网格密度（命题3.1）。
        
        :param grid_coords: 网格坐标
        :param current_time: 当前时间
        """
        if grid_coords not in self.grid_list:
            # 新网格，初始化特征向量
            self.grid_list[grid_coords] = {
                'tg': current_time,
                'tm': current_time,
                'D': 1.0,
                'label': None,
                'status': 'NORMAL'
            }
            if self.debug:
                print(f"New grid created at {grid_coords}")
        else:
            # 更新现有网格的密度
            grid = self.grid_list[grid_coords]
            elapsed_time = current_time - grid['tg']
            grid['D'] = grid['D'] * (self.LAMBDA ** elapsed_time) + 1
            grid['tg'] = current_time
            if self.debug:
                print(f"Grid {grid_coords} density updated to {grid['D']}")
    
    def _calculate_grid_density(self, grid_coords, current_time):
        """
        计算网格在当前时间的密度。
        
        :param grid_coords: 网格坐标
        :param current_time: 当前时间
        :return: 密度值
        """
        if grid_coords not in self.grid_list:
            return 0.0
        grid = self.grid_list[grid_coords]
        elapsed_time = current_time - grid['tg']
        return grid['D'] * (self.LAMBDA ** elapsed_time)
    
    def _classify_grid(self, grid_coords, current_time):
        """
        根据密度对网格进行分类（稠密、稀疏、过渡）。
        
        :param grid_coords: 网格坐标
        :param current_time: 当前时间
        :return: 网格类型
        """
        density = self._calculate_grid_density(grid_coords, current_time)
        avg_density = self.UPPERBOUND / len(self.grid_list) if len(self.grid_list) > 0 else 0
        
        if density >= self.Cm * avg_density:
            return 'DENSE'
        elif density <= self.Cl * avg_density:
            return 'SPARSE'
        else:
            return 'TRANSITIONAL'
    
    def _is_sporadic(self, grid_coords, current_time):
        """
        判断网格是否为松散网格（定义4.1）。
        
        :param grid_coords: 网格坐标
        :param current_time: 当前时间
        :return: 是否为松散网格
        """
        grid = self.grid_list[grid_coords]
        density = self._calculate_grid_density(grid_coords, current_time)
        threshold = self.UPPERBOUND * (self.LAMBDA ** (current_time - grid['tm']))
        
        return density < threshold and (current_time - grid['tg']) > self.gap
    
    def _remove_sporadic_grids(self, current_time):
        """
        移除松散网格（规则D1和D2）。
        """
        to_remove = []
        for grid_coords in list(self.grid_list.keys()):
            grid = self.grid_list[grid_coords]
            if grid['status'] == 'SPORADIC' and (current_time - grid['tg']) > self.gap:
                to_remove.append(grid_coords)
                if self.debug:
                    print(f"Removing sporadic grid at {grid_coords}")
            elif self._is_sporadic(grid_coords, current_time):
                grid['status'] = 'SPORADIC'
                if self.debug:
                    print(f"Marking grid at {grid_coords} as sporadic")
        
        for grid_coords in to_remove:
            del self.grid_list[grid_coords]
    
    def _find_adjacent_grids(self, grid_coords):
        """
        找到邻接网格（定义3.3）。
        
        :param grid_coords: 网格坐标
        :return: 邻接网格列表
        """
        adjacent = []
        for dim in range(self.DIMENSION):
            for delta in [-1, 1]:
                neighbor = list(grid_coords)
                neighbor[dim] += delta
                neighbor = tuple(neighbor)
                if neighbor in self.grid_list:
                    adjacent.append(neighbor)
        return adjacent
    
    def _form_clusters(self, current_time):
        """
        形成簇（定义3.6）。
        
        :return: 簇的列表
        """
        clusters = []
        visited = set()
        
        for grid_coords in self.grid_list:
            if grid_coords in visited:
                continue
            grid_type = self._classify_grid(grid_coords, current_time)
            if grid_type == 'DENSE':
                cluster = self._expand_cluster(grid_coords, visited, current_time)
                if cluster:
                    clusters.append(cluster)
                    if self.debug:
                        print(f"New cluster formed: {cluster}")
        
        return clusters
    
    def _expand_cluster(self, grid_coords, visited, current_time):
        """
        扩展簇（广度优先搜索）。
        """
        cluster = []
        queue = [grid_coords]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            current_type = self._classify_grid(current, current_time)
            if current_type == 'DENSE' or current_type == 'TRANSITIONAL':
                cluster.append(current)
                neighbors = self._find_adjacent_grids(current)
                queue.extend(neighbors)
        
        return cluster if len(cluster) > 0 else None
    
    def process_point(self, point):
        """
        处理一个数据点（在线部分）。
        
        :param point: 数据点
        :return: 簇列表或None
        """
        self.current_time += 1
        grid_coords = self._map_to_grid(point)
        self._update_grid_density(grid_coords, self.current_time)
        
        # 每隔gap时间执行离线部分
        if self.current_time % self.gap == 0:
            if self.debug:
                print(f"\n=== Offline processing at time {self.current_time} ===")
                print(f"Total grids: {len(self.grid_list)}")
            
            self._remove_sporadic_grids(self.current_time)
            clusters = self._form_clusters(self.current_time)
            
            if self.debug:
                print(f"Found {len(clusters)} clusters\n")
            
            return clusters
        return None

def generate_clustered_point():
    """生成有聚类结构的数据点"""
    cluster_center = random.choice([
        [0.3, 0.3],  # 第一个簇中心
        [0.7, 0.7],  # 第二个簇中心
        [0.5, 0.2]   # 第三个簇中心
    ])
    return [random.gauss(center, 0.05) for center in cluster_center]

if __name__ == "__main__":
    # 参数设置
    DIMENSION = 2       # 数据维度
    LAMBDA = 0.998      # 衰减因子
    GRID_LEN = 0.1      # 网格长度
    UPPERBOUND = 1000   # 密度阈值上限
    
    # 初始化D-Stream，设置较小的gap值
    dstream = DStream(DIMENSION, LAMBDA, GRID_LEN, UPPERBOUND, gap=10)
    
    # 模拟数据流 - 使用有聚类结构的数据
    for i in range(500):  # 增加循环次数
        point = generate_clustered_point()
        clusters = dstream.process_point(point)
        
        if clusters is not None:
            print(f"\n=== Clusters formed at time {dstream.current_time} ===")
            for j, cluster in enumerate(clusters, 1):
                print(f"Cluster {j}: {len(cluster)} grids")