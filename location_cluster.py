import numpy as np
from sklearn.cluster import DBSCAN
from typing import Dict, Tuple, List
import pickle
import os

class LocationCluster:
    def __init__(self, eps=0.001, min_samples=3, memory_file='location_memory.pkl'):
        self.eps = eps
        self.min_samples = min_samples
        self.memory_file = memory_file
        self.location_memory = {}  # 存储已知位置的字典
        self.next_cluster_id = 0
        self.load_memory()
    
    def load_memory(self):
        """加载已保存的位置记忆"""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'rb') as f:
                saved_data = pickle.load(f)
                self.location_memory = saved_data['memory']
                self.next_cluster_id = saved_data['next_id']
    
    def save_memory(self):
        """保存位置记忆到文件"""
        with open(self.memory_file, 'wb') as f:
            pickle.dump({
                'memory': self.location_memory,
                'next_id': self.next_cluster_id
            }, f)
    
    def find_matching_location(self, center: np.ndarray) -> int:
        """查找匹配的已知位置"""
        for loc_id, loc_info in self.location_memory.items():
            stored_center = loc_info['center']
            # 计算与已存储位置的距离
            distance = np.sqrt(np.sum((center - stored_center) ** 2))
            # 如果距离小于eps，认为是同一个位置
            if distance < self.eps:
                return loc_id
        return -1
    
    def fit_predict(self, coordinates: np.ndarray) -> np.ndarray:
        """对坐标进行聚类，保持位置标签的一致性"""
        # 首先使用DBSCAN进行基础聚类
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        initial_clusters = dbscan.fit_predict(coordinates)
        
        # 为每个聚类分配一致的标签
        final_clusters = np.full_like(initial_clusters, -1)
        unique_clusters = np.unique(initial_clusters)
        
        for cluster in unique_clusters:
            if cluster == -1:
                continue
                
            # 计算当前聚类的中心
            mask = initial_clusters == cluster
            center = coordinates[mask].mean(axis=0)
            
            # 查找是否匹配已知位置
            matching_id = self.find_matching_location(center)
            
            if matching_id == -1:
                # 如果是新位置，分配新的ID
                matching_id = self.next_cluster_id
                self.location_memory[matching_id] = {
                    'center': center,
                    'visits': 1
                }
                self.next_cluster_id += 1
            else:
                # 更新已知位置的信息
                self.location_memory[matching_id]['visits'] += 1
                # 更新中心点（使用移动平均）
                old_center = self.location_memory[matching_id]['center']
                visits = self.location_memory[matching_id]['visits']
                new_center = (old_center * (visits - 1) + center) / visits
                self.location_memory[matching_id]['center'] = new_center
            
            # 分配最终的聚类标签
            final_clusters[mask] = matching_id
        
        # 保存更新后的位置记忆
        self.save_memory()
        
        return final_clusters
    
    def get_location_info(self, cluster_id: int) -> Dict:
        """获取位置信息"""
        return self.location_memory.get(cluster_id, None)