import torch
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from torch_geometric.data import Data
from typing import Dict, List, Tuple
import os
from location_cluster import LocationCluster
from datetime import datetime
import sys
class DataProcessor:
    def __init__(self, base_path: str, user_id: str):
        self.base_path = base_path
        self.user_id = user_id
        self.graphs = {}  # 存储不同日期的图
        self.location_cluster = LocationCluster(
            memory_file=f'processed_data\location_memory_u{user_id}.pkl'
        )
        
    def load_sensor_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载传感器数据"""
        print("开始加载传感器数据...")
        
        # 加载GPS数据
        gps_path = os.path.join(self.base_path, f'sensing/gps/gps_u{self.user_id}.csv')
        # print(f"尝试加载GPS数据: {gps_path}")
        if not os.path.exists(gps_path):
            raise FileNotFoundError(f"GPS数据文件不存在: {gps_path}")
        
        # 读取GPS数据
        df_gps = pd.read_csv(gps_path, low_memory=False)
        # print("GPS数据列名:", df_gps.columns.tolist())
        # print("GPS数据前几行:")
        # print(df_gps.head())
        # print(f"GPS数据加载成功，原始数据量: {len(df_gps)}")
        
        # 重命名索引列为时间戳
        df_gps = df_gps.reset_index()
        df_gps = df_gps.rename(columns={'index': 'timestamp'})
        
        # 转换时间戳
        try:
            df_gps['time'] = pd.to_datetime(df_gps['timestamp'], unit='s')
        except:
            print("警告：时间戳转换失败，尝试其他方法")
            df_gps['time'] = pd.to_datetime(df_gps.index, unit='s')
        
        # 删除无效的GPS记录
        df_gps = df_gps.dropna(subset=['latitude', 'longitude'])
        df_gps['date'] = df_gps['time'].dt.date
        
        print(f"GPS数据清理后数据量: {len(df_gps)}")
        # if len(df_gps) > 0:
        #     print("清理后的GPS数据日期范围:", df_gps['date'].min(), "到", df_gps['date'].max())
        #     print("GPS坐标范围:")
        #     print(f"纬度: {df_gps['latitude'].min():.6f} 到 {df_gps['latitude'].max():.6f}")
        #     print(f"经度: {df_gps['longitude'].min():.6f} 到 {df_gps['longitude'].max():.6f}")
        
        # 加载活动数据
        activity_path = os.path.join(self.base_path, f'sensing/activity/activity_u{self.user_id}.csv')
        print(f"尝试加载活动数据: {activity_path}")
        if not os.path.exists(activity_path):
            raise FileNotFoundError(f"活动数据文件不存在: {activity_path}")
        df_activity = pd.read_csv(activity_path, 
                                 names=['timestamp', 'activity_inference'],
                                 dtype={'timestamp': str, 'activity_inference': str},
                                 low_memory=False)
        # print(f"活动数据加载成功，原始数据量: {len(df_activity)}")
        
        # 转换活动数据时间戳
        df_activity['timestamp'] = pd.to_numeric(df_activity['timestamp'], errors='coerce')
        df_activity = df_activity.dropna(subset=['timestamp'])
        df_activity['time'] = pd.to_datetime(df_activity['timestamp'], unit='s')
        # print(f"活动数据清理后数据量: {len(df_activity)}")
        
        # 加载蓝牙数据
        bluetooth_path = os.path.join(self.base_path, f'sensing/bluetooth/bt_u{self.user_id}.csv')
        # print(f"尝试加载蓝牙数据: {bluetooth_path}")
        if not os.path.exists(bluetooth_path):
            raise FileNotFoundError(f"蓝牙数据文件不存在: {bluetooth_path}")
        df_bluetooth = pd.read_csv(bluetooth_path, 
                                  names=['time', 'MAC', 'class_id', 'level'],
                                  dtype={'time': str, 'MAC': str, 'class_id': str, 'level': str},
                                  low_memory=False)
        # print(f"蓝牙数据加载成功，原始数据量: {len(df_bluetooth)}")
        
        # 转换蓝牙数据时间戳
        df_bluetooth['time'] = pd.to_numeric(df_bluetooth['time'], errors='coerce')
        df_bluetooth = df_bluetooth.dropna(subset=['time'])
        df_bluetooth['time'] = pd.to_datetime(df_bluetooth['time'], unit='s')
        # print(f"蓝牙数据清理后数据量: {len(df_bluetooth)}")
        
        # 确保所有数据都有有效的时间戳和坐标
        if len(df_gps) == 0:
            raise ValueError("GPS数据清理后为空，请检查数据格式")
        
        return df_gps, df_activity, df_bluetooth
    
    def create_location_graph(self, df_gps_day: pd.DataFrame, eps=0.001, min_samples=3) -> nx.Graph:
        """创建位置图"""
        # 使用LocationCluster进行聚类
        coordinates = df_gps_day[['latitude', 'longitude']].values
        clusters = self.location_cluster.fit_predict(coordinates)
        
        G = nx.Graph()
        
        # 添加位置节点
        unique_clusters = np.unique(clusters)
        for cluster in unique_clusters:
            if cluster != -1:  # 排除噪声点
                mask = clusters == cluster
                center = coordinates[mask].mean(axis=0)
                location_info = self.location_cluster.get_location_info(cluster)
                G.add_node(f'L{cluster}', 
                          type='location',
                          latitude=center[0],
                          longitude=center[1],
                          visits=location_info['visits'] if location_info else 1)
        
        # 添加连续访问位置之间的边
        prev_cluster = None
        for cluster in clusters:
            if cluster != -1:
                if prev_cluster is not None and prev_cluster != cluster:
                    G.add_edge(f'L{prev_cluster}', f'L{cluster}')
                prev_cluster = cluster
        
        return G
    
    def add_activity_subgraph(self, G: nx.Graph, df_activity_day: pd.DataFrame) -> nx.Graph:
        """添加活动子图"""
        activities = df_activity_day['activity_inference'].unique()
        for activity in activities:
            G.add_node(f'A{activity}', type='activity')
        
        # 添加连续活动之间的边
        activity_sequence = df_activity_day['activity_inference'].values
        for i in range(len(activity_sequence)-1):
            G.add_edge(f'A{activity_sequence[i]}', f'A{activity_sequence[i+1]}')
        
        return G
    
    def add_bluetooth_subgraph(self, G: nx.Graph, df_bluetooth_day: pd.DataFrame) -> nx.Graph:
        """添加蓝牙子图"""
        devices = df_bluetooth_day['MAC'].unique()
        for device in devices:
            G.add_node(f'B{device}', type='bluetooth')
        
        # 添加同时出现的蓝牙设备之间的边
        time_groups = df_bluetooth_day.groupby('time')['MAC'].apply(list)
        for devices in time_groups:
            for i in range(len(devices)):
                for j in range(i+1, len(devices)):
                    G.add_edge(f'B{devices[i]}', f'B{devices[j]}')
        
        return G
    
    def create_multi_channel_graph(self, G: nx.Graph, 
                                 df_gps_day: pd.DataFrame, 
                                 df_activity_day: pd.DataFrame, 
                                 df_bluetooth_day: pd.DataFrame) -> Dict[str, nx.Graph]:
        """创建多通道动态图，按GPS聚类划分"""
        # 使用DBSCAN聚类GPS坐标
        coordinates = df_gps_day[['latitude', 'longitude']].values
        dbscan = DBSCAN(eps=0.001, min_samples=3)
        clusters = dbscan.fit_predict(coordinates)
        
        # 为GPS数据添加聚类标签
        df_gps_day['cluster'] = clusters
        
        dynamic_graphs = {
            'location': {},
            'activity': {},
            'bluetooth': {}
        }
        
        # 对每个有效的聚类创建子图
        unique_clusters = np.unique(clusters)
        for cluster in unique_clusters:
            if cluster == -1:  # 跳过噪声点
                continue
            
            # 获取当前聚类的时间范围
            cluster_data = df_gps_day[df_gps_day['cluster'] == cluster]
            cluster_times = cluster_data['time']
            start_time = cluster_times.min()
            end_time = cluster_times.max()
            
            # 获取该时间范围内的各类数据
            gps_window = df_gps_day[
                (df_gps_day['time'] >= start_time) & 
                (df_gps_day['time'] <= end_time) &
                (df_gps_day['cluster'] == cluster)
            ]
            
            activity_window = df_activity_day[
                (df_activity_day['time'] >= start_time) & 
                (df_activity_day['time'] <= end_time)
            ]
            
            bluetooth_window = df_bluetooth_day[
                (df_bluetooth_day['time'] >= start_time) & 
                (df_bluetooth_day['time'] <= end_time)
            ]
            
            # 为每个聚类创建子图 （这个不需要了吧cluster 直接当location 了，不需要再给里面地点细化地点了）
            if not gps_window.empty:
                # 创建位置子图，只包含当前聚类的节点
                G_location = nx.Graph()
                mask = clusters == cluster
                center = coordinates[mask].mean(axis=0)
                G_location.add_node(f'L{cluster}', 
                                  type='location',
                                  latitude=center[0],
                                  longitude=center[1])
                dynamic_graphs['location'][cluster] = G_location
            
            if not activity_window.empty:
                # 创建活动子图
                G_activity = nx.Graph()
                self.add_activity_subgraph(G_activity, activity_window)
                dynamic_graphs['activity'][cluster] = G_activity
            
            if not bluetooth_window.empty:
                # 创建蓝牙子图
                G_bluetooth = nx.Graph()
                self.add_bluetooth_subgraph(G_bluetooth, bluetooth_window)
                dynamic_graphs['bluetooth'][cluster] = G_bluetooth
        
        return dynamic_graphs
    
    def convert_to_pytorch_geometric(self, G: nx.Graph) -> Data:
        """将NetworkX图转换为PyTorch Geometric数据格式"""
        # 创建节点特征
        node_features = []
        node_types = []
        nodes_list = list(G.nodes())
        
        for node in nodes_list:
            if node.startswith('L'):  # 位置节点
                coords = np.array([G.nodes[node]['latitude'], G.nodes[node]['longitude']])
                # 修改填充维度以匹配模型输入维度
                node_features.append(np.pad(coords, (0, 62)))  # 填充到64维
                node_types.append(0)
            elif node.startswith('A'):  # 活动节点
                feature = np.zeros(64)  # 修改为64维
                feature[int(node[1:])] = 1  # one-hot编码
                node_features.append(feature)
                node_types.append(1)
            elif node.startswith('B'):  # 蓝牙节点
                feature = np.random.normal(0, 0.1, 64)  # 修改为64维
                node_features.append(feature)
                node_types.append(2)
        
        # 创建边索引
        edge_index = []
        for edge in G.edges():
            # 获取节点的索引
            source = list(G.nodes()).index(edge[0])
            target = list(G.nodes()).index(edge[1])
            edge_index.append([source, target])
            edge_index.append([target, source])  # 无向图需要添加反向边
        
        # 转换为PyTorch张量
        x = torch.FloatTensor(node_features)
        edge_index = torch.LongTensor(edge_index).t()
        node_types = torch.LongTensor(node_types)
        
        return Data(x=x, edge_index=edge_index, node_type=node_types) 
    
    def convert_to_pytorch_geometric_temporal(self, static_graph: nx.Graph, 
                                            dynamic_graphs: Dict[str, Dict]) -> Tuple[Data, Dict[str, List[Data]]]:
        """将静态图和动态图转换为PyTorch Geometric时序数据格式"""
        # 转换静态图
        static_data = self.convert_to_pytorch_geometric(static_graph)
        
        # 转换动态图
        dynamic_data = {
            'location': [],
            'activity': [],
            'bluetooth': []
        }
        
        # 获取所有时间戳并排序
        
        timestamps = sorted(list(dynamic_graphs['location'].keys()))
        # print('这个timestamps 真的有吗',list(dynamic_graphs['activity']),list(dynamic_graphs['location']))

        # sys.exit()
        for timestamp in timestamps:
            # 处理每个通道的动态图
            for channel in ['location', 'activity', 'bluetooth']:
                if timestamp in dynamic_graphs[channel]:
                    graph_data = self.convert_to_pytorch_geometric(
                        dynamic_graphs[channel][timestamp]
                    )
                    dynamic_data[channel].append(graph_data)
                    #graph_data变成这种 Data(x=x, edge_index=edge_index, node_type=node_types) 
        
        return static_data, dynamic_data
    
    def build_daily_graphs(self) -> Dict[str, Tuple[nx.Graph, Dict[str, Dict[str, nx.Graph]]]]:
        """构建每日的静态图和动态图"""
        print("开始构建每日图...")
        
        # 尝试加载已存在的图
        # existing_graphs = self.load_existing_graphs()
        # if existing_graphs:
        #     print(f"加载到 {len(existing_graphs)} 天的已存在图")
        #     return existing_graphs  # 如果存在，直接返回已加载的图
        
        df_gps, df_activity, df_bluetooth = self.load_sensor_data()
        daily_graphs = {}
        
        # 打印日期范围信息
        print(f"GPS数据日期范围: {df_gps['date'].min()} 到 {df_gps['date'].max()}")
        
        # 按日期分组处理数据
        for date, df_gps_day in df_gps.groupby('date'):
            # print(f"处理日期: {date}")
            date_str = str(date) # 转化为日期一天
            df_activity_day = df_activity[df_activity['time'].dt.date == date]
            df_bluetooth_day = df_bluetooth[df_bluetooth['time'].dt.date == date]
            
            # print(f"当日数据量 - GPS: {len(df_gps_day)}, 活动: {len(df_activity_day)}, 蓝牙: {len(df_bluetooth_day)}")
            
            if len(df_gps_day) == 0:
                # print(f"警告: {date} 没有GPS数据，跳过")
                continue
            
            # 创建静态图（基础结构）
            static_graph = self.create_location_graph(df_gps_day)
            static_graph = self.add_activity_subgraph(static_graph, df_activity_day)
            static_graph = self.add_bluetooth_subgraph(static_graph, df_bluetooth_day)
            
            # 创建动态图（按GPS聚类划分）
            dynamic_graphs = self.create_multi_channel_graph(
                static_graph, df_gps_day, df_activity_day, df_bluetooth_day
            )
            
            # 保存图
            daily_graphs[date_str] = (static_graph, dynamic_graphs)
            
            # 保存为边列表文件
            output_dir = os.path.join('graph', self.user_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存静态图
            nx.write_edgelist(static_graph, 
                             os.path.join(output_dir, f'static_graph_{date_str}.edgelist'))
            
            # 保存动态图
            for channel, cluster_graphs in dynamic_graphs.items():
                channel_dir = os.path.join(output_dir, channel)
                os.makedirs(channel_dir, exist_ok=True)
                for cluster, graph in cluster_graphs.items():
                    nx.write_edgelist(graph, 
                                    os.path.join(channel_dir, f'graph_{date_str}_cluster_{cluster}.edgelist'))
            
            # print(f"完成 {date} 的图构建")
        
        print(f"总共构建了 {len(daily_graphs)} 天的图")
        return daily_graphs
    
    def load_existing_graphs(self) -> Dict[str, Tuple[nx.Graph, Dict[str, Dict[str, nx.Graph]]]]:
        """加载已存在的图"""
        existing_graphs = {}
        output_dir = os.path.join('graph', self.user_id)
        
        for filename in os.listdir(output_dir):
            if filename.endswith('.edgelist'):
                date_str = filename.split('_')[2]  # 提取日期部分
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()  # 确保日期格式正确
                    graph_path = os.path.join(output_dir, filename)
                    graph = nx.read_edgelist(graph_path)  # 读取图
                    existing_graphs[date_str] = graph
                    print(f"加载图: {filename} 对应日期: {date_str}")
                except ValueError as e:
                    print(f"处理 {filename} 的数据时出错: {str(e)}")
        
        return existing_graphs
