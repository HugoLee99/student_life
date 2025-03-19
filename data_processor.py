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
import matplotlib.pyplot as plt
import random
# from NodeSketch import NodeSketch
from NodeSketch2 import NodeSketch
class DataProcessor:
    def __init__(self, base_path: str, user_id: str):
        self.base_path = base_path
        self.user_id = user_id
        self.graphs = {}  # 存储不同日期的图
        self.location_cluster = LocationCluster(
            memory_file=f'processed_data\location_memory_u{user_id}.pkl'
        )
        self.node_sketch = NodeSketch()  # 初始化 NodeSketch 实例
        
    # 转化数据为日期时间格式，删除无效的数据，如缺少经纬度，时间戳等等
    def load_sensor_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载传感器数据"""
        print("开始加载传感器数据...")

        # 加载GPS数据
        gps_path = os.path.join(self.base_path, f'sensing/gps/gps_u{self.user_id}.csv')

        if not os.path.exists(gps_path):
            raise FileNotFoundError(f"GPS数据文件不存在: {gps_path}")

        # 读取 GPS 数据
        df_gps = pd.read_csv(gps_path, low_memory=False)
        df_gps = df_gps.dropna(subset=['latitude', 'longitude'])

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

        # 加载活动数据
        activity_path = os.path.join(self.base_path, f'sensing/activity/activity_u{self.user_id}.csv')
        print(f"尝试加载活动数据: {activity_path}")
        if not os.path.exists(activity_path):
            raise FileNotFoundError(f"活动数据文件不存在: {activity_path}")

        # 读取活动数据
        df_activity = pd.read_csv(activity_path, 
                                  names=['timestamp', 'activity_inference'],
                                  dtype={'timestamp': str, 'activity_inference': str},
                                  low_memory=False)
        df_activity['timestamp'] = pd.to_numeric(df_activity['timestamp'], errors='coerce')
        df_activity = df_activity.dropna(subset=['timestamp'])
        df_activity['time'] = pd.to_datetime(df_activity['timestamp'], unit='s')
        print(f"活动数据清理后数据量: {len(df_activity)}")

        # 加载蓝牙数据
        bluetooth_path = os.path.join(self.base_path, f'sensing/bluetooth/bt_u{self.user_id}.csv')
        print(f"尝试加载蓝牙数据: {bluetooth_path}")
        if not os.path.exists(bluetooth_path):
            raise FileNotFoundError(f"蓝牙数据文件不存在: {bluetooth_path}")

        # 读取蓝牙数据
        df_bluetooth = pd.read_csv(bluetooth_path, 
                                   names=['time', 'MAC', 'class_id', 'level'],
                                   dtype={'time': str, 'MAC': str, 'class_id': str, 'level': str},
                                   low_memory=False)
        df_bluetooth['time'] = pd.to_numeric(df_bluetooth['time'], errors='coerce')
        df_bluetooth = df_bluetooth.dropna(subset=['time'])
        df_bluetooth['time'] = pd.to_datetime(df_bluetooth['time'], unit='s')

        if len(df_gps) == 0:
            raise ValueError("GPS数据清理后为空，请检查数据格式")

        return df_gps, df_activity, df_bluetooth
    
    def visualize_location_graph(self, G: nx.Graph,save_path):
        save_path = 'visualize/'+ save_path
        """可视化位置图"""
        pos = {}
        labels = {}
        default_position = lambda: (random.uniform(-180, 180), random.uniform(-90, 90))  # 随机默认位置生成器

        for node, data in G.nodes(data=True):
            longitude, latitude = data.get('longitude'), data.get('latitude')
            if longitude is None or latitude is None:
                longitude, latitude = default_position()
            pos[node] = (longitude, latitude)
            labels[node] = f"{node}\nVisits: {data.get('visits', 'N/A')}"
        
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold')
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        plt.title("Location Graph")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        
    #对 GPS 坐标进行聚类，并将聚类结果添加为图的节点。添加连续访问位置之间的边。
    # hiversine 100米 半径
    def create_location_graph(self, df_gps_day: pd.DataFrame, eps=100, min_samples=3) -> nx.Graph:
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
                          visits=location_info['visits'] if location_info else 1)#visits是次数
        
        # 添加连续访问位置之间的边
        prev_cluster = None
        for cluster in clusters:
            if cluster != -1:
                if prev_cluster is not None and prev_cluster != cluster:
                    G.add_edge(f'L{prev_cluster}', f'L{cluster}')
                prev_cluster = cluster
        
        return G
    # 添加活动子图。 考虑能不能加个权重？？
    def add_activity_subgraph(self, G: nx.Graph, df_activity_day: pd.DataFrame) -> nx.Graph:
        """添加活动子图"""
        activities = df_activity_day['activity_inference'].unique()
        for activity in activities:
            G.add_node(f'A{activity}', type='activity')
        
        # 添加相邻时间段之间的边
        activity_sequence = df_activity_day[['time', 'activity_inference']].values
        for i in range(len(activity_sequence) - 1):
            current_activity = activity_sequence[i][1]
            next_activity = activity_sequence[i + 1][1]
            current_time = activity_sequence[i][0]
            next_time = activity_sequence[i + 1][0]
            # 仅在相邻时间段之间添加边
            if next_time > current_time:
                G.add_edge(f'A{current_activity}', f'A{next_activity}')

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
    
    def create_multi_channel_graph(self, 
                                 df_gps_day: pd.DataFrame, 
                                 df_activity_day: pd.DataFrame, 
                                 df_bluetooth_day: pd.DataFrame,
                                 date) -> Dict[str, nx.Graph]:
        """创建多通道动态图，按GPS聚类划分"""
        # 使用DBSCAN聚类GPS坐标
        coordinates = df_gps_day[['latitude', 'longitude']].values
        # dista
        dbscan = DBSCAN(eps=0.05, min_samples=3)# eps 是以千米为单位 eps 小于这个值就算一个cluster 精度20-50 米
        clusters = dbscan.fit_predict(coordinates)
        
        # 为GPS数据添加聚类标签
        df_gps_day['cluster'] = clusters
        
       
        
        # 对每个有效的聚类创建子图
        unique_clusters = np.unique(clusters)
        G_location = nx.Graph() # 子图一个Afeature Bfeature loc
        main_graph = nx.Graph() # 一张图
        prev_cluster = None
        prev_end_time = None
        for cluster in unique_clusters:
            if cluster == -1:  # 跳过噪声点
                continue
            
            # 获取当前聚类的时间范围
            cluster_data = df_gps_day[df_gps_day['cluster'] == cluster]
            cluster_times = cluster_data['time']
            start_time = cluster_times.min()
            end_time = cluster_times.max()
            
            # 获取当前聚类该时间范围内的各类数据
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
            
            A_feature = None
            B_feature = None
            if not activity_window.empty:
                # 创建活动子图
                G_activity = nx.Graph()
                self.add_activity_subgraph(G_activity, activity_window)
                A_feature = self.get_graphs_embedding(G_activity)
                self.add_activity_subgraph(main_graph, activity_window)
            
                # self.visualize_location_graph(G_activity, os.path.join('activity_', self.user_id, f'{date}_{cluster}.png'))
            
            if not bluetooth_window.empty:
                # 创建蓝牙子图
                G_bluetooth = nx.Graph()
                self.add_bluetooth_subgraph(G_bluetooth, bluetooth_window)
                B_feature = self.get_graphs_embedding(G_bluetooth)
                self.add_bluetooth_subgraph(main_graph, bluetooth_window)
                
                # self.visualize_location_graph(G_bluetooth, os.path.join('bluetooth_', self.user_id, f'{date}_{cluster}.png'))
            if not gps_window.empty:
                # 创建位置子图，只包含当前聚类的节点

                mask = clusters == cluster #是一个布尔数组，用于标识当前聚类的节点。 mask = np.array([True, False, True, False, False, False, True])
                center = coordinates[mask].mean(axis=0) # 计算中间坐标

                # 这个是记录为一天的整张位置图
                G_location.add_node(f'L{cluster}',
                                  type='location',
                                  latitude=center[0],
                                  longitude=center[1],
                                  A_feature=A_feature,
                                  B_feature=B_feature)
                
                main_graph.add_node(f'L{cluster}',
                                  type='location',
                                  latitude=center[0],
                                  longitude=center[1],
                                  A_feature=A_feature,
                                  B_feature=B_feature)
                

            # 添加相邻地点聚类相邻之间的边
            if prev_cluster is not None and prev_end_time is not None:
                # print(f'当前时间{start_time},上一个时间{prev_end_time}')
                if start_time > prev_end_time:
                    # print(f"符合添加边: L{prev_cluster} -> L{cluster}")
                    G_location.add_edge(f'L{prev_cluster}', f'L{cluster}')
            
            prev_cluster = cluster
            prev_end_time = end_time
            
            # self.visualize_location_graph(G_location, os.path.join('location_', self.user_id, f'{date}.png'))
            # main_graph 可以替代static_graph

        return G_location,main_graph
    # 将 NetworkX 图转换为 PyTorch Geometric 数据格式。
    # 创建节点特征和边索引，并转换为 PyTorch 张量。
    def get_graphs_embedding(self, G: nx.Graph) -> np.ndarray:
        # 初始化 NodeSketch 实例
        sketch = NodeSketch()
        sketch.fit(G)
        
        # 为每个节点生成嵌入表示并添加到节点属性中
        for node in G.nodes():
            embedding = sketch.get_embedding(node)
            G.nodes[node]['feature'] = embedding
        
        # 计算整张图的特征表示（这里使用所有节点嵌入的平均值）
        all_embeddings = np.array([G.nodes[node]['feature'] for node in G.nodes()])
        graph_feature = np.mean(all_embeddings, axis=0)
        return graph_feature
        
        
    def convert_to_pytorch_geometric(self, G: nx.Graph) -> Data:
        """将NetworkX图转换为PyTorch Geometric数据格式"""
        # 检查 G 是否为空
        if G is None or len(G.nodes()) == 0:
            print("Graph G is empty or None.")
            raise ValueError("Graph G is empty or None.")
        
        # 训练 Node2Vec 模型
        self.node_sketch.fit(G)
        
        # 创建节点特征
        node_features = []
        nodes_list = list(G.nodes())
        
        for node in nodes_list:
            # 使用 Node Sketch 嵌入生成节点特征
            embedding = self.node_sketch.get_embedding(node)
            # 检查节点是否有 feature 属性
            if 'A_feature' and 'B_feature' in G.nodes[node] and G.nodes[node]['A_feature'] is not None and G.nodes[node]['B_feature'] is not None:
                # 将 NodeSketch 嵌入与 feature 属性中的嵌入进行融合
                A_feature = G.nodes[node]['A_feature']
                B_feature = G.nodes[node]['A_feature']
                fused_embedding = (embedding + A_feature + B_feature) / 3  # 可以更改为拼接吗，考虑到信息保留最大化
            else:
                fused_embedding = embedding
            node_features.append(fused_embedding)
            
        
        # 创建边索引和边权重
        edge_index = []
        edge_weight = []
        for edge in G.edges():
            # 获取节点的索引
            source = list(G.nodes()).index(edge[0])
            target = list(G.nodes()).index(edge[1])
            edge_index.append([source, target])
            edge_index.append([target, source])  # 无向图需要添加反向边
            # 添加边的权重
            weight = G.edges[edge].get('weight', 1.0)#如果键 'weight' 不存在，则返回默认值 1.0
            edge_weight.append(weight)
            edge_weight.append(weight)  # 无向图需要添加反向边的权重
        
        # 转换为PyTorch张量
        x = torch.FloatTensor(node_features)
        edge_index = torch.LongTensor(edge_index).t()
        edge_weight = torch.FloatTensor(edge_weight)
   
        
        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)   
    def convert_to_pytorch_geometric_temporal(self, full_graph: nx.Graph, 
                                            location_graph: Dict[str, Dict]) -> Tuple[Data, Dict[str, List[Data]]]:
        """将静态图和动态图转换为PyTorch Geometric时序数据格式"""
        # convert_to_pytorch_geometric 要改动成为！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        full_graph = self.convert_to_pytorch_geometric(full_graph)
        location_graph = self.convert_to_pytorch_geometric(location_graph)
        
        
        return location_graph,full_graph
    
    #调用主函数，主要用于构建每一天的图
    def build_daily_graphs(self) -> Dict[str, Tuple[nx.Graph, Dict[str, Dict[str, nx.Graph]]]]:
        """构建每日的全局图和地点特征融合图"""
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
            
            
            # 创建动态图（按GPS聚类划分）
            location_graph,static_graph = self.create_multi_channel_graph(
                df_gps_day, df_activity_day, df_bluetooth_day,date
            )
            
            # 保存图
            daily_graphs[date_str] = (static_graph, location_graph)
        
        print(f"总共构建了 {len(daily_graphs)} 天的图")
        return daily_graphs
    
    # 作废了不知到为什么会读取出错
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
