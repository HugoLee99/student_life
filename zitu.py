import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import networkx as nx
import matplotlib.pyplot as plt
import random


# 读取 GPS 数据
user = "00"
df_gps = pd.read_csv(f'dataset/sensing/gps/gps_u{user}.csv', header=0, index_col=False)

# 读取活动推断数据
df_activity = pd.read_csv(f'dataset\sensing/activity/activity_u{user}.csv', header=0, names=['timestamp', 'activity_inference'])
df_bluetooth = pd.read_csv(f'dataset\sensing/bluetooth/bt_u{user}.csv', header=0,names=["time","MAC","class_id","level"])
# 将 Unix 时间戳转换为可读时间格式，并按天分组
df_gps['time'] = pd.to_datetime(df_gps['time'], unit='s', errors='coerce')
df_gps['date'] = df_gps['time'].dt.date

# 将活动推断数据的时间戳也转换为 datetime
df_activity['time'] = pd.to_datetime(df_activity['timestamp'], unit='s', errors='coerce')
df_bluetooth['time'] = pd.to_datetime(df_bluetooth['time'], unit='s', errors='coerce')
def create_daily_graph(df_day, df_activity_day,df_bluetooth_day, eps=0.001, min_samples=3):
    # 提取经纬度进行聚类
    coordinates = df_day[['latitude', 'longitude']].dropna()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df_day['cluster'] = dbscan.fit_predict(coordinates)
    
    # 初始化簇的时间字典和图
    cluster_times = {}
    G = nx.Graph()
    
    # 添加聚类后的地点为节点，使用 cluster 编号作为唯一标识
    clusters = df_day['cluster'].unique()
    for cluster in clusters:
        if cluster != -1:  # 排除噪声点（DBSCAN 中簇编号为 -1）
            G.add_node(cluster, subgraph_activities=nx.Graph(),subgraph_bluetooth=nx.Graph())
    
    # 按时间序列排序，并创建边（每一对连续访问的地点之间）
    df_day_sorted = df_day.sort_values(by='time')
    
    previous_cluster = None
    for i, row in df_day_sorted.iterrows():
        current_cluster = row['cluster']
        current_time = row['time']
        
        # 记录每个簇的初始时间和离开时间
        if current_cluster != -1:
            if current_cluster not in cluster_times:
                cluster_times[current_cluster] = {'start_time': current_time, 'end_time': current_time}
            else:
                cluster_times[current_cluster]['end_time'] = current_time
        
        # 添加边
        if previous_cluster is not None and current_cluster != -1 and previous_cluster != -1:
            G.add_edge(previous_cluster, current_cluster)
        
        previous_cluster = current_cluster
    
    # 为每个簇添加活动推断数据作为子图
    for cluster, times in cluster_times.items():
        start_time = times['start_time']
        end_time = times['end_time']
        
        # 过滤该时间段内的活动数据
        cluster_activities = df_activity_day[(df_activity_day['time'] >= start_time) & 
                                             (df_activity_day['time'] <= end_time)]
        

        cluster_bluetooth = df_bluetooth_day[(df_bluetooth_day['time'] >= start_time) & 
                                             (df_bluetooth['time'] <= end_time)]
        # 处理活动数据
        subgraph = G.nodes[cluster]['subgraph_activities']
        for _, activity_row in cluster_activities.iterrows():
            activity = activity_row['activity_inference']
            if not subgraph.has_node(activity):
                subgraph.add_node(activity, times=[])
            subgraph.nodes[activity]['times'].append(activity_row['time'])

        
        # 处理蓝牙数据
        subgraph2 = G.nodes[cluster]['subgraph_bluetooth']
        for _, bluetooth_row in cluster_bluetooth.iterrows():
            bluetooth = bluetooth_row['MAC']
            # print(f"Adding Bluetooth node: {bluetooth}")  # 调试信息
            if not subgraph2.has_node(bluetooth):
                subgraph2.add_node(bluetooth, times=[])
                # print(f"Node {bluetooth} added to subgraph2")  # 调试信息
            subgraph2.nodes[bluetooth]['times'].append(bluetooth_row['time'])
        # 添加子图中活动之间的边（如果它们有连续的时间戳）
        activities = list(subgraph.nodes())
        for i in range(len(activities) - 1):
            for j in range(i+1, len(activities)):
                if any(t1 < t2 for t1 in subgraph.nodes[activities[i]]['times'] 
                                for t2 in subgraph.nodes[activities[j]]['times']):
                    subgraph.add_edge(activities[i], activities[j])
        
         # 添加子图中活动之间的边（如果它们有连续的时间戳）
        bluetooth = list(subgraph2.nodes())
        for i in range(len(bluetooth) - 1):
            for j in range(i+1, len(bluetooth)):
                if any(t1 < t2 for t1 in subgraph2.nodes[bluetooth[i]]['times'] 
                                for t2 in subgraph2.nodes[bluetooth[j]]['times']):
                    subgraph2.add_edge(bluetooth[i], bluetooth[j])
        
        print(f"Cluster {cluster}: Start time: {start_time}, End time: {end_time}")
        print(f"  Activities: {len(subgraph.nodes)}")
        print(f"  Bluetooth: {len(subgraph.nodes)}")
    
    return G, dbscan.components_  # 返回图和簇的中心

# 创建按天分组的数据
grouped_gps = df_gps.groupby('date')
grouped_activity = df_activity.groupby(df_activity['time'].dt.date)
grouped_bluetooth = df_bluetooth.groupby(df_bluetooth['time'].dt.date)
# 遍历每一天的数据并生成图

import networkx as nx
import json
import os
for date, group_gps in grouped_gps:
    print(f'Processing date: {date}')
    
    # 获取当天的活动数据
    group_activity = grouped_activity.get_group(date) if date in grouped_activity.groups else pd.DataFrame()
    group_bluetooth = grouped_bluetooth.get_group(date) if date in grouped_bluetooth.groups else pd.DataFrame()
    
    if group_bluetooth.empty or group_activity.empty:
        print("df_bluetooth_day or df_activity is empty.")
        continue    
    G, cluster_centers = create_daily_graph(group_gps, group_activity,group_bluetooth, eps=0.005, min_samples=3)  # 可根据数据调整 eps 和 min_samples
    
    

   

    # 确保路径存在
    directory = f'graph\\{user}'
    os.makedirs(directory, exist_ok=True)  # 创建目录，exist_ok=True 表示如果目录已存在不会报错
    # 保存为边列表
    nx.write_edgelist(G, f'{directory}/graph_{date}.edgelist', data=True)  # data=True 表示保留边属性

    # with open(f'graph\{user}\graph_{date}.json', 'w') as f:
    #     json.dump(graph_json, f)
# # 从 JSON 文件加载图
# with open(f'graph_{date}.json', 'r') as f:
#     graph_json = json.load(f)
# G = nx.node_link_graph(graph_json)

    # # 获取聚类中心作为每个节点的经纬度坐标
    # if len(cluster_centers) > 0:  # 确保存在聚类中心
    #     # 获取聚类中心作为每个节点的经纬度坐标，添加随机偏移
    #     pos = {i: (cluster_centers[i][1] + random.uniform(-0.001, 0.001),
    #                cluster_centers[i][0] + random.uniform(-0.001, 0.001)) for i in range(len(cluster_centers))}
        
    #     # 绘制主图
    #     plt.figure(figsize=(12, 8))
    #     nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10)
    #     plt.title(f"Main Graph for {date}")
    #     plt.show()
        
    #     # 绘制每个簇的子图（活动推断）
    #     for cluster in G.nodes():
    #         subgraph = G.nodes[cluster]['subgraph_bluetooth']
    #         if len(subgraph.nodes) > 0:
    #             plt.figure(figsize=(8, 4))
    #             nx.draw(subgraph, with_labels=True, node_size=300, node_color='lightgreen', font_size=8)
    #             nx.draw_networkx_labels(subgraph, nx.spring_layout(subgraph), 
    #                                     {node: f"{node}\n({len(data['times'])} occurrences)" 
    #                                      for node, data in subgraph.nodes(data=True)})
    #             plt.title(f"Activity Subgraph for Cluster {cluster} on {date}")
    #             plt.show()
    # else:
    #     print(f"No clusters found for {date}")