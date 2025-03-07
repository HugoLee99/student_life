import os
from h11 import Data
import torch
from data_processor import DataProcessor
import pandas as pd
from typing import List
from datetime import datetime
from imblearn.over_sampling import RandomOverSampler
import sys
import math
def calculate_f1_score(predictions, labels):
    """计算F1分数"""
    from sklearn.metrics import f1_score
    # 将预测转换为类别
    pred_labels = torch.argmax(predictions, dim=1).cpu().numpy()
    true_labels = labels.cpu().numpy()
    return f1_score(true_labels, pred_labels, average='weighted')

def save_f1(f1_scores, model_name, model_type):
    column_name = f'{model_type}_{model_name}'
    file_path = f'output/f1_scores_{model_type}_new.csv'
    
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 读取现有文件
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            # 如果列名不存在，则添加新的列名
            df[column_name] = pd.Series([None] * len(df))
    else:
        # 如果文件不存在，则创建一个新的 DataFrame
        df = pd.DataFrame(columns=['Round', column_name])
    
    # 更新或添加 F1 分数
    if f1_scores:  # 只有在有 F1 分数时才保存
        new_data = pd.DataFrame({
            'Round': range(1, len(f1_scores) + 1),
            column_name: f1_scores
        })
        df = pd.concat([df, new_data], ignore_index=True)
        df = df.groupby('Round', as_index=False).first()  # 去重，保留第一个出现的值
        df.to_csv(file_path, index=False)
        print(f"{column_name}F1分数已保存到 {file_path}")
    else:
        # 创建一个空的 CSV 文件
        df.to_csv(file_path, index=False)
        print(f"没有 F1 分数，已创建空文件 {file_path}")
                 
def average_weights(weights_list: List[dict]) -> dict:
    """聚合多个模型的权重"""
    avg_weights = {}
    for key in weights_list[0].keys():
        avg_weights[key] = sum(weights[key] for weights in weights_list) / len(weights_list)
    return avg_weights
def increment_data_transform(time_step, train_data,valid_data):
    processed_train = []
    processed_val = []
    print("time_step = ",time_step)
    for i in range(time_step):
        
        one_batch_train = [] # for each client in this time step
        one_batch_val = []
        for client_data in train_data:
            one_batch_train.append(client_data[i])
        for client_data in valid_data:
            one_batch_val.append(client_data[i])

        processed_val.append(one_batch_val)
        processed_train.append(one_batch_train)
    return processed_train, processed_val
    # 输入是[client nums, batch nums, 三元组]，输出变成一个二维列表[batch nums, client nums, 三元组]
    
    
def save_all_batches_f1(f1_scores, model_name, model_type):


    column_name = f'{model_type}_{model_name}'
    file_path = f'output/CatstroForget_Compare_{model_type}.csv'
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 读取现有文件
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            # 如果列名不存在，则添加新的列名
            df[column_name] = pd.Series([None] * len(df))
    else:
        # 如果文件不存在，则创建一个新的 DataFrame
        df = pd.DataFrame(columns=['Batch', column_name])
    # 更新或添加 F1 分数
    if f1_scores:  # 只有在有 F1 分数时才保存
        new_data = pd.DataFrame({
            'Batch': range(1, len(f1_scores) + 1),
            column_name: f1_scores
        })
        df = pd.concat([df, new_data], ignore_index=True)
        df = df.groupby('Batch', as_index=False).first()  # 去重，保留第一个出现的值
        df.to_csv(file_path, index=False)
        print(f"{column_name} F1分数已保存到 {file_path}")
    else:
        # 创建一个空的 CSV 文件
        df.to_csv(file_path, index=False)
        print(f"没有 F1 分数，已创建空文件 {file_path}")   
#加y标签和负责划分数据的调用 主函数
def prepare_data(client_pergroup, client_nums):
    # 处理label 的
    print("开始准备数据...")
    all_expert_data = []
    all_apprentice_data = []
    all_expert_batches = []
    all_apprentice_batches = []
    all_expert_val_batches = []
    all_apprentice_val_batches = []
    count = 0  # 用于每三个用户为一组
    server_num = 0
    processed_data = []  # 将 processed_data 改为列表格式

    user_index = 0  # 用于跟踪实际处理的用户数量
    while user_index < client_nums and user_index < 60:
        user_id = f"{user_index:02d}"
        print(f"\n处理用户 {user_id} 的数据...")

        # 加载压力水平数据
        stress_file = f'processed_stress/stress_levels_u{user_id}.csv'
        if not os.path.exists(stress_file):
            print(f"警告: 用户 {user_id} 在processed_stress 目录下 没有压力水平数据，跳过")
            user_index += 1
            client_nums += 1
            continue

        stress_df = pd.read_csv(stress_file)
        stress_df['date'] = pd.to_datetime(stress_df['date']).dt.date
        stress_dict = dict(zip(stress_df['date'], stress_df['stress_level']))

        # 返回的是一个字典，键是日期，值是 (静态图, 动态图) 的元组
        processor = DataProcessor(base_path='dataset', user_id=user_id)
        daily_graphs = processor.build_daily_graphs()

        if not daily_graphs:
            print(f"警告: 用户 {user_id} 在processed_stress 目录下的 csv 里面没有图数据")
            user_index += 1 
            client_nums += 1
            continue

        # 转换为PyTorch Geometric格式
        print(f"开始转换用户 {user_id} 的数据为PyTorch Geometric格式...")

        for date, (full_graph, location_graph) in daily_graphs.items():
        
            # 确保这一天有压力水平数据
            date_obj = datetime.strptime(date, '%Y-%m-%d').date()
            

            # print(f"处理 {date} 的图数据")
            if len(full_graph.nodes()) == 0 or len(location_graph.nodes()) == 0:
                print(f"警告: {date} 的图数据为空，跳过")
                continue

            static_data, full_graph = processor.convert_to_pytorch_geometric_temporal(
                full_graph, location_graph
            )

            # 检查并修复空图
            if not hasattr(static_data, 'edge_index') or static_data.edge_index.numel() == 0:
                print(f"警告: {date} 的静态图没有边，添加自环")
                num_nodes = static_data.x.size(0)
                self_loops = torch.arange(num_nodes, dtype=torch.long)
                static_data.edge_index = torch.stack([self_loops, self_loops], dim=0)

            # 添加压力水平作为标签
            if date_obj in stress_dict:
                stress_level = stress_dict[date_obj]
                static_data.y = torch.tensor(stress_level, dtype=torch.long)
                
                processed_data.append({
                    'static': static_data,
                    'unlabeled': None
                })
            else:
                processed_data.append({
                    'static': None,
                    'unlabeled': static_data
                })
            

        count += 1
        user_index += 1  # 成功处理用户数据后递增计数器
        if count < client_pergroup:
            print('processed_data 个数', len(processed_data))
            continue  # 等待processed_data中的数据达到3个用户
        else:
            count = 0
            server_num += 1
            print('processed_data 个数', len(processed_data))
            all_expert_batches, all_expert_val_batches, all_apprentice_batches, all_apprentice_val_batches = output_data(
                processed_data, server_num, all_expert_data, all_expert_batches, all_apprentice_batches, all_apprentice_data, all_expert_val_batches, all_apprentice_val_batches)
            processed_data = []
            all_expert_data = []
            all_apprentice_data = []

    return all_expert_batches, all_expert_val_batches, all_apprentice_batches, all_apprentice_val_batches,server_num
def output_data(processed_data, server_id, all_expert_data, all_expert_batches, all_apprentice_batches, all_apprentice_data, all_expert_val_batches, all_apprentice_val_batches):
    num_days = len(processed_data)
    print(f"server {server_id} 总共有 {num_days} 天的数据")
    train_size = max(1, int(num_days * 0.4)) 
    val_size = max(1, int(train_size * 0.2))
    increment_size = max(1, int(num_days * 0.6 * 0.8)) 
    print(f"用户 {server_id} 训练集大小: {train_size} 天, 验证集大小: {val_size} 天")

    # 划分数据
    expert_data = processed_data[:train_size - val_size] # 百分之40中的80%数据作为训练集
    expert_val_data = processed_data[train_size - val_size:train_size] #百分之40中的20%数据作为验证集
    apprentice_data = processed_data[train_size:train_size + increment_size] # 百分之60中的80%数据作为增量数据训练集
    apprentice_val_data = processed_data[train_size + increment_size:] #百分之60中的20%数据作为增量数据验证集

    # 准备专家模型无划分训练数据  三元组(有标签静态图, 无标签图, 标签)
    expert_client_data = (
        [data['static'] for data in expert_data if data['static'] is not None],
        [data['unlabeled'] for data in expert_data if data['unlabeled'] is not None],
        torch.cat([torch.tensor([data['static'].y]) for data in expert_data if data['static'] is not None])
    )
    all_expert_data.append(expert_client_data)  # 一个用户的专家数据 (静态图，动态图，标签) 有多少个用户就有多少个三元组

    # 准备学徒模型无划分的训练数据
    apprentice_data_container = (
        [data['static'] for data in apprentice_data if data['static'] is not None],
        [data['unlabeled'] for data in apprentice_data if data['unlabeled'] is not None],
        torch.cat([torch.tensor([data['static'].y]) for data in apprentice_data if data['static'] is not None])
    )
    all_apprentice_data.append(apprentice_data_container)  # 一个用户的学徒数据 (静态图，动态图，标签) 有多少个用户就有多少个三元组

    # 准备专家模型有划分的训练数据
    print('专家模型训练数据')
    all_expert_batches = divide_batches(server_id, expert_data, all_expert_batches, 4)

    # 准备专家模型验证数据
    print('专家模型验证数据')
    all_expert_val_batches = divide_batches(server_id, expert_val_data, all_expert_val_batches, 4)

    print('学徒模型训练数据')
    all_apprentice_batches = divide_batches(server_id, apprentice_data, all_apprentice_batches, 8)

    print('学徒模型验证数据')
    all_apprentice_val_batches = divide_batches(server_id, apprentice_val_data, all_apprentice_val_batches, 8)

    if not all_expert_data:
        print("警告: 没有任何用户的数据可用，使用空数据继续")
        # 创建一个空的数据集而不是抛出错误
        all_expert_data = [([], [], torch.tensor([]))]
        all_expert_batches = [[]]
        all_apprentice_batches = [[]]
        all_apprentice_data = [([], [], torch.tensor([]))]

    # 调试输出数据结构
    print("\n调试输出数据结构:")
    print(f"all_expert_batches: {type(all_expert_batches)}, 用户个数长度: {len(all_expert_batches)}")
    print(f"all_expert_val_batches: {type(all_expert_val_batches)}, 长度: {len(all_expert_val_batches)}")
    print(f"all_apprentice_batches: {type(all_apprentice_batches)}, 长度: {len(all_apprentice_batches)}")
    print(f"all_apprentice_val_batches: {type(all_apprentice_val_batches)}, 长度: {len(all_apprentice_val_batches)}")

    # 检查每个批次的数据结构
    for i, batch in enumerate(all_expert_batches):
        print(f"all_expert_batches[{i}]: {type(batch)}, 表示时间长度: {len(batch)}")
    for i, batch in enumerate(all_expert_val_batches):
        print(f"all_expert_val_batches[{i}]: {type(batch)}, 长度: {len(batch)}")
    for i, batch in enumerate(all_apprentice_batches):
        print(f"all_apprentice_batches[{i}]: {type(batch)}, 长度: {len(batch)}")
    for i, batch in enumerate(all_apprentice_val_batches):
        print(f"all_apprentice_val_batches[{i}]: {type(batch)}, 长度: {len(batch)}")

    return all_expert_batches, all_expert_val_batches, all_apprentice_batches, all_apprentice_val_batches

def divide_batches(server_id, origin_data, result, num_batches):
    num_days = len(origin_data)
    batch_size = math.ceil(num_days / num_batches)

    user_expert_batches = []

    for i in range(0, num_days, batch_size):
        batch_data = origin_data[i:i + batch_size]

        if batch_data:
            # 三元组(有标签静态图, 无标签图, 标签)
            di_san = torch.tensor([])
            for data in batch_data:
                if data['static'] is not None:
                    di_san = torch.tensor([data['static'].y])
                    if len(di_san) != 0:
                        di_san = torch.cat([di_san])
                        
            batch_client_data = (
                [data['static'] for data in batch_data if data['static'] is not None],
                [data['unlabeled'] for data in batch_data if data['unlabeled'] is not None],
                di_san
            )
            user_expert_batches.append(batch_client_data)

    if len(user_expert_batches) > num_batches:
        user_expert_batches = user_expert_batches[:num_batches]

    while len(user_expert_batches) < num_batches:
        empty_static = []
        empty_dynamic = []
        empty_labels = torch.tensor([], dtype=torch.long)
        user_expert_batches.append((empty_static, empty_dynamic, empty_labels))

    result.append(user_expert_batches)  # 有[[4 batches]*users]，每个batch有一组数据分别是(静态图，动态图，标签)
    print(f"client server {server_id} 数据处理完成 - 数据: {num_days} 天, batches: {len(user_expert_batches)},应该要有{num_batches}个batch,每个批次有{batch_size}天")
    return result