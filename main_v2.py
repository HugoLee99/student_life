import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import sys
from models import GCNModel, GATModel, GCNLSTM
import math
from torch_geometric.data import Data
import random
from FL_hub import local_update_cr, incremental_local_update_cr, local_update_flwf ,local_update_fedavg, local_update_fedsem_ft, local_update_fedmatch_ft
from function import calculate_f1_score,save_f1,average_weights,increment_data_transform,save_all_batches_f1,prepare_data
random.seed(42)
torch.manual_seed(42)
def random_perturbation(x: torch.Tensor) -> torch.Tensor:
    """对输入数据添加随机扰动"""
    noise = torch.randn_like(x) * 0.1
    return x + noise


def federated_learning(
    model_type: str,
    clients: List[int],
    model: nn.Module,
    train_data: List[Tuple[List[Data], List[Data], torch.Tensor]],
    val_data: List[Tuple[List[Data], List[Data], torch.Tensor]],
    num_rounds: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    lambda_cr: float,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    model_name: str  # 添加模型名称参数
) -> Tuple[nn.Module, List[float]]:  # 修改返回类型
    """端到端增量联邦学习的专家模型训练阶段"""
    global_weights = model.state_dict()
    
    # 初始化F1分数列表
    f1_scores = []

    time_step = len(train_data[0])
    # 用于记录每个批次训练完成后，对之前所有批次数据的F1值 用于检测灾难性遗忘
    processed_data, val_data = increment_data_transform(time_step,train_data,val_data)
    # print(processed_data)
    # print('-'*50)
    # print(val_data)
    # sys.exit()
    # 输出变成[batch nums, client nums, 三元组]
    for time_index, (batch_client_data, val_data) in enumerate(zip(processed_data, val_data)):
        print(f'Traning on data batch：{time_index+1}/{time_step}')
        
        for round in range(num_rounds):
            # 每一轮通信后，清空用户的权重列表
            local_weights = []
            valid_clients = 0
            
            print(f"\nRound {round + 1}/{num_rounds}")
            print("-" * 50)
            
            # 本地更新
            for k in clients:
                local_model = type(model)(  # 创建新的本地模型实例并传递所需参数
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim
                )
                local_model.load_state_dict(global_weights)
                
                if k < len(batch_client_data):  # train_data 是一个包含所有客户端训练数据的列表
                    train_graphs, unlabeled_graphs, labels = batch_client_data[k]
                    
                    if train_graphs:
                        print(f"\nTraining Client {k}") 
                        updated_model = local_update_cr(
                            k, local_model, train_graphs, unlabeled_graphs, labels,
                            num_epochs, batch_size, learning_rate, lambda_cr
                        )
                        local_weights.append(updated_model.state_dict())
                        valid_clients += 1
            
            if local_weights:
                global_weights = average_weights(local_weights)
                model.load_state_dict(global_weights)
                print(f"\nAggregated {valid_clients} local models for round {round + 1}/{num_rounds}")
                
            # 评估当前轮次的模型性能
            predictions, all_labels = [], []
            model.eval()
            with torch.no_grad():
                for client in clients:
                    if client < len(val_data):  # 添加索引检查
                        val_graphs, _, labels = val_data[client]
                        for graph, label in zip(val_graphs, labels):
                            out = model(graph.x, graph.edge_index)
                            out_mean = torch.mean(out, dim=0, keepdim=True) # 因为是每个节点来表示一张图，所以取平均
                            # out_mean = torch.max(out, dim=0,keepdim=True)  # 最大池化保留显著特征
                            predictions.append(out_mean)  # 这里是张量
                            all_labels.append(label)  # 不再将标签移至设备
            
            # 检查 all_labels 是否为空
            if all_labels:
                # 将 predictions 和 all_labels 转换为张量
                print(len(predictions))
                predictions = torch.cat(predictions, dim=0)  # 合并所有预测
                all_labels = torch.stack(all_labels)  # 合并所有标签，确保是张量
                
                
                # 打印长度以调试
                print(f"预测数量: {predictions.size(0)}, 标签数量: {all_labels.size(0)}")
                
                if predictions.size(0) != all_labels.size(0):
                    print("警告: 预测和标签数量不一致！")
                    continue
                
                f1 = calculate_f1_score(predictions, all_labels)
                f1_scores.append(f1)
                print(f"Round {round + 1} F1 Score: {f1:.4f}")  # 打印每轮的 F1 分数
            else:
                print("警告: all_labels 为空，无法计算 F1 分数。")
            
    save_f1(f1_scores, model_name, model_type)  # 保存 F1 分数到 CSV 文件
    
    return model, f1_scores

def incremental_federated_learning(
    model_type: str,
    clients: List[int],
    apprentice_model: nn.Module,
    expert_model: nn.Module,
    train_data: List[List[Tuple[List[Data], List[Data], torch.Tensor]]],
    valid_data: List[List[Tuple[List[Data], List[Data], torch.Tensor]]],
    num_rounds: int,
    num_epochs: int,
    batch_size: int, # 这里的batch size 是指客户端内部的训练里面的size，time step 才是指分成的数据有多少batch
    
    learning_rate: float,
    alpha: float,
    lambda_cr: float,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    model_name: str  # 添加模型名称参数
) -> Tuple[nn.Module, List[float]]:  # 修改返回类型
    """端到端增量联邦学习的学徒模型训练阶段"""
    global_weights = apprentice_model.state_dict()
    # 初始化F1分数列表
    f1_scores = []
    all_batch_f1_scores =[]
    # 用于记录每个批次训练完成后，对之前所有批次数据的F1值 用于检测灾难性遗忘

    # 获取设备
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 输入是[client nums, batch nums, 三元组]，输出变成[batch nums, client nums, 三元组]
    time_step = len(train_data[0])
    processed_data, val_data = increment_data_transform(time_step,train_data,valid_data)

    for time_index, (client_batch_data, val_data) in enumerate(zip(processed_data, val_data)):
        print(f'Traning on data batch：{time_index+1}/{time_step}')
        for round in range(num_rounds):
            local_weights = []
            # print(f"for each time batch, Communication Round {round + 1}/{num_rounds}")
            # k = 0,1,2,3...
            for k in clients: 
                local_model = type(apprentice_model)(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim
                )
                local_model.load_state_dict(global_weights)

                # 现在再处理整理过的数据形式[batch 数量，client 数量，三元组]
                
                if k < len(client_batch_data):
                    # print('now k=', k, 'client_batch_data=', time_index)
                    train_graphs, unlabeled_graphs, labels = client_batch_data[k]
                    if train_graphs:
                        updated_model = incremental_local_update_cr(
                            k, local_model, expert_model, train_graphs, unlabeled_graphs, labels,
                            num_epochs, batch_size, learning_rate, alpha, lambda_cr
                        )
                        local_weights.append(updated_model.state_dict())

            if local_weights:
                global_weights = average_weights(local_weights)
                apprentice_model.load_state_dict(global_weights)
                print(f"Aggregated {len(local_weights)} local models")
            expert_model.load_state_dict(apprentice_model.state_dict())
            print(f"Updated expert model with communication round {round + 1} results")
            # 评估当前轮次的模型性能
            try:
                all_predictions = []
                all_labels = []
                apprentice_model.eval()
                with torch.no_grad():
                    for k in clients:
                        if k < len(val_data):
                            val_graphs, _, labels = val_data[k]
                            if val_graphs:
                                for graph, label in zip(val_graphs, labels):
                                    # 确保图和标签在设备上
                                    # graph = graph.to(device)
                                    # label = label.to(device)
                                    out = apprentice_model(graph.x, graph.edge_index)
                                    out = torch.mean(out, dim=0, keepdim=True)
                                    # out = torch.max(out, dim=0,keepdim=True)  # 最大池化保留显著特征
                                    
                                    # print("label",label)
                                    # print("out",out)
                                    all_predictions.append(out)
                                    all_labels.append(label)
                                
                if all_predictions:
                    predictions = torch.cat(all_predictions, dim=0)
                    labels = torch.stack(all_labels)# 确保标签在同一设备上
                    f1 = calculate_f1_score(predictions, labels)
                    f1_scores.append(f1)
                    print(f"Batch {time_index + 1}, communication Round {round + 1} F1-Score: {f1:.4f}")
        
            except Exception as e:
                print(f"评估模型时出错: {str(e)}")
        # 评估模型在之前所有批次数据上的性能
        all_hist_predictions = []
        all_hist_labels = []
        apprentice_model.eval()
      
        # with torch.no_grad():
        #     for prev_time_index in range(time_index + 1):
        #         prev_client_batch_data = val_data[prev_time_index] # 这个批次之前的每一个batch 拿出来算平均
        #         for k in clients:
        #             if k < len(prev_client_batch_data):
        #                 val_graphs, _, labels = prev_client_batch_data[k]
        #                 if val_graphs:
        #                     for graph, label in zip(val_graphs, labels):
        #                         out = apprentice_model(graph.x, graph.edge_index)
        #                         out = torch.mean(out, dim=0, keepdim=True)
        #                         all_hist_predictions.append(out)
        #                         all_hist_labels.append(label)

        if all_hist_predictions:
            predictions = torch.cat(all_hist_predictions, dim=0)
            labels = torch.stack(all_hist_labels)
            hist_f1 = calculate_f1_score(predictions, labels)
            all_batch_f1_scores.append(hist_f1)
            print(f"F1 Score on all batches after training batch {time_index + 1}: {hist_f1:.4f}")

    # 保存所有批次训练后的F1分数到CSV文件
    save_all_batches_f1(all_batch_f1_scores, model_name, model_type)
    save_f1(f1_scores, model_name, model_type)
    
    return apprentice_model, f1_scores
    # 保存F1分数到CSV文件

def fedavg(
    model_type: str,
    clients: List[int],
    model: nn.Module,
    train_data: List[Tuple[List[Data], List[Data], torch.Tensor]],
    val_data: List[Tuple[List[Data], List[Data], torch.Tensor]],
    num_rounds: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    lambda_cr: float,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    model_name: str
) -> Tuple[nn.Module, List[float]]:
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    
    f1_scores = []


    time_step = len(train_data[0])
    processed_data, val_data = increment_data_transform(time_step,train_data,val_data)
    for time_index, (train_data, val_data) in enumerate(zip(processed_data, val_data)):
        print(f'Traning on data batch：{time_index+1}/{time_step}')
      
        for round in range(num_rounds):
            local_weights = []
            print('客户总人数:', len(clients))
            
            for client in clients:
                print('处理对象:', client)
                if client >= time_step:  # 添加索引检查
                    print(f"警告: 客户端 {client} 超出训练数据范围")
                    continue
                
                # 创建新的本地模型实例并传递所需参数
                local_model = model.__class__(input_dim, hidden_dim, output_dim)
                local_model.load_state_dict(model.state_dict())
                # print('训练数据:',train_data)
                print('客户数量',len(train_data),'时间段数据包数量：',len(train_data[0]))
                # 获取客户端数据
                train_graphs, unlabeled_graphs, labels = train_data[client]
                print('三元组的训练图列表数量:',len(train_graphs))
                
                # # 将数据移至设备
                # train_graphs = [g.to(device) for g in train_graphs if g is not None]
                # unlabeled_graphs = [g.to(device) for g in unlabeled_graphs if g is not None]
                # labels = labels.to(device)
                
                # 本地训练
                updated_model = local_update_fedavg(
                    client, local_model, train_graphs, labels,
                    num_epochs, batch_size, learning_rate
                )
                local_weights.append(updated_model.state_dict())
            
            # 聚合权重
            global_weights = average_weights(local_weights)
            model.load_state_dict(global_weights)
            
            # 评估模型并计算F1分数
            predictions, all_labels = [], []
            model.eval()
            with torch.no_grad():
                for client in clients:
                    if client < len(val_data):  # 添加索引检查
                        val_graphs, _, labels = val_data[client]
                        for graph, label in zip(val_graphs, labels):
                            # graph = graph.to(device)  # 确保图在设备上
                            out = model(graph.x, graph.edge_index)
                            out = torch.mean(out, dim=0, keepdim=True)
                            # out = torch.max(out, dim=0,keepdim=True)  # 最大池化保留显著特征
                            predictions.append(out)  # 这里是张量
                            all_labels.append(label)  # 确保标签在设备上
            
             # 检查 all_labels 是否为空
            if all_labels:
                # 将 predictions 和 all_labels 转换为张量
                print(len(predictions))
                predictions = torch.cat(predictions, dim=0)  # 合并所有预测
                all_labels = torch.stack(all_labels)  # 合并所有标签，确保是张量
                
                
                # 打印长度以调试
                print(f"预测数量: {predictions.size(0)}, 标签数量: {all_labels.size(0)}")
                
                if predictions.size(0) != all_labels.size(0):
                    print("警告: 预测和标签数量不一致！")
                    continue
                
                f1 = calculate_f1_score(predictions, all_labels)
                f1_scores.append(f1)
                print(f"Round {round + 1} F1 Score: {f1:.4f}")  # 打印每轮的 F1 分数
            else:
                print("警告: all_labels 为空，无法计算 F1 分数。")
        
    save_f1(f1_scores, model_name, model_type)
    
    return model, f1_scores
def increment_fedavg(
    # time_step: int,
    model_type: str,
    clients: List[int],
    model: nn.Module,
    train_data: List[Tuple[List[Data], List[Data], torch.Tensor]],
    val_data: List[Tuple[List[Data], List[Data], torch.Tensor]],
    num_rounds: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    lambda_cr: float,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    model_name: str
) -> Tuple[nn.Module, List[float]]:
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    
    f1_scores = []
    all_batch_f1_scores =[]
    time_step = len(train_data[0])
    processed_data, val_data = increment_data_transform(time_step,train_data,val_data)
    for time_index, (client_batch_data, val_data) in enumerate(zip(processed_data, val_data)):
        print(f"Training on the collected data from {time_index+1}/{len(train_data)+1} time ")
        for round in range(num_rounds):
            local_weights = []
            print(f"for each time batch, Communication Round {round + 1}/{num_rounds}")
            for client in clients:
                print('处理对象:', client)
                if client >= len(train_data):  # 添加索引检查
                    print(f"警告: 客户端 {client} 超出训练数据范围")
                    continue
                
                # 创建新的本地模型实例并传递所需参数
                local_model = model.__class__(input_dim, hidden_dim, output_dim)
                local_model.load_state_dict(model.state_dict())
                # print('训练数据:',train_data)
            
                # 获取客户端数据
               
                if client < len(client_batch_data):
                    print('now k=', client, 'client_batch_data=',time_index)
                    train_graphs, unlabeled_graphs, labels = client_batch_data[client]
                    if train_graphs:
                        # 本地训练
                        updated_model = local_update_fedavg(
                            client, local_model, train_graphs, labels,
                            num_epochs, batch_size, learning_rate
                        )
                        local_weights.append(updated_model.state_dict())
                
                local_weights.append(updated_model.state_dict())

            if local_weights:
                # 聚合权重
                global_weights = average_weights(local_weights)
                model.load_state_dict(global_weights)
                print(f"Aggregated {len(local_weights)} local models")
            # 评估模型并当前轮次计算F1分数
            predictions, all_labels = [], []
            model.eval()
            with torch.no_grad():
                for client in clients:
                    if client < len(val_data):  # 添加索引检查
                        val_graphs, _, labels = val_data[client]
                        
                        for graph, label in zip(val_graphs, labels):
                            # graph = graph.to(device)  # 确保图在设备上
                            out = model(graph.x, graph.edge_index)
                            out = torch.mean(out, dim=0, keepdim=True)
                            # out = torch.max(out, dim=0,keepdim=True)  # 最大池化保留显著特征
                            predictions.append(out)  # 这里是张量
                            all_labels.append(label)  # 确保标签在设备上
            
             # 检查 all_labels 是否为空
            if all_labels:
                # 将 predictions 和 all_labels 转换为张量
                print(len(predictions))
                predictions = torch.cat(predictions, dim=0)  # 合并所有预测
                all_labels = torch.stack(all_labels)  # 合并所有标签，确保是张量
                
                
                # 打印长度以调试
                print(f"预测数量: {predictions.size(0)}, 标签数量: {all_labels.size(0)}")
                
                if predictions.size(0) != all_labels.size(0):
                    print("警告: 预测和标签数量不一致！")
                    continue
                
                f1 = calculate_f1_score(predictions, all_labels)
                f1_scores.append(f1)
                print(f"Round {round + 1} F1 Score: {f1:.4f}")  # 打印每轮的 F1 分数
            else:
                print("警告: all_labels 为空，无法计算 F1 分数。")
        # 评估模型在之前所有批次数据上的性能
        all_hist_predictions = []
        all_hist_labels = []

        # with torch.no_grad():
        #     for prev_time_index in range(time_index + 1):
        #         prev_client_batch_data = val_data[prev_time_index] # 这个批次之前的每一个batch 拿出来算平均
        #         for k in clients:
        #             if k < len(prev_client_batch_data):
        #                 val_graphs, _, labels = prev_client_batch_data[k]
        #                 if val_graphs:
        #                     for graph, label in zip(val_graphs, labels):
        #                         out = model(graph.x, graph.edge_index)
        #                         out = torch.mean(out, dim=0, keepdim=True)
        #                         all_hist_predictions.append(out)
        #                         all_hist_labels.append(label)

        if all_hist_predictions:
            predictions = torch.cat(all_hist_predictions, dim=0)
            labels = torch.stack(all_hist_labels)
            hist_f1 = calculate_f1_score(predictions, labels)
            all_batch_f1_scores.append(hist_f1)
            print(f"F1 Score on all batches after training batch {time_index + 1}: {hist_f1:.4f}")

    # 保存所有批次训练后的F1分数到CSV文件
    save_all_batches_f1(all_batch_f1_scores, model_name, model_type)
    save_f1(f1_scores, model_name, model_type)
    
    return model, f1_scores

        
def fedsem_ft(
    model_type: str,
    clients: List[int],
    model: nn.Module,
    train_data: List[Tuple[List[Data], List[Data], torch.Tensor]],
    val_data: List[Tuple[List[Data], List[Data], torch.Tensor]],
    num_rounds: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    lambda_cr: float,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    model_name: str
) -> Tuple[nn.Module, List[float]]:
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    
    f1_scores = []
    time_step = len(train_data[0])
    processed_data, val_data = increment_data_transform(time_step,train_data,val_data)
    for time_index, (train_data, val_data) in enumerate(zip(processed_data, val_data)):
            
        for round in range(num_rounds):
            local_weights = []
            
            for client in clients:
                # 创建新的本地模型实例并传递所需参数
                local_model = model.__class__(input_dim, hidden_dim, output_dim)
                local_model.load_state_dict(model.state_dict())
                
                # 获取客户端数据
                train_graphs, unlabeled_graphs, labels = train_data[client]
                # 将数据移至设备
                # train_graphs = [g.to(device) for g in train_graphs if g is not None]
                # unlabeled_graphs = [g.to(device) for g in unlabeled_graphs if g is not None]
                # labels = labels.to(device)
                
                # 本地训练
                updated_model = local_update_fedsem_ft(
                    client, local_model, train_graphs, unlabeled_graphs, labels,
                    num_epochs, batch_size, learning_rate
                )
                local_weights.append(updated_model.state_dict())
            
            # 聚合权重
            global_weights = average_weights(local_weights)
            model.load_state_dict(global_weights)
            
            # 评估模型并计算F1分数
            predictions, all_labels = [], []
            model.eval()
            with torch.no_grad():
                for client in clients:
                    if client < len(val_data):  # 添加索引检查
                        val_graphs, _, labels = val_data[client]
                        for graph, label in zip(val_graphs, labels):
                            # graph = graph.to(device)  # 确保图在设备上
                            out = model(graph.x, graph.edge_index)
                            out = torch.mean(out, dim=0, keepdim=True)
                            # out = torch.max(out, dim=0,keepdim = True)  # 最大池化保留显著特征
                            predictions.append(out)  # 这里是张量
                            all_labels.append(label)
            
             # 检查 all_labels 是否为空
            if all_labels:
                # 将 predictions 和 all_labels 转换为张量
                print(len(predictions))
                predictions = torch.cat(predictions, dim=0)  # 合并所有预测
                all_labels = torch.stack(all_labels)  # 合并所有标签，确保是张量
                
                
                # 打印长度以调试
                print(f"预测数量: {predictions.size(0)}, 标签数量: {all_labels.size(0)}")
                
                if predictions.size(0) != all_labels.size(0):
                    print("警告: 预测和标签数量不一致！")
                    continue
                
                f1 = calculate_f1_score(predictions, all_labels)
                f1_scores.append(f1)
                print(f"Round {round + 1} F1 Score: {f1:.4f}")  # 打印每轮的 F1 分数
            else:
                print("警告: all_labels 为空，无法计算 F1 分数。")
            
    save_f1(f1_scores, model_name, model_type)
    
    return model, f1_scores
def increment_fedsem_ft(
    # time_step: int,
    model_type: str,
    clients: List[int],
    model: nn.Module,
    train_data: List[Tuple[List[Data], List[Data], torch.Tensor]],
    val_data: List[Tuple[List[Data], List[Data], torch.Tensor]],
    num_rounds: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    lambda_cr: float,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    model_name: str
) -> Tuple[nn.Module, List[float]]:
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    
    f1_scores = []
    all_batch_f1_scores =[]
    time_step = len(train_data[0])
    processed_data, val_data = increment_data_transform(time_step,train_data,val_data)
    for time_index, (client_batch_data, val_data) in enumerate(zip(processed_data, val_data)):
        print(f"Training on the collected data from {time_index+1}/{len(train_data)+1} time ")
        for round in range(num_rounds):
            local_weights = []
            print(f"for each time batch, Communication Round {round + 1}/{num_rounds}")
            for client in clients:
                print('处理对象:', client)
                if client >= len(client_batch_data):  # 添加索引检查
                    print(f"警告: 客户端 {client} 超出训练数据范围")
                    continue
                
                # 创建新的本地模型实例并传递所需参数
                local_model = model.__class__(input_dim, hidden_dim, output_dim)
                local_model.load_state_dict(model.state_dict())
                # print('训练数据:',train_data)
            
                # 获取客户端数据
                # train_graphs, unlabeled_graphs, labels = train_data[client]
                if client < len(client_batch_data):
                    print('now k=', client, 'client_batch_data=', time_index)
                    train_graphs, unlabeled_graphs, labels = client_batch_data[client]
                    if train_graphs:
                        # 本地训练
                        updated_model = local_update_fedsem_ft(
                            client, local_model, train_graphs, unlabeled_graphs, labels,
                            num_epochs, batch_size, learning_rate
                        )
                        local_weights.append(updated_model.state_dict())
                
            
                local_weights.append(updated_model.state_dict())

            if local_weights:
                # 聚合权重
                global_weights = average_weights(local_weights)
                model.load_state_dict(global_weights)
                print(f"Aggregated {len(local_weights)} local models")
            # 评估模型并当前轮次计算F1分数
            predictions, all_labels = [], []
            model.eval()
            with torch.no_grad():
                for client in clients:
                    if client < len(val_data):  # 添加索引检查
                        val_graphs, _, labels = val_data[client]
                        for graph, label in zip(val_graphs, labels):
                            # graph = graph.to(device)  # 确保图在设备上
                            out = model(graph.x, graph.edge_index)
                            out = torch.mean(out, dim=0, keepdim=True)
                            # out = torch.max(out, dim=0,keepdim=True)  # 最大池化保留显著特征
                            predictions.append(out)  # 这里是张量
                            all_labels.append(label)  # 确保标签在设备上
            
             # 检查 all_labels 是否为空
            if all_labels:
                # 将 predictions 和 all_labels 转换为张量
                print(len(predictions))
                predictions = torch.cat(predictions, dim=0)  # 合并所有预测
                all_labels = torch.stack(all_labels)  # 合并所有标签，确保是张量
                
                
                # 打印长度以调试
                print(f"预测数量: {predictions.size(0)}, 标签数量: {all_labels.size(0)}")
                
                if predictions.size(0) != all_labels.size(0):
                    print("警告: 预测和标签数量不一致！")
                    continue
                
                f1 = calculate_f1_score(predictions, all_labels)
                f1_scores.append(f1)
                print(f"Round {round + 1} F1 Score: {f1:.4f}")  # 打印每轮的 F1 分数
            else:
                print("警告: all_labels 为空，无法计算 F1 分数。")
        # 评估模型在之前所有批次数据上的性能
        all_hist_predictions = []
        all_hist_labels = []

        # with torch.no_grad():
        #     for prev_time_index in range(time_index + 1):
        #         prev_client_batch_data = val_data[prev_time_index] # 这个批次之前的每一个batch 拿出来算平均
        #         for k in clients:
        #             if k < len(prev_client_batch_data):
        #                 val_graphs, _, labels = prev_client_batch_data[k]
        #                 if val_graphs:
        #                     for graph, label in zip(val_graphs, labels):
        #                         out = model(graph.x, graph.edge_index)
        #                         out = torch.mean(out, dim=0, keepdim=True)
        #                         all_hist_predictions.append(out)
        #                         all_hist_labels.append(label)

        if all_hist_predictions:
            predictions = torch.cat(all_hist_predictions, dim=0)
            labels = torch.stack(all_hist_labels)
            hist_f1 = calculate_f1_score(predictions, labels)
            all_batch_f1_scores.append(hist_f1)
            print(f"F1 Score on all batches after training batch {time_index + 1}: {hist_f1:.4f}")

    # 保存所有批次训练后的F1分数到CSV文件
    save_all_batches_f1(all_batch_f1_scores, model_name, model_type)
    save_f1(f1_scores, model_name, model_type)
    
    return model, f1_scores

def fedmatch_ft(
    model_type: str,
    clients: List[int],
    model: nn.Module,
    train_data: List[Tuple[List[Data], List[Data], torch.Tensor]],
    val_data: List[Tuple[List[Data], List[Data], torch.Tensor]],
    num_rounds: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    lambda_cr: float,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    model_name: str
) -> Tuple[nn.Module, List[float]]:
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    
    f1_scores = []
    time_step = len(train_data[0])
    processed_data, val_data = increment_data_transform(time_step,train_data,val_data)
    for time_index, (train_data, val_data) in enumerate(zip(processed_data, val_data)):
        for round in range(num_rounds):
            local_weights = []
            
            for client in clients:
                # 创建新的本地模型实例并传递所需参数
                local_model = model.__class__(input_dim, hidden_dim, output_dim)
                local_model.load_state_dict(model.state_dict())
                
                # 获取客户端数据
                train_graphs, unlabeled_graphs, labels = train_data[client]
                # 将数据移至设备
                # train_graphs = [g.to(device) for g in train_graphs if g is not None]
                # unlabeled_graphs = [g.to(device) for g in unlabeled_graphs if g is not None]
                # labels = labels.to(device)
                
                # 本地训练
                updated_model = local_update_fedmatch_ft(
                    client, local_model, train_graphs, unlabeled_graphs, labels,
                    num_epochs, batch_size, learning_rate, lambda_cr
                )
                local_weights.append(updated_model.state_dict())
            
            # 聚合权重
            global_weights = average_weights(local_weights)
            model.load_state_dict(global_weights)
            
            # 评估模型并计算F1分数
            predictions, all_labels = [], []
            model.eval()
            with torch.no_grad():
                for client in clients:
                    if client < len(train_data):  # 添加索引检查
                        val_graphs, _, labels = train_data[client]
                        for graph, label in zip(val_graphs, labels):
                            # graph = graph.to(device)  # 确保图在设备上
                            out = model(graph.x, graph.edge_index)
                            out = torch.mean(out, dim=0, keepdim=True)
                            # out = torch.max(out, dim=0，keepdim=True)  # 最大池化保留显著特征
                            predictions.append(out)  # 这里是张量
                            all_labels.append(label)
            
             # 检查 all_labels 是否为空
            if all_labels:
                # 将 predictions 和 all_labels 转换为张量
                print(len(predictions))
                predictions = torch.cat(predictions, dim=0)  # 合并所有预测
                all_labels = torch.stack(all_labels)  # 合并所有标签，确保是张量
                
                
                # 打印长度以调试
                print(f"预测数量: {predictions.size(0)}, 标签数量: {all_labels.size(0)}")
                
                if predictions.size(0) != all_labels.size(0):
                    print("警告: 预测和标签数量不一致！")
                    continue
                
                f1 = calculate_f1_score(predictions, all_labels)
                f1_scores.append(f1)
                print(f"Round {round + 1} F1 Score: {f1:.4f}")  # 打印每轮的 F1 分数
            else:
                print("警告: all_labels 为空，无法计算 F1 分数。")
    save_f1(f1_scores, model_name, model_type)
    
    return model, f1_scores

def increment_fedmatch_ft(
    # time_step: int,
    model_type: str,
    clients: List[int],
    model: nn.Module,
    train_data: List[Tuple[List[Data], List[Data], torch.Tensor]],
    val_data: List[Tuple[List[Data], List[Data], torch.Tensor]],
    num_rounds: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    lambda_cr: float,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    model_name: str
) -> Tuple[nn.Module, List[float]]:
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    
    f1_scores = []
    all_batch_f1_scores =[]
    time_step = len(train_data[0])
    processed_data, val_data = increment_data_transform(time_step,train_data,val_data)
    for time_index, (client_batch_data, val_data) in enumerate(zip(processed_data, val_data)):
        print(f"Training on the collected data from {time_index+1}/{len(train_data)+1} time ")
        for round in range(num_rounds):
            local_weights = []
            print(f"for each time batch, Communication Round {round + 1}/{num_rounds}")
            for client in clients:
                print('处理对象:', client)
                if client >= len(client_batch_data):  # 添加索引检查
                    print(f"警告: 客户端 {client} 超出训练数据范围")
                    continue
                
                # 创建新的本地模型实例并传递所需参数
                local_model = model.__class__(input_dim, hidden_dim, output_dim)
                local_model.load_state_dict(model.state_dict())
                # print('训练数据:',train_data)
            
                # 获取客户端数据
                # train_graphs, unlabeled_graphs, labels = train_data[client]
                if client < len(client_batch_data):
                    print('now k=', client, 'client_batch_data=', time_index)
                    train_graphs, unlabeled_graphs, labels = client_batch_data[client]
                    if train_graphs:
                        # 本地训练
                        updated_model = local_update_fedmatch_ft(
                            client, local_model, train_graphs, unlabeled_graphs, labels,
                            num_epochs, batch_size, learning_rate,lambda_cr
                        )
                        local_weights.append(updated_model.state_dict())
                
            
                local_weights.append(updated_model.state_dict())

            if local_weights:
                # 聚合权重
                global_weights = average_weights(local_weights)
                model.load_state_dict(global_weights)
                print(f"Aggregated {len(local_weights)} local models")
            # 评估模型并当前轮次计算F1分数
            predictions, all_labels = [], []
            model.eval()
            with torch.no_grad():
                for client in clients:
                    if client < len(client_batch_data):  # 添加索引检查
                        val_graphs, _, labels = client_batch_data[client]
                        for graph, label in zip(val_graphs, labels):
                            # graph = graph.to(device)  # 确保图在设备上
                            out = model(graph.x, graph.edge_index)
                            out = torch.mean(out, dim=0, keepdim=True)
                            # out = torch.max(out, dim=0,keepdim=True)  # 最大池化保留显著特征
                            predictions.append(out)  # 这里是张量
                            all_labels.append(label)  # 确保标签在设备上
            
             # 检查 all_labels 是否为空
            if all_labels:
                # 将 predictions 和 all_labels 转换为张量
                print(len(predictions))
                predictions = torch.cat(predictions, dim=0)  # 合并所有预测
                all_labels = torch.stack(all_labels)  # 合并所有标签，确保是张量
                
                
                # 打印长度以调试
                print(f"预测数量: {predictions.size(0)}, 标签数量: {all_labels.size(0)}")
                
                if predictions.size(0) != all_labels.size(0):
                    print("警告: 预测和标签数量不一致！")
                    continue
                
                f1 = calculate_f1_score(predictions, all_labels)
                f1_scores.append(f1)
                print(f"Round {round + 1} F1 Score: {f1:.4f}")  # 打印每轮的 F1 分数
            else:
                print("警告: all_labels 为空，无法计算 F1 分数。")
        # 评估模型在之前所有批次数据上的性能
        all_hist_predictions = []
        all_hist_labels = []

        # with torch.no_grad():
        #     for prev_time_index in range(time_index + 1):
        #         prev_client_batch_data = val_data[prev_time_index] # 这个批次之前的每一个batch 拿出来算平均
        #         for k in clients:
        #             if k < len(prev_client_batch_data):
        #                 val_graphs, _, labels = prev_client_batch_data[k]
        #                 if val_graphs:
        #                     for graph, label in zip(val_graphs, labels):
        #                         out = model(graph.x, graph.edge_index)
        #                         out = torch.mean(out, dim=0, keepdim=True)
        #                         all_hist_predictions.append(out)
        #                         all_hist_labels.append(label)

        if all_hist_predictions:
            predictions = torch.cat(all_hist_predictions, dim=0)
            labels = torch.stack(all_hist_labels)
            hist_f1 = calculate_f1_score(predictions, labels)
            all_batch_f1_scores.append(hist_f1)
            print(f"F1 Score on all batches after training batch {time_index + 1}: {hist_f1:.4f}")

    # 保存所有批次训练后的F1分数到CSV文件
    save_all_batches_f1(all_batch_f1_scores, model_name, model_type)
    save_f1(f1_scores, model_name, model_type)
    
    return model, f1_scores

def flwf(
    model_type: str,
    clients: List[int],
    model: torch.nn.Module,
    train_data: List[Tuple[List[Data], List[Data], torch.Tensor]],
    val_data: List[Tuple[List[Data], List[Data], torch.Tensor]],
    num_rounds: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    lambda_cr: float,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    model_name: str
) -> Tuple[torch.nn.Module, List[float]]:
    
    f1_scores = []
    old_model = model.__class__(input_dim, hidden_dim, output_dim)
    old_model.load_state_dict(model.state_dict())
    time_step = len(train_data[0])
    processed_data, val_data = increment_data_transform(time_step,train_data,val_data)
    for time_index, (client_batch_data, val_data) in enumerate(zip(processed_data, val_data)):
        print(f"Training on the collected data from {time_index+1}/{len(train_data)+1} time ")
        for round_num in range(num_rounds):
            local_weights = []
            for client in clients:
                # 创建新的本地模型实例并传递所需参数
                local_model = model.__class__(input_dim, hidden_dim, output_dim)
                local_model.load_state_dict(model.state_dict())
                train_graphs, unlabeled_graphs, labels = client_batch_data[client]
                
                # 本地训练
                updated_model = local_update_flwf(
                    client, local_model, old_model, train_graphs, labels,
                    num_epochs, batch_size, learning_rate, lambda_distill=lambda_cr
                )
                local_weights.append(updated_model.state_dict())
            # 聚合权重
            global_weights = average_weights(local_weights)
            model.load_state_dict(global_weights)
            # 更新旧模型
            old_model.load_state_dict(model.state_dict())
            # 评估模型并计算F1分数
            predictions, all_labels = [], []
            model.eval()
            with torch.no_grad():
                for client in clients:
                    if client < len(val_data):
                        val_graphs, _, labels = val_data[client]
                        for graph, label in zip(val_graphs, labels):
                            out = model(graph.x, graph.edge_index)
                            out = torch.mean(out, dim=0, keepdim=True)
                            # out = torch.max(out, dim=0，keepdim=True)  # 最大池化保留显著特征
                            predictions.append(out)
                            all_labels.append(label)
            # 将 predictions 和 all_labels 转换为张量
            if predictions:
                predictions = torch.cat(predictions, dim=0)
                all_labels = torch.stack(all_labels)
                # 打印长度以调试
                print(f"预测数量: {predictions.size(0)}, 标签数量: {all_labels.size(0)}")
                if predictions.size(0) != all_labels.size(0):
                    print("警告: 预测和标签数量不一致！")
                    continue
                f1 = calculate_f1_score(predictions, all_labels)
                f1_scores.append(f1)
            else:
                print("警告: 没有有效的预测结果！")
    save_f1(f1_scores, model_name, model_type)
    return model, f1_scores
def increment_flwf(
    # time_step: int,
    model_type: str,
    clients: List[int],
    model: nn.Module,
    train_data: List[Tuple[List[Data], List[Data], torch.Tensor]],
    val_data: List[Tuple[List[Data], List[Data], torch.Tensor]],
    num_rounds: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    lambda_cr: float,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    model_name: str
) -> Tuple[nn.Module, List[float]]:
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    
    f1_scores = []
    all_batch_f1_scores =[]
    time_step = len(train_data[0])
    processed_data, val_data = increment_data_transform(time_step,train_data,val_data)
    old_model = model.__class__(input_dim, hidden_dim, output_dim)
    old_model.load_state_dict(model.state_dict())
    for time_index, (client_batch_data, val_data) in enumerate(zip(processed_data, val_data)):
        print(f"Training on the collected data from {time_index+1}/{len(train_data)+1} time ")
        for round in range(num_rounds):
            local_weights = []
            print(f"for each time batch, Communication Round {round + 1}/{num_rounds}")
            for client in clients:
                print('处理对象:', client)
                if client >= len(train_data):  # 添加索引检查
                    print(f"警告: 客户端 {client} 超出训练数据范围")
                    continue
                
                # 创建新的本地模型实例并传递所需参数
                local_model = model.__class__(input_dim, hidden_dim, output_dim)
                local_model.load_state_dict(model.state_dict())
                # 更新旧模型
                old_model.load_state_dict(model.state_dict())
      
            
                # 获取客户端数据
                # train_graphs, unlabeled_graphs, labels = train_data[client]
                if client < len(client_batch_data):
                    print('now k=', client, 'client_batch_data=', time_index)
                    train_graphs, unlabeled_graphs, labels = client_batch_data[client]
                    if train_graphs:
                        # 本地训练
                        updated_model = local_update_flwf(
                            client, local_model,old_model, train_graphs, labels,
                            num_epochs, batch_size, learning_rate,lambda_distill=lambda_cr
                        )
                        local_weights.append(updated_model.state_dict())
                
            
                local_weights.append(updated_model.state_dict())

            if local_weights:
                # 聚合权重
                global_weights = average_weights(local_weights)
                model.load_state_dict(global_weights)
                print(f"Aggregated {len(local_weights)} local models")
            # 评估模型并当前轮次计算F1分数
            predictions, all_labels = [], []
            model.eval()
            with torch.no_grad():
                for client in clients:
                    if client < len(client_batch_data):  # 添加索引检查
                        val_graphs, _, labels = client_batch_data[client]
                        for graph, label in zip(val_graphs, labels):
                            # graph = graph.to(device)  # 确保图在设备上
                            out = model(graph.x, graph.edge_index)
                            out = torch.mean(out, dim=0, keepdim=True)
                            # out = torch.max(out, dim=0,keepdim=True)  # 最大池化保留显著特征
                            predictions.append(out)  # 这里是张量
                            all_labels.append(label)  # 确保标签在设备上
            
            # 将 predictions 和 all_labels 转换为张量
            predictions = torch.cat(predictions, dim=0)  # 合并所有预测
            all_labels = torch.stack(all_labels)# 确保标签在
            
            # 打印长度以调试
            print(f"预测数量: {predictions.size(0)}, 标签数量: {all_labels.size(0)}")
            
            if predictions.size(0) != all_labels.size(0):
                print("警告: 预测和标签数量不一致！")
                continue
            
            f1 = calculate_f1_score(predictions, all_labels)
            f1_scores.append(f1)
        # 评估模型在之前所有批次数据上的性能
        all_hist_predictions = []
        all_hist_labels = []

        # with torch.no_grad():
        #     for prev_time_index in range(time_index + 1):
        #         prev_client_batch_data = val_data[prev_time_index] # 这个批次之前的每一个batch 拿出来算平均
        #         for k in clients:
        #             if k < len(prev_client_batch_data):
        #                 val_graphs, _, labels = prev_client_batch_data[k]
        #                 if val_graphs:
        #                     for graph, label in zip(val_graphs, labels):
        #                         out = model(graph.x, graph.edge_index)
        #                         out = torch.mean(out, dim=0, keepdim=True)
        #                         all_hist_predictions.append(out)
        #                         all_hist_labels.append(label)

        if all_hist_predictions:
            predictions = torch.cat(all_hist_predictions, dim=0)
            labels = torch.stack(all_hist_labels)
            hist_f1 = calculate_f1_score(predictions, labels)
            all_batch_f1_scores.append(hist_f1)
            print(f"F1 Score on all batches after training batch {time_index + 1}: {hist_f1:.4f}")

    # 保存所有批次训练后的F1分数到CSV文件
    save_all_batches_f1(all_batch_f1_scores, model_name, model_type)
    save_f1(f1_scores, model_name, model_type)
    
    return model, f1_scores

def initialize_model(model_type,input_dim, hidden_dim, output_dim):

    if model_type == 'GCN':
        model = GCNModel(input_dim, hidden_dim, output_dim)
    elif model_type == 'GAT':
        model = GATModel(input_dim, hidden_dim, output_dim)
    elif model_type == 'GCN_LSTM':
        model = GCNLSTM(input_dim, hidden_dim, output_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model
def main():
    # 检查CUDA环境
    # check_cuda()
    
    
    # print(expert_batches)
    # print('-'*50)
    # print(expert_vals)
    # sys.exit()
    # 设置参数
    input_dim = 64
    hidden_dim = 128
    output_dim = 2
    num_epochs = 15  # 论文里要求是150轮
    batch_size = 256  # 256个batch size


    learning_rate = 1e-5 # 1e-5
    lambda_cr = 1e-4 # 1e-5 一致性加大看看会如何
    alpha = 0.5
    
    
    #  GCN, GAT, GCN_LSTM
    model_type = 'GCN'

   
    expert_model = initialize_model(
        model_type = model_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    
    # 训练专家模型（前40个通信轮次）
    num_rounds_expert = 10
    num_clients = 60 # 60个用户
    client_perserver = 5  # 每个服务器5个用户
  
    
   
     # 准备数据
    expert_batches, expert_vals,apprentice_batches, apprentice_vals,server_nums = prepare_data(client_perserver,num_clients) 
    clients=range(server_nums)   # 确保clients的索引在有效范围内
    # print("expert_batches:", len(expert_batches))
    # print("expert_vals:", len(expert_vals))

    # print("apprentice_batches:", len(apprentice_batches))
    # print("apprentice_vals:", len(apprentice_vals))
    # sys.exit()
    # 初始化模型

    # 进行增量更新（41到120个通信轮次）
    # num_rounds_incremental = 80  # 从41到120共80轮  
   
    communication_round = 10 # 8 batches * 10 = 80 只给我们的模型用
    expert_model, expert_f1_scores = federated_learning(
        model_type = model_type,
        clients=clients,
        model=expert_model,
        train_data=expert_batches,
        val_data = expert_vals,
        num_rounds=num_rounds_expert,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lambda_cr=lambda_cr,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        model_name='FedMH'  # 修改为FedMH
    )
    # sys.exit()
    print('完成专家模型初步训练---FedMH')
    
    apprentice_model = initialize_model(
        model_type = model_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    
    apprentice_model, apprentice_f1_scores = incremental_federated_learning(
        model_type=model_type,
        clients=clients,
        apprentice_model=apprentice_model,
        expert_model=expert_model,
        train_data=apprentice_batches,
        valid_data = apprentice_vals,
        num_rounds=communication_round,
        num_epochs=num_epochs,
        batch_size=batch_size,
       
        learning_rate=learning_rate,
        alpha=alpha,
        lambda_cr=lambda_cr,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        model_name='FedMH_incremental'  # 修改为增量模型名称
    )
    print('完成增量训练---FedMH')
    
    # sys.exit()
    # 初始化模型
    expert_model = initialize_model(
        model_type = model_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    print('正在初始化参数')
    # 训练对比模型（FedAvg）
    print('开始训练对比模型---FedAvg')
    fedavg_model, fedavg_f1_scores = fedavg(
        model_type=model_type,
        clients=clients,
        model=expert_model,
        train_data=expert_batches,
        val_data = expert_vals,
        num_rounds=num_rounds_expert,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lambda_cr=lambda_cr,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        model_name='FedAvg'
    )
    
    # 进行增量更新（41到120个通信轮次）对比模型
    print('开始增量训练对比模型---FedAvg')
    fedavg_model, fedavg_incremental_f1_scores = increment_fedavg(
       
        model_type=model_type,
        clients=clients,
        model=fedavg_model,
       
        train_data=apprentice_batches,
        val_data = apprentice_vals,
        num_rounds=communication_round,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
  
        lambda_cr=lambda_cr,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        model_name='FedAvg_incremental'
    )
    
    
    # 训练对比模型（FedSem_FT）
    print('开始训练对比模型---FedSem_FT')
    # 初始化模型
    expert_model = initialize_model(
        model_type = model_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    print('正在初始化参数')
    fedsem_ft_model, fedsem_ft_f1_scores = fedsem_ft(
        model_type=model_type,
        clients=clients,
        model=expert_model,
        train_data=expert_batches,
        val_data = expert_vals,
        num_rounds=num_rounds_expert,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lambda_cr=lambda_cr,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        model_name='FedSem_FT'
    )
    
    # 进行增量更新（41到120个通信轮次）对比模型
    print('开始增量训练对比模型---FedSem_FT')
    fedsem_ft_model, fedsem_ft_incremental_f1_scores =increment_fedsem_ft(

        model_type=model_type,
        clients=clients,
        model=fedsem_ft_model,
        
        train_data=apprentice_batches,
        val_data = apprentice_vals,
        num_rounds=communication_round,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
  
        lambda_cr=lambda_cr,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        model_name='FedSem_FT_incremental'
    )
    
    # 训练对比模型（FedMatch_FT）
    print('开始训练对比模型---FedMatch_FT')
    # 初始化模型
    expert_model = initialize_model(
        model_type = model_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    print('正在初始化参数')
    fedmatch_ft_model, fedmatch_ft_f1_scores = fedmatch_ft(
        model_type=model_type,
        clients=clients,
        model=expert_model,
        train_data=expert_batches,
        val_data = expert_vals,
        num_rounds=num_rounds_expert,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lambda_cr=lambda_cr,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        model_name='FedMatch_FT'
    )
    
    # 进行增量更新（41到120个通信轮次）对比模型
    print('开始增量训练对比模型---FedMatch_FT')
    fedmatch_ft_model, fedmatch_ft_incremental_f1_scores = increment_fedmatch_ft(
      
        model_type=model_type,
        clients=clients,

        model=fedmatch_ft_model,
        train_data=apprentice_batches,
        val_data = apprentice_vals,
        num_rounds=communication_round,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,

        lambda_cr=lambda_cr,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        model_name='FedMatch_FT_incremental'
    )
    
    # 训练对比模型（FLwF）
    print('开始训练对比模型---FLwF')
    # 初始化模型
    expert_model = initialize_model(
        model_type = model_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    print('正在初始化参数')
    flwf_model, flwf_f1_scores = flwf(
        model_type=model_type,
        clients=clients,
        model=expert_model,

        train_data=expert_batches,
        val_data = expert_vals,
        num_rounds=num_rounds_expert,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lambda_cr=lambda_cr,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        model_name='FLwF'
    )
    
    # 进行增量更新（41到120个通信轮次）对比模型
    print('开始增量训练对比模型---FLwF')
    flwf_model, flwf_incremental_f1_scores = increment_flwf(
     
        model_type=model_type,
        clients=clients,
        
        model=flwf_model,
        train_data=apprentice_batches,
        val_data=apprentice_vals,
        num_rounds=communication_round,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,

        lambda_cr=lambda_cr,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        model_name='FLwF_incremental'
    )
    
 
    try:
        import pandas as pd
        combined_df = pd.DataFrame({
            'Round': range(1, len(expert_f1_scores) + 1),
            'FedMH': expert_f1_scores,
            'FedMH_incremental': apprentice_f1_scores,  # 增量更新的F1分数
            'FedAvg': fedavg_f1_scores,
            'FedAvg_incremental': fedavg_incremental_f1_scores,
            'FedSem_FT': fedsem_ft_f1_scores,
            'FedSem_FT_incremental': fedsem_ft_incremental_f1_scores,
            'FedMatch_FT': fedmatch_ft_f1_scores,
            'FedMatch_FT_incremental': fedmatch_ft_incremental_f1_scores,
            'FLwF': flwf_f1_scores,
            'FLwF_incremental': flwf_incremental_f1_scores,
            
        })
        combined_df.to_csv('f1_score_comparison.csv', index=False)
        print("已生成完整的F1分数比较数据文件")
    except Exception as e:
        print(f"合并F1分数时出错: {str(e)}")

if __name__ == "__main__":
    main()
