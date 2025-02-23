import torch
import torch.nn as nn
from typing import List
from torch_geometric.data import Data

import torch.nn.functional as F
from typing import List
from torch_geometric.data import Data
import random
def local_update_cr(
    k: int,
    model: nn.Module,
    train_graphs: List[Data],
    unlabeled_graphs: List[Data],
    labels: torch.Tensor,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    lambda_cr: float
) -> nn.Module:
    """一致性正则化的本地更新"""
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    epoch_losses = []
    
    # 过滤掉无效的无标签图
    valid_unlabeled_graphs = []
    if unlabeled_graphs:
        for graph in unlabeled_graphs:
            if hasattr(graph, 'x') and hasattr(graph, 'edge_index') and \
               graph.x.size(0) > 0 and graph.edge_index.size(1) > 0:
                valid_unlabeled_graphs.append(graph)
    
    print(f"Client {k}: {len(valid_unlabeled_graphs)} 个有效的无标签图在{len(unlabeled_graphs)}中")

    
    for epoch in range(num_epochs):
        total_loss = 0
        valid_batches = 0
        
        for i in range(0, len(train_graphs), batch_size):
            batch_graphs = train_graphs[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            batch_loss = 0
            valid_count = 0
            
            for graph, label in zip(batch_graphs, batch_labels):
                # 检查图是否有效
                if not hasattr(graph, 'edge_index') or graph.edge_index.numel() == 0:
                    num_nodes = graph.x.size(0)
                    self_loops = torch.arange(num_nodes, dtype=torch.long)
                    graph.edge_index = torch.stack([self_loops, self_loops], dim=0)
                
                try:
                    # 确保图在CPU上
                    out = model(graph.x, graph.edge_index)  # 一张图的输出
                    out = torch.mean(out, dim=0, keepdim=True)
                    # out = torch.max(out, dim=0,keepdim=True)  # 最大池化保留显著特征
                    batch_loss += criterion(out, label.unsqueeze(0))
                    valid_count += 1
                except Exception as e:
                    print(f"处理训练图时出错: {str(e)}")
                    continue
            
            if valid_count > 0:
                loss = batch_loss / valid_count
                
                # 一致性正则化
                if valid_unlabeled_graphs:
                    cr_loss = 0
                    cr_count = 0
                    
                    # 随机选择一部分无标签图进行一致性正则化
                    batch_unlabeled = random.sample(
                        valid_unlabeled_graphs, 
                        min(batch_size, len(valid_unlabeled_graphs))
                    )
                    
                    for graph in batch_unlabeled:
                        try:
                            # 确保无标签图在CPU上
                            out = model(graph.x, graph.edge_index)
                            out = torch.mean(out, dim=0, keepdim=True)
                            # out = torch.max(out, dim=0,keepdim=True)  # 最大池化保留显著特征
                            
                            # 扰动后的输出
                            perturbed_x = random_perturbation(graph.x)
                            out_perturbed = model(perturbed_x, graph.edge_index)
                            out_perturbed = torch.mean(out_perturbed, dim=0, keepdim=True)
                            
                            cr_loss += F.kl_div(
                                F.log_softmax(out, dim=1),
                                F.softmax(out_perturbed, dim=1),
                                reduction='batchmean'
                            )
                            cr_count += 1
                        except Exception as e:
                            continue
                    
                    if cr_count > 0:
                        loss += lambda_cr * (cr_loss / cr_count)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                valid_batches += 1
        
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            epoch_losses.append(avg_loss)
            if epoch % 10 == 0:  # 每10个epoch打印一次
                print(f"Client {k}, Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model

def incremental_local_update_cr(
    k: int,
    apprentice_model: nn.Module,
    expert_model: nn.Module,
    train_graphs: List[Data],
    unlabeled_graphs: List[Data],
    labels: torch.Tensor,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    alpha: float,
    lambda_cr: float
) -> nn.Module:
    """增量式一致性正则化的本地更新"""
    # optimizer = torch.optim.SGD(apprentice_model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(apprentice_model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        for i in range(0, len(train_graphs), batch_size):
            batch_graphs = train_graphs[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            batch_loss = 0
            valid_count = 0
            for graph, label in zip(batch_graphs, batch_labels):
                # 检查图是否有效
                if not hasattr(graph, 'edge_index') or graph.edge_index.numel() == 0:
                    num_nodes = graph.x.size(0)
                    self_loops = torch.arange(num_nodes, dtype=torch.long)
                    graph.edge_index = torch.stack([self_loops, self_loops], dim=0)
                try:
                    # 学徒模型预测
                    out = apprentice_model(graph.x, graph.edge_index)
                    out = torch.mean(out, dim=0, keepdim=True)
                    # out = torch.max(out, dim=0,keepdim=True)  # 最大池化保留显著特征
                    # 专家模型预测
                    with torch.no_grad():
                        expert_out = expert_model(graph.x, graph.edge_index)
                        expert_out = torch.mean(expert_out, dim=0, keepdim=True)
                        # out = torch.max(out, dim=0,keepdim=True)  # 最大池化保留显著特征
                    # 计算监督损失和知识蒸馏损失
                    l_ce = (1 - alpha) * criterion(out, label.unsqueeze(0))
                    l_kd = alpha * F.kl_div(
                        F.log_softmax(out, dim=1),
                        F.softmax(expert_out, dim=1),
                        reduction='batchmean'
                    )
                    
                    batch_loss += l_ce + l_kd
                    valid_count += 1
                except Exception as e:
                    print(f"处理图时出错: {str(e)}")
                    print(f"图的信息 - 节点数: {graph.x.size(0)}, 边的形状: {graph.edge_index.shape if hasattr(graph, 'edge_index') else 'No edges'}")
                    continue
            if valid_count > 0:
                loss = batch_loss / valid_count
                # 一致性正则化
                if unlabeled_graphs:
                    cr_loss = 0
                    cr_count = 0
                    for graph in unlabeled_graphs[:batch_size]:
                        # 原始输出
                        out = apprentice_model(graph.x, graph.edge_index)
                        out = torch.mean(out, dim=0, keepdim=True)
                        # out = torch.max(out, dim=0)[0]  # 最大池化保留显著特征
                        # 扰动后的输出
                        perturbed_x = random_perturbation(graph.x)
                        out_perturbed = apprentice_model(perturbed_x, graph.edge_index)
                        out_perturbed = torch.mean(out_perturbed, dim=0, keepdim=True)
                        # out_perturbed = torch.max(out_perturbed, dim=0,keepdim=True)  # 最大池化保留显著特征
                        # print(f"out_perturbed shape: {out_perturbed.shape}")
                        # 计算一致性损失
                        cr_loss += F.kl_div(
                            F.log_softmax(out, dim=1),
                            F.softmax(out_perturbed, dim=1),
                            reduction='batchmean'
                        )
                         # 添加打印语句来检查 out 和 expert_out 的形状
                        # print(f"out shape: {out.shape}")
                        # print(f"out_perturbed shape: {out_perturbed.shape}")
                        cr_count += 1
                    if cr_count > 0:
                        loss += lambda_cr * (cr_loss / cr_count)
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return apprentice_model

def local_update_fedavg(
        
    k: int,
    model: nn.Module,
    train_graphs: List[Data],
    labels: torch.Tensor,
    num_epochs: int,
    batch_size: int,
    learning_rate: float
) -> nn.Module:
    """FedAvg的本地更新"""
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    epoch_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        valid_batches = 0
        
        for i in range(0, len(train_graphs), batch_size):
            batch_graphs = train_graphs[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            batch_loss = 0
            valid_count = 0
            
            for graph, label in zip(batch_graphs, batch_labels):
                # 检查图是否有效
                if not hasattr(graph, 'edge_index') or graph.edge_index.numel() == 0:
                    num_nodes = graph.x.size(0)
                    self_loops = torch.arange(num_nodes, dtype=torch.long)
                    graph.edge_index = torch.stack([self_loops, self_loops], dim=0)
                
                try:
                    # 确保图在CPU上
                    out = model(graph.x, graph.edge_index)  # 一张图的输出
                    out = torch.mean(out, dim=0, keepdim=True)
                    batch_loss += criterion(out, label.unsqueeze(0))
                    valid_count += 1
                except Exception as e:
                    print(f"处理训练图时出错: {str(e)}")
                    continue
            
            if valid_count > 0:
                loss = batch_loss / valid_count
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                valid_batches += 1
        
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            epoch_losses.append(avg_loss)
            if epoch % 10 == 0:  # 每10个epoch打印一次
                print(f"Client {k}, Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model


def local_update_fedsem_ft(
    k: int,
    model: nn.Module,
    train_graphs: List[Data],
    unlabeled_graphs: List[Data],
    labels: torch.Tensor,
    num_epochs: int,
    batch_size: int,
    learning_rate: float
) -> nn.Module:
    """FedSem - FT的本地更新"""
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    epoch_losses = []
    
    # 过滤掉无效的无标签图
    valid_unlabeled_graphs = []
    if unlabeled_graphs:
        for graph in unlabeled_graphs:
            if hasattr(graph, 'x') and hasattr(graph, 'edge_index') and \
               graph.x.size(0) > 0 and graph.edge_index.size(1) > 0:
                valid_unlabeled_graphs.append(graph)
    
    print(f"Client {k}: {len(valid_unlabeled_graphs)} 个有效的无标签图在{len(unlabeled_graphs)}中")
    for epoch in range(num_epochs):
        total_loss = 0
        valid_batches = 0
        
        for i in range(0, len(train_graphs), batch_size):
            batch_graphs = train_graphs[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            batch_loss = 0
            valid_count = 0
            
            for graph, label in zip(batch_graphs, batch_labels):
                # 检查图是否有效
                if not hasattr(graph, 'edge_index') or graph.edge_index.numel() == 0:
                    num_nodes = graph.x.size(0)
                    self_loops = torch.arange(num_nodes, dtype=torch.long)
                    graph.edge_index = torch.stack([self_loops, self_loops], dim=0)
                
                try:
                    # 确保图在CPU上
                    out = model(graph.x, graph.edge_index)  # 一张图的输出
                    out = torch.mean(out, dim=0, keepdim=True)
                    batch_loss += criterion(out, label.unsqueeze(0))
                    valid_count += 1
                except Exception as e:
                    print(f"处理训练图时出错: {str(e)}")
                    continue
            
            # 处理无标签数据，生成伪标签
            if valid_unlabeled_graphs:
                for graph in valid_unlabeled_graphs:
                    try:
                        out = model(graph.x, graph.edge_index)
                        out = torch.mean(out, dim=0, keepdim=True)
                        pseudo_label = torch.argmax(out, dim=1)
                        batch_loss += criterion(out, pseudo_label)
                        valid_count += 1
                    except Exception as e:
                        continue
            
            if valid_count > 0:
                loss = batch_loss / valid_count
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                valid_batches += 1
        
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            epoch_losses.append(avg_loss)
            if epoch % 10 == 0:  # 每10个epoch打印一次
                print(f"Client {k}, Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model

def local_update_fedmatch_ft(
    k: int,
    model: nn.Module,
    train_graphs: List[Data],
    unlabeled_graphs: List[Data],
    labels: torch.Tensor,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    lambda_cr: float
) -> nn.Module:
    """FedMatch - FT的本地更新"""
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    epoch_losses = []
    
    # 过滤掉无效的无标签图
    valid_unlabeled_graphs = []
    if unlabeled_graphs:
        for graph in unlabeled_graphs:
            if hasattr(graph, 'x') and hasattr(graph, 'edge_index') and \
               graph.x.size(0) > 0 and graph.edge_index.size(1) > 0:
                valid_unlabeled_graphs.append(graph)
    
    print(f"Client {k}: {len(valid_unlabeled_graphs)} 个有效的无标签图在{len(unlabeled_graphs)}中")

    for epoch in range(num_epochs):
        total_loss = 0
        valid_batches = 0
        
        for i in range(0, len(train_graphs), batch_size):
            batch_graphs = train_graphs[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            batch_loss = 0
            valid_count = 0
            
            for graph, label in zip(batch_graphs, batch_labels):
                # 检查图是否有效
                if not hasattr(graph, 'edge_index') or graph.edge_index.numel() == 0:
                    num_nodes = graph.x.size(0)
                    self_loops = torch.arange(num_nodes, dtype=torch.long)
                    graph.edge_index = torch.stack([self_loops, self_loops], dim=0)
                
                try:
                    # 确保图在CPU上
                    out = model(graph.x, graph.edge_index)  # 一张图的输出
                    out = torch.mean(out, dim=0, keepdim=True)
                    batch_loss += criterion(out, label.unsqueeze(0))
                    valid_count += 1
                except Exception as e:
                    print(f"处理训练图时出错: {str(e)}")
                    continue
            
            if valid_count > 0:
                labeled_loss = batch_loss / valid_count
                
                # 处理无标签数据，计算一致性损失
                if valid_unlabeled_graphs:
                    cr_loss = 0
                    cr_count = 0
                    
                    # 随机选择一部分无标签图进行一致性正则化
                    batch_unlabeled = random.sample(
                        valid_unlabeled_graphs, 
                        min(batch_size, len(valid_unlabeled_graphs))
                    )
                    
                    for graph in batch_unlabeled:
                        try:
                            # 确保无标签图在CPU上
                            out = model(graph.x, graph.edge_index)
                            out = torch.mean(out, dim=0, keepdim=True)
                            
                            # 扰动后的输出
                            perturbed_x = random_perturbation(graph.x)
                            out_perturbed = model(perturbed_x, graph.edge_index)
                            out_perturbed = torch.mean(out_perturbed, dim=0, keepdim=True)
                            
                            cr_loss += F.kl_div(
                                F.log_softmax(out_perturbed, dim=1),
                                F.softmax(out, dim=1),
                                reduction='batchmean'
                            )
                            cr_count += 1
                        except Exception as e:
                            continue
                    
                    if cr_count > 0:
                        unlabeled_loss = cr_loss / cr_count
                        loss = labeled_loss + lambda_cr * unlabeled_loss
                    else:
                        loss = labeled_loss
                else:
                    loss = labeled_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                valid_batches += 1
        
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            epoch_losses.append(avg_loss)
            if epoch % 10 == 0:  # 每10个epoch打印一次
                print(f"Client {k}, Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model

def random_perturbation(x):
    # 简单的随机扰动示例
    noise = torch.randn_like(x) * 0.1
    return x + noise



def local_update_flwf(
    k: int,
    model: nn.Module,
    old_model: nn.Module,
    train_graphs: List[Data],
    labels: torch.Tensor,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    lambda_distill: float
) -> nn.Module:
    """FLwF的本地更新"""
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    epoch_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        valid_batches = 0
        
        for i in range(0, len(train_graphs), batch_size):
            batch_graphs = train_graphs[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            batch_loss = 0
            valid_count = 0
            
            for graph, label in zip(batch_graphs, batch_labels):
                # 检查图是否有效
                if not hasattr(graph, 'edge_index') or graph.edge_index.numel() == 0:
                    num_nodes = graph.x.size(0)
                    self_loops = torch.arange(num_nodes, dtype=torch.long)
                    graph.edge_index = torch.stack([self_loops, self_loops], dim=0)
                
                try:
                    # 确保图在CPU上
                    out = model(graph.x, graph.edge_index)  # 一张图的输出
                    out = torch.mean(out, dim=0, keepdim=True)
                    batch_loss += criterion(out, label.unsqueeze(0))
                    valid_count += 1
                except Exception as e:
                    print(f"处理训练图时出错: {str(e)}")
                    continue
            
            if valid_count > 0:
                ce_loss = batch_loss / valid_count
                
                # 知识蒸馏损失
                distill_loss = 0
                for graph in train_graphs:
                    if not hasattr(graph, 'edge_index') or graph.edge_index.numel() == 0:
                        num_nodes = graph.x.size(0)
                        self_loops = torch.arange(num_nodes, dtype=torch.long)
                        graph.edge_index = torch.stack([self_loops, self_loops], dim=0)
                    try:
                        old_out = old_model(graph.x, graph.edge_index)
                        old_out = torch.mean(old_out, dim=0, keepdim=True)
                        new_out = model(graph.x, graph.edge_index)
                        new_out = torch.mean(new_out, dim=0, keepdim=True)
                        distill_loss += F.kl_div(
                            F.log_softmax(new_out, dim=1),
                            F.softmax(old_out, dim=1),
                            reduction='batchmean'
                        )
                    except Exception as e:
                        continue
                
                total_loss_batch = ce_loss + lambda_distill * distill_loss
                
                optimizer.zero_grad()
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                valid_batches += 1
        
        if valid_batches > 0:
            avg_loss = total_loss / valid_batches
            epoch_losses.append(avg_loss)
            if epoch % 10 == 0:  # 每10个epoch打印一次
                print(f"Client {k}, Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model