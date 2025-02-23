
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