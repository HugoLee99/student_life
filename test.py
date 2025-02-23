def incremental_federated_learning(
    model_type: str,
    clients: List[int],
    apprentice_model: nn.Module,
    expert_model: nn.Module,
    train_data: List[List[Tuple[List[Data], List[Data], torch.Tensor]]],
    num_rounds: int,
    num_epochs: int,
    batch_size: int,
    time_step: int,
    learning_rate: float,
    alpha: float,
    lambda_cr: float,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    model_name: str  # 添加模型名称参数
) -> Tuple[nn.Module, List[float]]:  # 修改返回类型
    global_weights = apprentice_model.state_dict()
    # 初始化F1分数列表，用于记录每个批次训练后在所有批次数据上的F1分数
    all_batch_f1_scores = []
    # 输入是[client nums, batch nums, 三元组]，输出变成一个二维列表[batch nums, client nums, 三元组]
    processed_data = increment_data_transform(time_step, train_data)
    for time_index, client_batch_data in enumerate(processed_data):
        print(f"Training on the collected data from {time_index + 1}/{len(train_data) + 1} time ")
        for round in range(num_rounds):
            local_weights = []
            print(f"for each time batch, Communication Round {round + 1}/{num_rounds}")
            # k = 0,1,2,3...
            for k in clients:
                local_model = type(apprentice_model)(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim
                )
                local_model.load_state_dict(global_weights)

                if k < len(client_batch_data):
                    print('now k=', k, 'client_batch_data=', len(client_batch_data))
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
                        if k < len(client_batch_data):
                            train_graphs, _, labels = client_batch_data[k]
                            if train_graphs:
                                for graph, label in zip(train_graphs, labels):
                                    out = apprentice_model(graph.x, graph.edge_index)
                                    out = torch.mean(out, dim=0, keepdim=True)
                                    all_predictions.append(out)
                                    all_labels.append(label)

                if all_predictions:
                    predictions = torch.cat(all_predictions, dim=0)
                    labels = torch.stack(all_labels)
                    f1 = calculate_f1_score(predictions, labels)
                    print(f"Batch {time_index + 1}, communication Round {round + 1} F1-Score: {f1:.4f}")
            except Exception as e:
                print(f"评估模型时出错: {str(e)}")

        # 评估模型在之前所有批次数据上的性能
        all_hist_predictions = []
        all_hist_labels = []
        apprentice_model.eval()
        with torch.no_grad():
            for prev_time_index in range(time_index + 1):
                prev_client_batch_data = processed_data[prev_time_index] # 这个批次之前的每一个batch 拿出来算平均
                for k in clients:
                    if k < len(prev_client_batch_data):
                        train_graphs, _, labels = prev_client_batch_data[k]
                        if train_graphs:
                            for graph, label in zip(train_graphs, labels):
                                out = apprentice_model(graph.x, graph.edge_index)
                                out = torch.mean(out, dim=0, keepdim=True)
                                all_hist_predictions.append(out)
                                all_hist_labels.append(label)

        if all_hist_predictions:
            predictions = torch.cat(all_hist_predictions, dim=0)
            labels = torch.stack(all_hist_labels)
            hist_f1 = calculate_f1_score(predictions, labels)
            all_batch_f1_scores.append(hist_f1)
            print(f"F1 Score on all batches after training batch {time_index + 1}: {hist_f1:.4f}")

    # 保存所有批次训练后的F1分数到CSV文件
    save_all_batch_f1(all_batch_f1_scores, model_name, model_type)

    return apprentice_model, all_batch_f1_scores