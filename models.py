import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data

class GNNBase(nn.Module):
    """基础GNN模型"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, dropout=0.1):
        super(GNNBase, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 特征处理MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Xavier初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

class GCNModel(GNNBase):
    """GCN模型"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, dropout=0.1):
        super(GCNModel, self).__init__(input_dim, hidden_dim, output_dim, dropout)
        
        # GCN层
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        # 特征处理
        x = self.feature_mlp(x)
        
        # GCN层
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        
        # 输出层
        out = self.output_layer(x)
        return out

class GATModel(GNNBase):
    """GAT模型"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, heads=4, dropout=0.1):
        super(GATModel, self).__init__(input_dim, hidden_dim, output_dim, dropout)
        
        # GAT层
        self.conv1 = GATConv(hidden_dim, hidden_dim // heads, heads=heads)
        self.conv2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        # 特征处理
        x = self.feature_mlp(x)
        
        # GAT层
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        
        # 输出层
        out = self.output_layer(x)
        return out

class GCNLSTM(GNNBase):
    """GCN-LSTM模型"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, dropout=0.1):
        super(GCNLSTM, self).__init__(input_dim, hidden_dim, output_dim, dropout)
        
        # GCN层
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, batch_size=1):
        # 特征处理
        x = self.feature_mlp(x)
        
        # GCN层
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        
        # 重塑张量以适应LSTM
        x = x.view(batch_size, -1, self.hidden_dim)
        
        # LSTM层
        x, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        x = x[:, -1, :]
        
        # 输出层
        out = self.output_layer(x)
        return out 