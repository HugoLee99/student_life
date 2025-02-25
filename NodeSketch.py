import numpy as np
import networkx as nx

class NodeSketch:
    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim

    def get_embedding(self, G: nx.Graph, node: str) -> np.ndarray:
        #基于节点的度数和邻居节点的度数
       
        # 示例：使用节点的度数和邻居节点的度数作为特征
        degree = G.degree[node]
        neighbors = list(G.neighbors(node))
        neighbor_degrees = [G.degree[neighbor] for neighbor in neighbors]
        
        # 创建嵌入特征
        embedding = np.zeros(self.embedding_dim)
        embedding[0] = degree
        embedding[1:len(neighbor_degrees)+1] = neighbor_degrees[:self.embedding_dim-1]
        
        return embedding