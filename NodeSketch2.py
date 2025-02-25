import numpy as np
import networkx as nx
from node2vec import Node2Vec

#随机游走
class NodeSketch:
    def __init__(self, embedding_dim=64, walk_length=20, num_walks=50, p=1, q=1):
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.model = None

    def fit(self, G: nx.Graph):
        """训练 Node2Vec 模型"""
        node2vec = Node2Vec(G, dimensions=self.embedding_dim, walk_length=self.walk_length,
                            num_walks=self.num_walks, p=self.p, q=self.q, workers=4)
        self.model = node2vec.fit(window=10, min_count=1, batch_words=4)

    def get_embedding(self, node: str) -> np.ndarray:
        """获取节点的嵌入特征"""
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        return self.model.wv[node]