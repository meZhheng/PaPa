import torch
import torch.nn as nn
from random import sample
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn.functional as F

class TimeAwareGCN(nn.Module):
    def __init__(self, config, relPosEncoder=None, absTimeEncoder=None):
        super().__init__()
        self.config = config

        self.embedding_dim = self.config.emb_dim
        self.gcn_hidden_dim = self.config.gcn_hidden_dim
        self.rumor_feature_dim = self.config.d_model

        self.dropout = self.config.gcn_dropout
        self.edge_drop_rate = self.config.gcn_edge_dropout

        self.relPosEncoder = relPosEncoder
        self.absTimeEncoder = absTimeEncoder

        # GCN 谣言检测模块
        self.biGCN = BiGCN(
            self.embedding_dim, 
            self.gcn_hidden_dim, 
            self.rumor_feature_dim,
            dropout = self.dropout
        )

    # 根据输入的任务标识进行前向迭代
    def forward(self, post_feature, edge_index_TD, edge_index_BU, root_index, abs_time, rel_pos):
        '''
        Iutput: 
            - node_token: transformers.BertTokenizer类的输出, 一般是, 未转成句向量
            - edge_index_TD: [[int]. [int]], [2, |E|], 自顶向下的传播树图
            - edge_index_BU: [[int]. [int]], [2, |E|], 自底向上的传播树图
            - root_index: list[int]
        '''
        if self.relPosEncoder is not None and self.absTimeEncoder is not None:

            pos_encoding = self.relPosEncoder(rel_pos)
            timeEncoding = self.absTimeEncoder(abs_time)
            
            X_post = post_feature + pos_encoding + timeEncoding
        
        else:
            X_post = post_feature

        # 采样生成保留边的编号，注意模型在eval模式下不应该dropedge
        if self.training:
            edgeNum = len(edge_index_TD[0])
            savedIndex = sample(range(edgeNum), int(edgeNum * (1 - self.edge_drop_rate)))
            savedIndex = sorted(savedIndex)
            edgeTD = [[edge_index_TD[row][col] for col in savedIndex] for row in range(2)]
            edgeTD = torch.LongTensor(edgeTD)
        else:
            edgeTD = torch.LongTensor(edge_index_TD)
        dataTD = Data(
            x = torch.clone(X_post), 
            edgeIndex = edgeTD.cuda(), 
            rootIndex = root_index
        )

        if self.training:
            edgeNum = len(edge_index_BU[0])
            savedIndex = sample(range(edgeNum), int(edgeNum * (1 - self.edge_drop_rate)))
            savedIndex = sorted(savedIndex)
            edgeBU = [[edge_index_BU[row][col] for col in savedIndex] for row in range(2)]
            edgeBU = torch.LongTensor(edgeBU)
        else:
            edgeBU = torch.LongTensor(edge_index_BU)
        dataBU = Data(
            x = torch.clone(X_post), 
            edgeIndex = edgeBU.cuda(), 
            rootIndex = root_index
        )

        rumorFeature = self.biGCN(dataTD, dataBU)

        return rumorFeature

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim + input_dim, output_dim)
        self.dropout = dropout
    
    def rootEnhancement(self, rootIndex, x):
        enhanceRoot = []
        for i, root_idx in enumerate(rootIndex):
            if root_idx == rootIndex[-1]:
                break
            
            current_post = x[root_idx]
            # 根据索引的差值确定要复制的大小
            size_difference = rootIndex[i+1] - root_idx
            # 将当前 posts 复制 size_difference 次并添加到结果列表中
            for _ in range(size_difference):
                enhanceRoot.append(current_post)

        # 将结果列表转换为张量
        enhanceRoot = torch.stack(enhanceRoot, dim=0)

        return enhanceRoot
    
    def forward(self, data: Data):
        posts, edge_index, rootIndex = data.x, data.edgeIndex, data.rootIndex # posts(n, input_dim), edgeIndex(2, |E|)

        root_index = torch.tensor([posts.shape[0]], dtype=torch.long, device='cuda')
        rootIndex = torch.cat((rootIndex, root_index), dim=0)

        postRoot = self.rootEnhancement(rootIndex, posts)
        conv1Out = self.conv1(posts, edge_index)
        conv1Root = self.rootEnhancement(rootIndex, conv1Out)

        conv2In = torch.cat([postRoot, conv1Out], dim=1)
        conv2In = F.relu(conv2In)
        conv2In = F.dropout(conv2In, training=self.training, p=self.dropout) # BiGCN对于dropout的实现，一次卷积之后随机舍弃一些点

        conv2Out = self.conv2(conv2In, edge_index)
        conv2Out = F.relu(conv2Out)

        feature = torch.add(conv1Root, conv2Out)

        return feature

# BiGCN
class BiGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, convOutput_dim, dropout = 0.5):
        super(BiGCN, self).__init__()
        self.TDGCN = GCN(input_dim, hidden_dim, convOutput_dim, dropout)
        self.BUGCN = GCN(input_dim, hidden_dim, convOutput_dim, dropout)

    def forward(self, dataTD, dataBU):
        TDOut = self.TDGCN(dataTD)
        BUOut = self.BUGCN(dataBU)

        feature = torch.add(TDOut, BUOut)

        return feature