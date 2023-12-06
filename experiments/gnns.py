import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.datasets import QM9
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool
from torch.nn import LayerNorm, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv

import numpy as np



# define the GIN model
class GINNet(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_layers):
        super(GINNet, self).__init__()

        self.conv1 = GINConv(torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
        ), train_eps=True)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
            ), train_eps=True))

        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_add_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
    

class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, num_node_features, num_edge_features):
        super().__init__()

        self.node_encoder = Linear(num_node_features, hidden_channels)
        self.edge_encoder = Linear(num_edge_features, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr,batch):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        x = self.layers[0].conv(x, edge_index, edge_attr)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)
        out=self.lin(x)
        graph_level_output = global_mean_pool(out, batch)
        
        return graph_level_output
    
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_layers):
        super(GCN, self).__init__()
        self.num_layers = num_layers

        # First Graph Convolutional Layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_channels))

        # Additional Graph Convolutional Layers
        for _ in range(1, num_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Linear Layer for Regression
        self.lin = Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply Graph Convolutional Layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # Global Mean Pooling
        x = global_mean_pool(x, batch)

        # Linear Layer for Regression
        x = self.lin(x)

        return x
  
  
    

class GATNet(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_layers, heads):
        super(GATNet, self).__init__()

        self.conv1 = GATConv(num_features, hidden_channels, heads=heads, dropout=0.6)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=0.6))

        self.lin1 = torch.nn.Linear(hidden_channels * heads, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
        x = global_add_pool(x, batch)
        x = F.elu(self.lin1(x))
        x = self.lin2(x)
        return x