import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class FloorplanGNN(nn.Module):
    """
    Graph Neural Network for predicting block coordinates and sizes.
    Inputs: Node features (9 channels) and Edge indices (Netlist)
    Outputs: [w, h, x, y] for each block (4 channels)
    """
    def __init__(self, in_channels=9, hidden_channels=128, out_channels=4):
        super(FloorplanGNN, self).__init__()
        
        # GCN Convolutional Layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        # Fully Connected Layer to output the final coordinates/sizes
        self.fc = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_attr', None)
        if edge_weight is not None and edge_weight.dim() > 1:
            edge_weight = edge_weight.squeeze(-1)
            
        # Pass through Graph Convolutional layers with ReLU activation
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        
        # Output layer
        out = self.fc(x)
        
        # Softplus ensures that predicted width (index 0) and height (index 1) 
        # are always positive numbers, plus a small epsilon to prevent absolute zero.
        w = F.softplus(out[:, 0:1]) + 1e-3
        h = F.softplus(out[:, 1:2]) + 1e-3
        xy = out[:, 2:]
        
        return torch.cat([w, h, xy], dim=1)