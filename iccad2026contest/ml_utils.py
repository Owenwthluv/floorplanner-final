import torch
from torch_geometric.data import Data

import numpy as np

def build_pyg_graph(area_targets, b2b_conn, constraints, p2b_conn=None, pins_pos=None, sol=None):
    """
    Converts a single FloorSet batch (batch_size=1) into a PyTorch Geometric Data object.
    Input shapes assume a leading batch dimension of size 1.
    """
    areas = area_targets[0]
    b2b = b2b_conn[0]
    const = constraints[0]
    
    n_blocks = areas.shape[0]
    
    # Compute pin gravity centers
    gravity_x = np.zeros(n_blocks, dtype=np.float32)
    gravity_y = np.zeros(n_blocks, dtype=np.float32)
    has_pin_conn = np.zeros(n_blocks, dtype=np.float32)
    
    if p2b_conn is not None and pins_pos is not None:
        p2b = p2b_conn[0].cpu().numpy() if isinstance(p2b_conn, torch.Tensor) else p2b_conn[0]
        pins = pins_pos[0].cpu().numpy() if isinstance(pins_pos, torch.Tensor) else pins_pos[0]
        
        weighted_sum_x = np.zeros(n_blocks, dtype=np.float32)
        weighted_sum_y = np.zeros(n_blocks, dtype=np.float32)
        total_weights = np.zeros(n_blocks, dtype=np.float32)
        
        for edge in p2b:
            pin_idx = int(edge[0])
            block_idx = int(edge[1])
            weight = float(edge[2])
            
            if pin_idx >= 0 and block_idx >= 0 and block_idx < n_blocks:
                pin_x = pins[pin_idx, 0]
                pin_y = pins[pin_idx, 1]
                
                weighted_sum_x[block_idx] += pin_x * weight
                weighted_sum_y[block_idx] += pin_y * weight
                total_weights[block_idx] += weight
                
        for j in range(n_blocks):
            if total_weights[j] > 0:
                gravity_x[j] = weighted_sum_x[j] / total_weights[j]
                gravity_y[j] = weighted_sum_y[j] / total_weights[j]
                has_pin_conn[j] = 1.0

    node_features = []
    for i in range(n_blocks):
        area = float(areas[i]) if areas[i] > 0 else 1.0
        c_fixed = float(const[i, 0] > 0)
        c_preplaced = float(const[i, 1] > 0)
        c_mib = float(const[i, 2])
        c_cluster = float(const[i, 3])
        c_bound = float(const[i, 4])
        
        # Node Feature Vector
        features = [
            area, c_fixed, c_preplaced, c_mib, c_cluster, c_bound,
            float(gravity_x[i]), float(gravity_y[i]), float(has_pin_conn[i])
        ]
        node_features.append(features)
        
    x = torch.tensor(node_features, dtype=torch.float32)
    
    edge_src = []
    edge_dst = []
    edge_weights = []
    
    for i in range(b2b.shape[0]):
        u = int(b2b[i, 0])
        v = int(b2b[i, 1])
        w = float(b2b[i, 2])
        
        # Add undirected edges for the graph
        edge_src.extend([u, v])
        edge_dst.extend([v, u])
        edge_weights.extend([w, w])
        
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).view(-1, 1)
    
    y = None
    if sol is not None:
        # FloorSet-Lite sol contains [w, h, x, y]
        y = sol[0].clone().detach().float()
        
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)