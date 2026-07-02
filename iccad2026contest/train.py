import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import FloorplanGNN 
from ml_utils import build_pyg_graph
from iccad2026_evaluate import get_training_dataloader

def compute_custom_constraints_loss(positions_xywh, constraints_tensor, area_targets):
    """
    Computes custom differentiable L1 penalties for soft constraints:
    - Boundary constraints: distance to current dynamic bounding box edges.
    - Grouping (Cluster) constraints: pairwise centroid distances within the same group.
    - MIB consistency constraints: pairwise dimension differences within the same group.
    """
    N = positions_xywh.shape[0]
    device = positions_xywh.device
    constraints_tensor = constraints_tensor.to(device)
    area_targets = area_targets.to(device)
    
    x = positions_xywh[:, 0]
    y = positions_xywh[:, 1]
    w = positions_xywh[:, 2]
    h = positions_xywh[:, 3]
    
    # Compute current bounding box
    x_min = x.min()
    y_min = y.min()
    x_max = (x + w).max()
    y_max = (y + h).max()
    
    boundary_loss = torch.tensor(0.0, device=device)
    grouping_loss = torch.tensor(0.0, device=device)
    mib_loss = torch.tensor(0.0, device=device)
    
    fixed_const = constraints_tensor[:, 0]
    preplaced_const = constraints_tensor[:, 1]
    mib_const = constraints_tensor[:, 2]
    clust_const = constraints_tensor[:, 3]
    bound_const = constraints_tensor[:, 4]
    
    # 1. Boundary constraints penalty (L1 norm)
    bound_mask = bound_const > 0
    num_boundary = bound_mask.sum().item()
    if num_boundary > 0:
        for i in torch.nonzero(bound_mask).flatten():
            code = int(bound_const[i].item())
            if code & 1:
                boundary_loss = boundary_loss + torch.abs(x[i] - x_min)
            if code & 2:
                boundary_loss = boundary_loss + torch.abs(x[i] + w[i] - x_max)
            if code & 4:
                boundary_loss = boundary_loss + torch.abs(y[i] + h[i] - y_max)
            if code & 8:
                boundary_loss = boundary_loss + torch.abs(y[i] - y_min)
        boundary_loss = boundary_loss / num_boundary
                
    # 2. Grouping (Cluster) constraints penalty (L1 norm)
    max_clust = int(clust_const.max().item())
    num_clust_groups = 0
    if max_clust > 0:
        for g in range(1, max_clust + 1):
            indices = torch.nonzero(clust_const == g).flatten()
            if len(indices) > 1:
                num_clust_groups += 1
                cx = x[indices] + w[indices] / 2
                cy = y[indices] + h[indices] / 2
                diff_x = cx.unsqueeze(0) - cx.unsqueeze(1)
                diff_y = cy.unsqueeze(0) - cy.unsqueeze(1)
                # Pairwise L1 distance of centroids
                dist_l1 = torch.abs(diff_x) + torch.abs(diff_y)
                grouping_loss = grouping_loss + dist_l1.sum() / (len(indices) * (len(indices) - 1))
        if num_clust_groups > 0:
            grouping_loss = grouping_loss / num_clust_groups
                
    # 3. MIB constraints penalty (L1 norm)
    max_mib = int(mib_const.max().item())
    num_mib_groups = 0
    if max_mib > 0:
        for g in range(1, max_mib + 1):
            indices = torch.nonzero(mib_const == g).flatten()
            if len(indices) > 1:
                num_mib_groups += 1
                pw = w[indices]
                ph = h[indices]
                diff_w = pw.unsqueeze(0) - pw.unsqueeze(1)
                diff_h = ph.unsqueeze(0) - ph.unsqueeze(1)
                mib_loss = mib_loss + (torch.abs(diff_w) + torch.abs(diff_h)).sum() / (len(indices) * (len(indices) - 1))
        if num_mib_groups > 0:
            mib_loss = mib_loss / num_mib_groups
                
    return boundary_loss, grouping_loss, mib_loss

# =============================================================================
# Training Loop
# =============================================================================
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target device: {device}")
    
    model = FloorplanGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Initializing official dataloader for training dataset...")
    # Train on 10,000 samples by default for fast convergence, can be changed to None for full dataset
    train_loader = get_training_dataloader(batch_size=1, num_samples=None)
    
    model.train()
    epochs = 3
    save_interval = 200000
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_loss_mse = 0.0
        total_loss_bound = 0.0
        total_loss_group = 0.0
        total_loss_mib = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            area_targets, b2b_conn, p2b_conn, pins_pos, constraints, tree_sol, sol, metrics = batch_data
            
            block_count = int((area_targets.squeeze(0) != -1).sum().item())
            if block_count == 0:
                continue
                
            graph_data = build_pyg_graph(area_targets, b2b_conn, constraints, p2b_conn, pins_pos, sol)
            graph_data = graph_data.to(device)
            
            optimizer.zero_grad()
            predictions = model(graph_data)
            
            positions_xywh = torch.stack([
                predictions[:, 2],  # x
                predictions[:, 3],  # y
                predictions[:, 0],  # w
                predictions[:, 1]   # h
            ], dim=1)
            
            # 1. Main MSE loss to Ground Truth (Forces correct sizes and placements)
            loss_mse = criterion(predictions, graph_data.y)
            
            # 2. Soft constraints differentiable L1 penalties
            bound_loss, group_loss, mib_loss = compute_custom_constraints_loss(
                positions_xywh,
                constraints.squeeze(0)[:block_count],
                area_targets.squeeze(0)[:block_count]
            )
            
            # Combine losses with stable weights
            loss = loss_mse + 0.1 * bound_loss + 0.1 * group_loss + 0.1 * mib_loss
            
            loss.backward()
            
            # Gradient clipping to prevent gradient explosion on large graphs
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_loss_mse += loss_mse.item()
            total_loss_bound += bound_loss.item()
            total_loss_group += group_loss.item()
            total_loss_mib += mib_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.2f}",
                'mse': f"{loss_mse.item():.2f}",
                'bound': f"{bound_loss.item():.2f}",
                'group': f"{group_loss.item():.2f}"
            })
            
            if (batch_idx + 1) % save_interval == 0:
                checkpoint_path = f"floorplan_gnn_ep{epoch+1}_step{batch_idx+1}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                
        avg_loss = total_loss / len(train_loader)
        print(f"\n==== Epoch {epoch+1:02d} Completed | Avg Loss: {avg_loss:.4f} ====")
        print(f"  MSE:     {total_loss_mse/len(train_loader):.4f}")
        print(f"  Bound:   {total_loss_bound/len(train_loader):.4f} | Group: {total_loss_group/len(train_loader):.4f} | MIB: {total_loss_mib/len(train_loader):.4f}\n")
        
        torch.save(model.state_dict(), f"floorplan_gnn_ep{epoch+1}_final.pth")
        
    print("Full scale training completed. Final model weights saved.")

if __name__ == "__main__":
    train_model()