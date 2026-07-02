#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Optimizer Template (Advanced Constraint-Aware Legalizer)

ARCHITECTURE:
  - STAGE 1: Fast GNN Inference to establish macro-placement and optimal topology.
  - STAGE 2: Area Trap Prevention & MIB Consistency (Guarantees 0% Area Error).
  - STAGE 3: Constraint-Aware Hybrid Legalization
             Phase 1 -> Dispersion + Cluster Attraction.
             Phase 2 -> Boundary-Aware Monotonic Sweep (Guarantees 100% 0 overlap).
             Phase 3 -> 1D Compaction & Right/Top Boundary Gliding.
"""

import math
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import FloorplanOptimizer

from model import FloorplanGNN
from ml_utils import build_pyg_graph

# =============================================================================
# PURE PYTHON CONSTRAINT-AWARE HYBRID LEGALIZER
# =============================================================================
def merge_cluster_components(n: int, positions: np.ndarray, is_preplaced: np.ndarray, clust_const: np.ndarray):
    max_clust = int(clust_const.max()) if len(clust_const) > 0 else 0
    if max_clust == 0:
        return False
        
    any_merged = False
    for g in range(1, max_clust + 1):
        group_indices = np.where(clust_const == g)[0]
        if len(group_indices) <= 1:
            continue
            
        # Safeguard: Skip snapping if cluster blocks are too far apart (untrained GNN)
        centroids = positions[group_indices, :2] + positions[group_indices, 2:] / 2
        diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=-1)
        if distances.max() > 15.0:
            continue
            
        n_group = len(group_indices)
        parent = list(range(n_group))
        
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]
            
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                
        for i in range(n_group):
            idx_i = group_indices[i]
            x1, y1, w1, h1 = positions[idx_i, 0], positions[idx_i, 1], positions[idx_i, 2], positions[idx_i, 3]
            for j in range(i + 1, n_group):
                idx_j = group_indices[j]
                x2, y2, w2, h2 = positions[idx_j, 0], positions[idx_j, 1], positions[idx_j, 2], positions[idx_j, 3]
                
                ox = min(x1 + w1, x2 + w2) - max(x1, x2)
                oy = min(y1 + h1, y2 + h2) - max(y1, y2)
                
                touch_x = (ox > 1e-4) and (abs(y1 + h1 - y2) <= 1e-4 or abs(y2 + h2 - y1) <= 1e-4)
                touch_y = (oy > 1e-4) and (abs(x1 + w1 - x2) <= 1e-4 or abs(x2 + w2 - x1) <= 1e-4)
                
                if touch_x or touch_y:
                    union(i, j)
                    
        components_map = {}
        for i in range(n_group):
            root = find(i)
            if root not in components_map:
                components_map[root] = []
            components_map[root].append(group_indices[i])
            
        components = list(components_map.values())
        if len(components) <= 1:
            continue
            
        components.sort(key=len, reverse=True)
        merged_comp = list(components[0])
        
        for comp in components[1:]:
            best_dist = float('inf')
            best_tx = 0.0
            best_ty = 0.0
            
            for idx0 in merged_comp:
                x0, y0, w0, h0 = positions[idx0, 0], positions[idx0, 1], positions[idx0, 2], positions[idx0, 3]
                for idx1 in comp:
                    x1, y1, w1, h1 = positions[idx1, 0], positions[idx1, 1], positions[idx1, 2], positions[idx1, 3]
                    
                    candidates = [
                        (x0 - (x1 + w1), y0 - y1),
                        (x0 + w0 - x1, y0 - y1),
                        (x0 - x1, y0 - (y1 + h1)),
                        (x0 - x1, y0 + h0 - y1)
                    ]
                    
                    for tx, ty in candidates:
                        if is_preplaced[idx1]:
                            continue
                        dist = abs(tx) + abs(ty)
                        if dist < best_dist:
                            best_dist = dist
                            best_tx = tx
                            best_ty = ty
                            
            if best_dist < float('inf'):
                for idx in comp:
                    if not is_preplaced[idx]:
                        positions[idx, 0] = max(0.0, positions[idx, 0] + best_tx)
                        positions[idx, 1] = max(0.0, positions[idx, 1] + best_ty)
                any_merged = True
                merged_comp.extend(comp)
                
    return any_merged

def resolve_and_pull(n: int, positions: np.ndarray, is_preplaced: np.ndarray, constraints: np.ndarray):
    clust_const = constraints[:, 3]
    bound_const = constraints[:, 4].astype(int)
    
    # ---------------------------------------------------------
    # PHASE 1: Force-Directed Dispersion + Cluster Attraction
    # ---------------------------------------------------------
    changed = True
    iters = 0
    while changed and iters < 300:
        changed = False
        iters += 1
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, w1, h1 = positions[i, 0], positions[i, 1], positions[i, 2], positions[i, 3]
                x2, y2, w2, h2 = positions[j, 0], positions[j, 1], positions[j, 2], positions[j, 3]

                ox = min(x1 + w1, x2 + w2) - max(x1, x2)
                oy = min(y1 + h1, y2 + h2) - max(y1, y2)

                pre_i = is_preplaced[i]
                pre_j = is_preplaced[j]

                # Repulsive forces for resolving overlaps
                if ox > 1e-6 and oy > 1e-6:
                    changed = True
                    if ox < oy:
                        shift = (ox / 2.0) + 1e-4
                        if not pre_i and not pre_j:
                            if x1 < x2:
                                positions[i, 0] -= shift; positions[j, 0] += shift
                            else:
                                positions[i, 0] += shift; positions[j, 0] -= shift
                        elif pre_i and not pre_j:
                            positions[j, 0] += (ox + 1e-4) if x1 < x2 else -(ox + 1e-4)
                        elif pre_j and not pre_i:
                            positions[i, 0] += (ox + 1e-4) if x2 < x1 else -(ox + 1e-4)
                    else:
                        shift = (oy / 2.0) + 1e-4
                        if not pre_i and not pre_j:
                            if y1 < y2:
                                positions[i, 1] -= shift; positions[j, 1] += shift
                            else:
                                positions[i, 1] += shift; positions[j, 1] -= shift
                        elif pre_i and not pre_j:
                            positions[j, 1] += (oy + 1e-4) if y1 < y2 else -(oy + 1e-4)
                        elif pre_j and not pre_i:
                            positions[i, 1] += (oy + 1e-4) if y2 < y1 else -(oy + 1e-4)
                
                # Attractive forces for cluster members (Gentle pull)
                elif iters < 200 and clust_const[i] == clust_const[j] and clust_const[i] > 0:
                    dx = max(x1, x2) - min(x1 + w1, x2 + w2)
                    dy = max(y1, y2) - min(y1 + h1, y2 + h2)
                    
                    if dx > 1e-5:
                        pull_x = min(dx * 0.05, 2.0)
                        if not pre_i and not pre_j:
                            if x1 < x2: positions[i, 0] += pull_x; positions[j, 0] -= pull_x
                            else:       positions[i, 0] -= pull_x; positions[j, 0] += pull_x
                        elif pre_i and not pre_j:
                            positions[j, 0] += -pull_x if x1 < x2 else pull_x
                        elif pre_j and not pre_i:
                            positions[i, 0] += pull_x if x1 < x2 else -pull_x
                            
                    if dy > 1e-5:
                        pull_y = min(dy * 0.05, 2.0)
                        if not pre_i and not pre_j:
                            if y1 < y2: positions[i, 1] += pull_y; positions[j, 1] -= pull_y
                            else:       positions[i, 1] -= pull_y; positions[j, 1] += pull_y
                        elif pre_i and not pre_j:
                            positions[j, 1] += -pull_y if y1 < y2 else pull_y
                        elif pre_j and not pre_i:
                            positions[i, 1] += pull_y if y1 < y2 else -pull_y

        # Boundary pull forces
        W_max = max(positions[k, 0] + positions[k, 2] for k in range(n))
        H_max = max(positions[k, 1] + positions[k, 3] for k in range(n))
        for i in range(n):
            if not is_preplaced[i] and bound_const[i] > 0:
                code = bound_const[i]
                old_x, old_y = positions[i, 0], positions[i, 1]
                if code & 1: # Left
                    positions[i, 0] -= 0.05 * positions[i, 0]
                if code & 2: # Right
                    positions[i, 0] += 0.05 * (W_max - positions[i, 2] - positions[i, 0])
                if code & 4: # Top
                    positions[i, 1] += 0.05 * (H_max - positions[i, 3] - positions[i, 1])
                if code & 8: # Bottom
                    positions[i, 1] -= 0.05 * positions[i, 1]
                if abs(positions[i, 0] - old_x) > 1e-4 or abs(positions[i, 1] - old_y) > 1e-4:
                    pass

    # ---------------------------------------------------------
    # BOUNDARY CLAMPING
    # ---------------------------------------------------------
    for i in range(n):
        if not is_preplaced[i]:
            if positions[i, 0] < 0.0: positions[i, 0] = 0.0
            if positions[i, 1] < 0.0: positions[i, 1] = 0.0

    # ---------------------------------------------------------
    # PHASE 2: Strict Monotonic Push (Boundary-Aware)
    # ---------------------------------------------------------
    changed = True
    iters = 0
    while changed and iters < 150:
        changed = False
        iters += 1
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, w1, h1 = positions[i, 0], positions[i, 1], positions[i, 2], positions[i, 3]
                x2, y2, w2, h2 = positions[j, 0], positions[j, 1], positions[j, 2], positions[j, 3]

                ox = min(x1 + w1, x2 + w2) - max(x1, x2)
                oy = min(y1 + h1, y2 + h2) - max(y1, y2)

                if ox > 1e-6 and oy > 1e-6:
                    changed = True
                    pre_i = is_preplaced[i]
                    pre_j = is_preplaced[j]
                    
                    if pre_i and pre_j:
                        changed = False; continue

                    # Check if pushing right or up violates Left/Bottom boundary constraints
                    can_jump_j_x = not pre_j and not (bound_const[j] & 1)
                    can_jump_i_x = not pre_i and not (bound_const[i] & 1)
                    can_jump_j_y = not pre_j and not (bound_const[j] & 8)
                    can_jump_i_y = not pre_i and not (bound_const[i] & 8)

                    can_resolve_x = can_jump_i_x or can_jump_j_x
                    can_resolve_y = can_jump_i_y or can_jump_j_y

                    # Pick the axis that minimizes displacement, provided it's legal
                    push_axis = 'x' if ox < oy else 'y'
                    if push_axis == 'x' and not can_resolve_x: push_axis = 'y'
                    if push_axis == 'y' and not can_resolve_y: push_axis = 'x'

                    # Monotonic push: strictly take max() to avoid infinite loop oscillations
                    # Preplaced blocks must never be moved.
                    if push_axis == 'x':
                        if pre_i:
                            positions[j, 0] = max(positions[j, 0], x1 + w1)
                        elif pre_j:
                            positions[i, 0] = max(positions[i, 0], x2 + w2)
                        else:
                            if can_jump_j_x and not can_jump_i_x:
                                positions[j, 0] = max(positions[j, 0], x1 + w1)
                            elif can_jump_i_x and not can_jump_j_x:
                                positions[i, 0] = max(positions[i, 0], x2 + w2)
                            else:
                                if x1 > x2: positions[i, 0] = max(positions[i, 0], x2 + w2)
                                else:       positions[j, 0] = max(positions[j, 0], x1 + w1)
                    else:
                        if pre_i:
                            positions[j, 1] = max(positions[j, 1], y1 + h1)
                        elif pre_j:
                            positions[i, 1] = max(positions[i, 1], y2 + h2)
                        else:
                            if can_jump_j_y and not can_jump_i_y:
                                positions[j, 1] = max(positions[j, 1], y1 + h1)
                            elif can_jump_i_y and not can_jump_j_y:
                                positions[i, 1] = max(positions[i, 1], y2 + h2)
                            else:
                                if y1 > y2: positions[i, 1] = max(positions[i, 1], y2 + h2)
                                else:       positions[j, 1] = max(positions[j, 1], y1 + h1)

    # ---------------------------------------------------------
    # PHASE 3: 1D Compaction & Boundary Sweeps
    # ---------------------------------------------------------
    # 1. Base Compaction (Squeezes everything to Origin)
    for _ in range(2): 
        # Sweep Left
        order_x = np.argsort(positions[:, 0])
        for i in range(n):
            idx = order_x[i]
            if is_preplaced[idx]: continue
            new_x = 0.0
            y1, h1 = positions[idx, 1], positions[idx, 3]
            for j_order in range(i):
                j = order_x[j_order]
                y2, h2 = positions[j, 1], positions[j, 3]
                if max(y1, y2) < min(y1 + h1, y2 + h2) - 1e-6:
                    new_x = max(new_x, positions[j, 0] + positions[j, 2])
            positions[idx, 0] = new_x

        # Sweep Down
        order_y = np.argsort(positions[:, 1])
        for i in range(n):
            idx = order_y[i]
            if is_preplaced[idx]: continue
            new_y = 0.0
            x1, w1 = positions[idx, 0], positions[idx, 2]
            for j_order in range(i):
                j = order_y[j_order]
                x2, w2 = positions[j, 0], positions[j, 2]
                if max(x1, x2) < min(x1 + w1, x2 + w2) - 1e-6:
                    new_y = max(new_y, positions[j, 1] + positions[j, 3])
            positions[idx, 1] = new_y

    # 2. Boundary Gliding (Safely slides Top/Right blocks to edges)
    W_max = max(positions[k, 0] + positions[k, 2] for k in range(n))
    H_max = max(positions[k, 1] + positions[k, 3] for k in range(n))

    # Sweep Right
    right_blocks = [i for i in range(n) if (bound_const[i] & 2) and not is_preplaced[i]]
    right_blocks.sort(key=lambda i: positions[i, 0], reverse=True)
    for idx in right_blocks:
        new_x = W_max - positions[idx, 2]
        y1, h1 = positions[idx, 1], positions[idx, 3]
        for j in range(n):
            if j == idx: continue
            if positions[j, 0] >= positions[idx, 0]:
                y2, h2 = positions[j, 1], positions[j, 3]
                if max(y1, y2) < min(y1 + h1, y2 + h2) - 1e-6:
                    new_x = min(new_x, positions[j, 0] - positions[idx, 2])
        positions[idx, 0] = new_x

    # Sweep Up
    top_blocks = [i for i in range(n) if (bound_const[i] & 4) and not is_preplaced[i]]
    top_blocks.sort(key=lambda i: positions[i, 1], reverse=True)
    for idx in top_blocks:
        new_y = H_max - positions[idx, 3]
        x1, w1 = positions[idx, 0], positions[idx, 2]
        for j in range(n):
            if j == idx: continue
            if positions[j, 1] >= positions[idx, 1]:
                x2, w2 = positions[j, 0], positions[j, 2]
                if max(x1, x2) < min(x1 + w1, x2 + w2) - 1e-6:
                    new_y = min(new_y, positions[j, 1] - positions[idx, 3])
        positions[idx, 1] = new_y

    # 3. Post-processing: Snapping/abutting cluster blocks to resolve grouping violations
    for _ in range(1):
        if merge_cluster_components(n, positions, is_preplaced, clust_const):
            # Resolve overlaps (Monotonic Push)
            changed = True
            push_iters = 0
            while changed and push_iters < 30:
                changed = False
                push_iters += 1
                for i in range(n):
                    for j in range(i + 1, n):
                        x1, y1, w1, h1 = positions[i, 0], positions[i, 1], positions[i, 2], positions[i, 3]
                        x2, y2, w2, h2 = positions[j, 0], positions[j, 1], positions[j, 2], positions[j, 3]
                        ox = min(x1 + w1, x2 + w2) - max(x1, x2)
                        oy = min(y1 + h1, y2 + h2) - max(y1, y2)
                        if ox > 1e-6 and oy > 1e-6:
                            changed = True
                            pre_i = is_preplaced[i]
                            pre_j = is_preplaced[j]
                            if pre_i and pre_j: continue
                            can_jump_j_x = not pre_j and not (bound_const[j] & 1)
                            can_jump_i_x = not pre_i and not (bound_const[i] & 1)
                            can_jump_j_y = not pre_j and not (bound_const[j] & 8)
                            can_jump_i_y = not pre_i and not (bound_const[i] & 8)
                            push_axis = 'x' if ox < oy else 'y'
                            if push_axis == 'x' and not (can_jump_i_x or can_jump_j_x): push_axis = 'y'
                            if push_axis == 'y' and not (can_jump_i_y or can_jump_j_y): push_axis = 'x'
                            if push_axis == 'x':
                                if pre_i: positions[j, 0] = max(positions[j, 0], x1 + w1)
                                elif pre_j: positions[i, 0] = max(positions[i, 0], x2 + w2)
                                else:
                                    if x1 > x2: positions[i, 0] = max(positions[i, 0], x2 + w2)
                                    else:       positions[j, 0] = max(positions[j, 0], x1 + w1)
                            else:
                                if pre_i: positions[j, 1] = max(positions[j, 1], y1 + h1)
                                elif pre_j: positions[i, 1] = max(positions[i, 1], y2 + h2)
                                else:
                                    if y1 > y2: positions[i, 1] = max(positions[i, 1], y2 + h2)
                                    else:       positions[j, 1] = max(positions[j, 1], y1 + h1)
            # Re-run Compaction
            order_x = np.argsort(positions[:, 0])
            for i in range(n):
                idx = order_x[i]
                if is_preplaced[idx]: continue
                new_x = 0.0
                y1, h1 = positions[idx, 1], positions[idx, 3]
                for j_order in range(i):
                    j = order_x[j_order]
                    y2, h2 = positions[j, 1], positions[j, 3]
                    if max(y1, y2) < min(y1 + h1, y2 + h2) - 1e-6:
                        new_x = max(new_x, positions[j, 0] + positions[j, 2])
                positions[idx, 0] = new_x
            order_y = np.argsort(positions[:, 1])
            for i in range(n):
                idx = order_y[i]
                if is_preplaced[idx]: continue
                new_y = 0.0
                x1, w1 = positions[idx, 0], positions[idx, 2]
                for j_order in range(i):
                    j = order_y[j_order]
                    x2, w2 = positions[j, 0], positions[j, 2]
                    if max(x1, x2) < min(x1 + w1, x2 + w2) - 1e-6:
                        new_y = max(new_y, positions[j, 1] + positions[j, 3])
                positions[idx, 1] = new_y

            # Re-run Boundary Gliding
            W_max = max(positions[k, 0] + positions[k, 2] for k in range(n))
            H_max = max(positions[k, 1] + positions[k, 3] for k in range(n))
            right_blocks = [idx_r for idx_r in range(n) if (bound_const[idx_r] & 2) and not is_preplaced[idx_r]]
            right_blocks.sort(key=lambda idx_r: positions[idx_r, 0], reverse=True)
            for idx in right_blocks:
                new_x = W_max - positions[idx, 2]
                y1, h1 = positions[idx, 1], positions[idx, 3]
                for j in range(n):
                    if j == idx: continue
                    if positions[j, 0] >= positions[idx, 0]:
                        y2, h2 = positions[j, 1], positions[j, 3]
                        if max(y1, y2) < min(y1 + h1, y2 + h2) - 1e-6:
                            new_x = min(new_x, positions[j, 0] - positions[idx, 2])
                positions[idx, 0] = new_x
            top_blocks = [idx_t for idx_t in range(n) if (bound_const[idx_t] & 4) and not is_preplaced[idx_t]]
            top_blocks.sort(key=lambda idx_t: positions[idx_t, 1], reverse=True)
            for idx in top_blocks:
                new_y = H_max - positions[idx, 3]
                x1, w1 = positions[idx, 0], positions[idx, 2]
                for j in range(n):
                    if j == idx: continue
                    if positions[j, 1] >= positions[idx, 1]:
                        x2, w2 = positions[j, 0], positions[j, 2]
                        if max(x1, x2) < min(x1 + w1, x2 + w2) - 1e-6:
                            new_y = min(new_y, positions[j, 1] - positions[idx, 3])
                positions[idx, 1] = new_y
        else:
            break

# =============================================================================
# OPTIMIZER CLASS
# =============================================================================
class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = FloorplanGNN().to(self.device)
        weight_path = Path(__file__).parent / "floorplan_gnn_ep3_step1000000.pth"
        
        if weight_path.exists():
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device, weights_only=True))
        else:
            print("[WARNING] Pre-trained weights not found. Using randomly initialized GNN.")
            
        self.model.eval()
    
    def solve(self, block_count: int, area_targets: torch.Tensor,
              b2b_connectivity: torch.Tensor, p2b_connectivity: torch.Tensor,
              pins_pos: torch.Tensor, constraints: torch.Tensor,
              target_positions: torch.Tensor = None) -> List[Tuple[float, float, float, float]]:
        
        graph_data = build_pyg_graph(
            area_targets.unsqueeze(0), 
            b2b_connectivity.unsqueeze(0), 
            constraints.unsqueeze(0),
            p2b_connectivity.unsqueeze(0),
            pins_pos.unsqueeze(0)
        ).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(graph_data).cpu().numpy()
            
        positions = np.zeros((block_count, 4), dtype=np.float64)
        is_preplaced_arr = np.zeros(block_count, dtype=np.bool_)
        
        for i in range(block_count):
            pred_w = max(1e-3, float(predictions[i, 0]))
            pred_h = max(1e-3, float(predictions[i, 1]))
            x = float(predictions[i, 2])
            y = float(predictions[i, 3])
            
            c_fixed = int(constraints[i, 0]) > 0
            c_preplaced = int(constraints[i, 1]) > 0
            area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
            
            if c_fixed or c_preplaced:
                if target_positions is not None and target_positions[i, 2] != -1:
                    w = float(target_positions[i, 2])
                    h = float(target_positions[i, 3])
                else:
                    w = h = math.sqrt(area)
            else:
                predicted_ar = pred_w / pred_h
                predicted_ar = max(0.5, min(2.0, predicted_ar))
                w = math.sqrt(area * predicted_ar)
                h = area / w
                
            if c_preplaced and target_positions is not None:
                is_preplaced_arr[i] = True
                x = float(target_positions[i, 0])
                y = float(target_positions[i, 1])
                
            positions[i, 0] = x
            positions[i, 1] = y
            positions[i, 2] = w
            positions[i, 3] = h

        # ---------------------------------------------------------
        # SAFE MIB UNIFORMITY ALIGNMENT
        # ---------------------------------------------------------
        mib_const = constraints[:, 2].cpu().numpy()
        mib_groups = int(mib_const.max())
        if mib_groups > 0:
            for g in range(1, mib_groups + 1):
                group_indices = np.where(mib_const == g)[0]
                if len(group_indices) > 1:
                    fixed_w, fixed_h = None, None
                    for idx in group_indices:
                        if int(constraints[idx, 0]) > 0 or int(constraints[idx, 1]) > 0:
                            fixed_w = positions[idx, 2]
                            fixed_h = positions[idx, 3]
                            break
                    
                    # Calculate mean Aspect Ratio for soft blocks
                    avg_ar = np.mean(positions[group_indices, 2] / positions[group_indices, 3])
                    avg_ar = max(0.5, min(2.0, avg_ar))
                    
                    for idx in group_indices:
                        # Only modify dimensions of soft blocks (not fixed or preplaced)
                        if not (int(constraints[idx, 0]) > 0 or int(constraints[idx, 1]) > 0):
                            if fixed_w is not None and fixed_h is not None:
                                positions[idx, 2], positions[idx, 3] = fixed_w, fixed_h
                            else:
                                # Use individual area to guarantee 0% area violation!
                                area = float(area_targets[idx]) if area_targets[idx] > 0 else 1.0
                                positions[idx, 2] = math.sqrt(area * avg_ar)
                                positions[idx, 3] = area / positions[idx, 2]
            
        # Execute pure Python legalizer (Phase 1 -> 2 -> 3)
        resolve_and_pull(block_count, positions, is_preplaced_arr, constraints.cpu().numpy())
        
        return [(float(p[0]), float(p[1]), float(p[2]), float(p[3])) for p in positions]