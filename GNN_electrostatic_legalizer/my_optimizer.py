#!/usr/bin/env python3
"""ENTRY POINT -- the optimizer the contest harness loads.

    python iccad2026_evaluate.py --evaluate my_optimizer.py

Three stages, one module each:

    my_optimizer.py           STAGE 1  GNN            (this file, + the driver)
    stage2_electrostatic.py   STAGE 2  ELECTROSTATIC
    stage3_legalizer.py       STAGE 3  LEGALIZER

STAGE 1 lives here: a single forward pass of the GNN, then the sizing rules the
contest imposes.

  * The checkpoint decides the graph width.  13 input channels means the net was
    trained with the four mandated target x/y/w/h appended to the 9 base
    features; 9 means it was not.  Read off `conv1.lin.weight` at load time, so
    any trained checkpoint drops in (GNN_WEIGHTS=... to switch).
  * Soft blocks take their target area with the aspect the net suggests, capped
    to [0.5, 2].  Fixed and preplaced blocks take their mandated dimensions
    verbatim, and preplaced also take their mandated position.
  * MIB groups are unified to a single shape -- the shape of a hard member if
    the group has one, otherwise the group's mean aspect.

Blocks overlap freely at the end of this stage; nothing here is legal yet.
STAGE 3 is what guarantees a legal floorplan.

Environment switches (all optional, defaults are the tuned values):
    GNN_WEIGHTS       checkpoint file                  floorplan_gnn_ar9_final.pth
    LD_CAP_END        final canvas of STAGE 2          0.5
    PACK_POOL         candidates per placement         1
    PACK_WIREW        weight of the wirelength term    0.5
    PACK_GRP          weight of cluster abutment       32
    EVAL_NO_RUNTIME=1 (on the evaluator) scores solution quality only

`viz_stages.py` renders all three stages side by side against the ground truth.
"""
import math
import os as _os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from iccad2026_evaluate import FloorplanOptimizer
from model import FloorplanGNN
from ml_utils import build_pyg_graph
from stage2_electrostatic import stage2_electrostatic
from stage3_legalizer import stage3_legalizer

# STAGE 1 checkpoint.  Kept in its own file so it can be swapped
# without touching the code; 13 input channels (see build_pyg_graph).
WEIGHTS = _os.environ.get("GNN_WEIGHTS", "floorplan_gnn_ar9_final.pth")


class MyOptimizer(FloorplanOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        wp = Path(__file__).parent / WEIGHTS
        # The graph width is read off the checkpoint rather than hard-coded, so
        # any of the trained nets loads: 13 channels means it was trained with
        # the four mandated target x/y/w/h appended to the 9 base features, 9
        # means it was not, and the graph is built to match.
        self.in_channels = 13
        if wp.exists():
            sd = torch.load(wp, map_location=self.device, weights_only=True)
            self.in_channels = int(sd["conv1.lin.weight"].shape[1])
            self.model = FloorplanGNN(in_channels=self.in_channels).to(self.device)
            self.model.load_state_dict(sd)
        else:
            print(f"[WARNING] {WEIGHTS} not found. Using randomly initialized GNN.")
            self.model = FloorplanGNN(in_channels=self.in_channels).to(self.device)
        self.model.eval()

    def solve(self, block_count, area_targets, b2b_connectivity, p2b_connectivity,
              pins_pos, constraints, target_positions=None):
        tp = None
        if self.in_channels >= 13:                    # net expects the targets
            tp = target_positions if target_positions is not None \
                else torch.full((block_count, 4), -1.0)
        graph_data = build_pyg_graph(
            area_targets.unsqueeze(0), b2b_connectivity.unsqueeze(0),
            constraints.unsqueeze(0), p2b_connectivity.unsqueeze(0),
            pins_pos.unsqueeze(0), target_pos=tp).to(self.device)
        with torch.no_grad():
            predictions = self.model(graph_data).cpu().numpy()

        positions = np.zeros((block_count, 4), dtype=np.float64)
        is_preplaced_arr = np.zeros(block_count, dtype=np.bool_)
        for i in range(block_count):
            pw = max(1e-3, float(predictions[i, 0])); ph = max(1e-3, float(predictions[i, 1]))
            x = float(predictions[i, 2]); y = float(predictions[i, 3])
            c_fixed = int(constraints[i, 0]) > 0; c_pre = int(constraints[i, 1]) > 0
            area = float(area_targets[i]) if area_targets[i] > 0 else 1.0
            if c_fixed or c_pre:
                if target_positions is not None and target_positions[i, 2] != -1:
                    w = float(target_positions[i, 2]); h = float(target_positions[i, 3])
                else:
                    w = h = math.sqrt(area)
            else:
                a_r = max(0.5, min(2.0, pw / ph)); w = math.sqrt(area * a_r); h = area / w
            if c_pre and target_positions is not None:
                is_preplaced_arr[i] = True
                x = float(target_positions[i, 0]); y = float(target_positions[i, 1])
            positions[i] = [x, y, w, h]

        # MIB uniformity (same as base)
        mib_const = constraints[:, 2].cpu().numpy(); mg = int(mib_const.max())
        if mg > 0:
            for g in range(1, mg + 1):
                gi = np.where(mib_const == g)[0]
                if len(gi) > 1:
                    fw = fh = None
                    for idx in gi:
                        if int(constraints[idx, 0]) > 0 or int(constraints[idx, 1]) > 0:
                            fw, fh = positions[idx, 2], positions[idx, 3]; break
                    avg_ar = max(0.5, min(2.0, float(np.mean(positions[gi, 2] / positions[gi, 3]))))
                    for idx in gi:
                        if not (int(constraints[idx, 0]) > 0 or int(constraints[idx, 1]) > 0):
                            if fw is not None:
                                positions[idx, 2], positions[idx, 3] = fw, fh
                            else:
                                a = float(area_targets[idx]) if area_targets[idx] > 0 else 1.0
                                positions[idx, 2] = math.sqrt(a * avg_ar)
                                positions[idx, 3] = a / positions[idx, 2]

        cons_np = constraints.cpu().numpy()
        # boundary mark as the GNN drew it, BEFORE density spreading
        _cd = cons_np[:, 4].astype(int)
        _l = [positions[i, 0] for i in range(block_count) if _cd[i] & 1]
        _r = [positions[i, 0] + positions[i, 2]
              for i in range(block_count) if _cd[i] & 2]
        w_hint = float(np.median(_r) - np.median(_l)) if (_l and _r) else None
        stage2_electrostatic(block_count, positions, is_preplaced_arr,
                         area_targets=area_targets.cpu().numpy(), constraints=cons_np,
                         b2b_edges=b2b_connectivity.cpu().numpy(),
                         cap_start=float(_os.environ.get("LD_CAP_START", "6.0")),
                         cap_end=float(_os.environ.get("LD_CAP_END", "0.5")),
                         anneal_frac=float(_os.environ.get("LD_ANNEAL", "0.6")),
                         rounds=int(_os.environ.get("LD_ROUNDS", "600")),
                         lr=float(_os.environ.get("LD_LR", "1.0")),
                         w_battr=float(_os.environ.get("LD_BATTR", "3.0")),
                         w_cattr=float(_os.environ.get("LD_CATTR", "2.0")),
                         w_conn=float(_os.environ.get("LD_CONN", "0.5")),
                         w_grav=float(_os.environ.get("LD_GRAV", "0.0")))
        stage3_legalizer(block_count, positions, is_preplaced_arr, cons_np,
                    b2b=b2b_connectivity.cpu().numpy(),
                    p2b=p2b_connectivity.cpu().numpy(),
                    pins=pins_pos.cpu().numpy(), w_hint=w_hint)

        return [(float(p[0]), float(p[1]), float(p[2]), float(p[3])) for p in positions]
