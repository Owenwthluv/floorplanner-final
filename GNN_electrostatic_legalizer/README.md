# GNN + electrostatic + legalizer

Floorplanner for the ICCAD 2026 FloorSet Challenge. Three stages, one module each.

## Run

Copy the files next to `iccad2026_evaluate.py` and the dataset, then:

```bash
python iccad2026_evaluate.py --evaluate my_optimizer.py
EVAL_NO_RUNTIME=1 python iccad2026_evaluate.py --evaluate my_optimizer.py   # quality only
```

## Results (100 validation cases)

| | |
|---|---|
| Total Score (runtime factor = 1) | **1.3560** |
| Avg Cost (runtime factor = 1) | **1.3803** |
| Feasible | **100 / 100** |
| Runtime | 57.7 s total, 0.577 s/case, 1.36 s worst |
| hpwl gap / area gap / violations | 0.305 / 0.177 / 0.052 |

Against the organisers' per-case median runtimes every case reaches the
`max(0.7, R^0.3)` floor — the worst case runs at 0.11x its median — so the
runtime-weighted score lands near **0.95**. The number the local script prints
(~1.74) is an artefact: it compares each case against the median of *this
submission's own* 100 runtimes, which always penalises the large cases, and
those carry ~99% of the exponentially weighted total.

## Stages

### STAGE 1 — GNN (`my_optimizer.py`)

One forward pass, then the sizing the contest imposes. The checkpoint decides
the graph width: 13 input channels means it was trained with the four mandated
target x/y/w/h appended to the 9 base features, 9 means it was not — read off
`conv1.lin.weight` at load time, so any trained checkpoint drops in. Soft blocks
take their target area with the net's aspect capped to [0.5, 2]; fixed and
preplaced take their mandated dimensions verbatim; MIB groups collapse to one
shape. Blocks overlap freely here.

### STAGE 2 — ELECTROSTATIC (`stage2_electrostatic.py`)

Density repulsion (Poisson solved by DCT on a 64×64 grid) against three
attractions — boundary, cluster, netlist — on a canvas annealing from
`cap_start=6.0` to `cap_end=0.5`.

That final canvas is **smaller than the total block area**, so this stage
cannot and does not remove overlap: 113 of 115 blocks still overlap when it
finishes, covering 64% of the block area. It compacts instead. That is the
point — the legalizer only needs the relative arrangement, and spreading far
enough to actually separate blocks destroys the neighbourhood structure the GNN
produced. Measured, `cap_end` 3.0 → 0.5 is worth **0.134**, the single largest
gain in the pipeline.

### STAGE 3 — LEGALIZER (`stage3_legalizer.py`)

Takes that overlapping pile and makes it legal:

- **frame** — width from target utilisation, pinned to a preplaced R block when
  one fixes that edge. The top is left free; it is the edge packing grows into.
- **bottom row** — BL corner, B blocks, BR corner, widths spread to span the
  frame. Against a preplaced obstacle a block first narrows into the pocket
  before it, then flattens to duck underneath, and only jumps past as a last
  resort.
- **side towers** — stacked bottom-up. A boundary block slides along its edge to
  meet a preplaced cluster peer (1 degree of freedom against 0), narrows to
  squeeze past an obstacle rather than hopping over it, and unpinned blocks drop
  into the holes left behind.
- **interior** — MaxRects free-rectangle fill. Unlike a skyline it can see the
  pocket under a floating preplaced block. One candidate per step, in bottom-up
  density order, scored

  ```
  score = S_area + 0.5·S_wire + 32·S_grp
  ```

  | term | formula | meaning |
  |---|---|---|
  | `S_area` | `W·max(0, y+h−H_current)/A` | only frame *growth* costs; landing inside the envelope is free |
  | `S_wire` | `HPWL_block / HPWL_density` | counts **already-placed** neighbours only |
  | `S_grp` | `gap_to_nearest_cluster_peer / √A` | cluster abutment |

- **top row** — laid side by side with a flat lid, then the topmost tower block
  stretches up to reach a T-coded cluster peer.
- **safety net** — two-tier push; tier 2 is strictly monotonic with preplaced
  immovable, so overlap is always exactly zero.

## What moved the score

| change | from → to |
|---|---|
| `cap_end` 3.0 → 0.5 (STAGE 2 compacts instead of spreading) | 1.520 → 1.386 |
| added `S_grp`, the cluster-abutment term | 1.952 → 1.748 |
| `S_wire` counts only already-placed neighbours | 1.722 → 1.633 |
| candidate window 10 → 1 (respect the bottom-up order) | 1.614 → 1.533 |

Every one of these replaces a *proxy* with a *direct* measurement. Mechanisms
built on proxies all lost and were removed: leftover-rectangle "fit", horizontal
hug, fixed-shape cluster super-blocks, pocket-driven fill.

## Tuning switches

Defaults are the tuned values; all optional.

| variable | default | effect |
|---|---|---|
| `GNN_WEIGHTS` | `floorplan_gnn_ar9_final.pth` | checkpoint (width auto-detected) |
| `LD_CAP_END` | `0.5` | STAGE 2 final canvas |
| `PACK_POOL` | `1` | candidates weighed per placement |
| `PACK_WIREW` | `0.5` | weight of the wirelength term |
| `PACK_GRP` | `32` | weight of cluster abutment |
| `PACK_AR` | `3.0` | aspect-ratio cap for soft blocks |

## Measurement note

The pipeline is **not reproducible run to run**: identical code and config give
different layouts on 99 of 100 cases, with a single case varying by up to 0.146.
Averaged over 100 cases the noise is ±0.003, so **only differences above 0.01
mean anything**. Multi-threaded GNN inference on CPU is the source. A few
defaults (`PACK_AR`, `PACK_PIN`, `PACK_SIDE_ORDER`) were settled on differences
below that threshold — they are arbitrary choices, not evidence.

## Visualisers

| script | shows |
|---|---|
| `viz_stages.py` | all three stages plus ground truth, side by side |
| `viz_gt.py` | final result against ground truth |
| `viz_frame.py` | one case in detail — boundary satisfaction, clusters, MIB |
| `viz_steps.py` | the interior fill block by block, with the free rectangles |
| `viz_density.py` | what STAGE 2 hands over, overlaps marked |
