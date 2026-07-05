#!/usr/bin/env python3
"""
visualize.py — render saved floorplan RESULTS to images so you can see *why*
each case scores the way it does (esp. grouping/boundary violations).

This tool does NOT run any optimizer. It reads a solutions JSON produced by the
official evaluator:

    python iccad2026_evaluate.py --evaluate my_optimizer --save-solutions
    # -> writes my_optimizer_solutions.json

then re-scores every stored solution with the exact evaluator logic and draws
two panels side by side per case:

    [ OUR SOLUTION ]   |   [ GROUND TRUTH ]

with, on the solution panel:
  * blocks coloured by CLUSTER (same colour = same grouping group); non-cluster
    soft blocks grey, fixed hatched '///', preplaced hatched 'xxx' + bold edge;
  * BOUNDARY blocks outlined GREEN if they touch their required bbox edge, RED
    if they violate it, annotated with the edges they must touch (L/R/T/B);
  * split GROUPING groups (connected components > 1) boxed with a red dashed
    hull and flagged in the title;
  * the bounding box (dashed) and a title with cost + HPWL/area gaps + the
    exact boundary / grouping / mib violation counts.

Constraint semantics are replicated from iccad2026_evaluate.evaluate_solution
(boundary bitmask touch, grouping = shapely connected components, mib =
distinct shapes) so the highlights match the official score.

Usage (from this iccad2026contest dir):
    python visualize.py                                  # ALL cases in the default results file
    python visualize.py results.json --worst 8           # 8 worst cases by cost
    python visualize.py results.json --ids 0,1,79        # specific cases
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CONTEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CONTEST_DIR))

from iccad2026_evaluate import ContestEvaluator, evaluate_solution, compute_cost  # noqa: E402

# Boundary bitmask (matches evaluator): 1=left 2=right 4=top 8=bottom.
_EDGE_LETTER = {1: "L", 2: "R", 4: "T", 8: "B"}


def _bbox(positions):
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    xe = [p[0] + p[2] for p in positions]
    ye = [p[1] + p[3] for p in positions]
    return min(xs), min(ys), max(xe), max(ye)


def _boundary_status(positions, bound_const, bc):
    """Per boundary block: (required-edge string, satisfied?) — evaluator logic."""
    x0, y0, x1, y1 = _bbox(positions)
    eps = 1e-6
    out = {}
    for i in range(bc):
        code = int(bound_const[i])
        if code == 0:
            continue
        bx, by, bw, bh = positions[i]
        touches = {1: abs(bx - x0) < eps, 2: abs(bx + bw - x1) < eps,
                   4: abs(by + bh - y1) < eps, 8: abs(by - y0) < eps}
        req = [b for b in (1, 2, 4, 8) if code & b]
        ok = all(touches[b] for b in req)
        out[i] = ("".join(_EDGE_LETTER[b] for b in req), ok)
    return out


def _split_clusters(positions, clust_const, bc):
    """Return {cluster_id: n_components} using shapely (as the evaluator does),
    falling back to rectangle-adjacency connected components if shapely absent."""
    groups = {}
    for i in range(bc):
        g = int(clust_const[i])
        if g > 0:
            groups.setdefault(g, []).append(i)
    result = {}
    try:
        from shapely.geometry import box
        from shapely.ops import unary_union
        for g, idx in groups.items():
            if len(idx) < 2:
                continue
            u = unary_union([box(positions[i][0], positions[i][1],
                                 positions[i][0] + positions[i][2],
                                 positions[i][1] + positions[i][3]) for i in idx])
            result[g] = len(u.geoms) if u.geom_type == "MultiPolygon" else 1
        return result, groups
    except Exception:
        pass
    # Fallback: union-find on edge-adjacency (share a border segment, no overlap needed).
    for g, idx in groups.items():
        if len(idx) < 2:
            continue
        parent = {i: i for i in idx}
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        eps = 1e-6
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                xi, yi, wi, hi = positions[i]; xj, yj, wj, hj = positions[j]
                ox = min(xi + wi, xj + wj) - max(xi, xj)
                oy = min(yi + hi, yj + hj) - max(yi, yj)
                touch = (abs(ox) < eps and oy > eps) or (abs(oy) < eps and ox > eps) \
                    or (ox > eps and oy > eps)
                if touch:
                    parent[find(i)] = find(j)
        result[g] = len({find(i) for i in idx})
    return result, groups


def _draw(ax, positions, constraints, bc, title, cmap):
    import matplotlib.patches as mpatches
    fixed = constraints[:, 0]
    pre = constraints[:, 1]
    clust = constraints[:, 3]
    bound = constraints[:, 4]

    bstat = _boundary_status(positions, bound, bc)
    splits, groups = _split_clusters(positions, clust, bc)
    x0, y0, x1, y1 = _bbox(positions)

    for i in range(bc):
        x, y, w, h = positions[i]
        g = int(clust[i])
        if g > 0:
            face = cmap(g % 20)
        elif int(fixed[i]) != 0:
            face = "lightgray"
        elif int(pre[i]) != 0:
            face = "wheat"
        else:
            face = "#cfe0f0"
        hatch = "///" if int(fixed[i]) != 0 else ("xxx" if int(pre[i]) != 0 else None)

        edge, lw = "black", 0.6
        if i in bstat:                              # boundary block: green=ok red=bad
            edge = "limegreen" if bstat[i][1] else "red"
            lw = 2.4
        ax.add_patch(mpatches.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                                        linewidth=lw, hatch=hatch, alpha=0.75))
        lbl = str(i)
        if i in bstat:
            lbl += f"\n{bstat[i][0]}"               # required edges, e.g. "TL"
        ax.text(x + w / 2, y + h / 2, lbl, ha="center", va="center", fontsize=6)

    # Red dashed hull around each SPLIT grouping group (the ones costing V_rel).
    for g, ncomp in splits.items():
        if ncomp > 1:
            idx = groups[g]
            gx0 = min(positions[i][0] for i in idx); gy0 = min(positions[i][1] for i in idx)
            gx1 = max(positions[i][0] + positions[i][2] for i in idx)
            gy1 = max(positions[i][1] + positions[i][3] for i in idx)
            ax.add_patch(mpatches.Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, fill=False,
                                            edgecolor="red", linewidth=1.4, linestyle=":"))

    ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    edgecolor="gray", linewidth=0.8, linestyle="--"))
    ax.set_title(title, fontsize=9)
    ax.set_xlim(x0 - (x1 - x0) * 0.03, x1 + (x1 - x0) * 0.03)
    ax.set_ylim(y0 - (y1 - y0) * 0.03, y1 + (y1 - y0) * 0.03)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)


def main():
    ap = argparse.ArgumentParser(
        description="Render saved floorplan solutions (from *_solutions.json) to PNGs.")
    ap.add_argument("results", nargs="?",
                    default=str(CONTEST_DIR / "my_optimizer_solutions.json"),
                    help="solutions JSON produced by "
                         "'iccad2026_evaluate.py --evaluate <opt> --save-solutions'")
    ap.add_argument("--ids", default=None, help="comma-separated case ids to render")
    ap.add_argument("--worst", type=int, default=None,
                    help="render only the WORST N cases by cost (default: all cases)")
    ap.add_argument("--out-dir", default=str(CONTEST_DIR / "viz_out"))
    ap.add_argument("--data-path", default=str(CONTEST_DIR.parent),
                    help="dir containing LiteTensorDataTest/ (default: repo root)")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")                       # headless server
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required: pip install matplotlib"); return

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"results file not found: {results_path}"); return
    with open(results_path) as f:
        data = json.load(f)
    solutions = {int(s["test_id"]): s for s in data.get("solutions", [])}
    print(f"Loaded {len(solutions)} solutions from {results_path.name} "
          f"(submission={data.get('submission', '?')})")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.cm.tab20

    ev = ContestEvaluator(data_path=args.data_path, verbose=False)
    ev._load_dataset()

    if args.ids:
        scan = [int(s) for s in args.ids.split(",")]
        missing = [i for i in scan if i not in solutions]
        if missing:
            print(f"warning: no stored solution for case(s) {missing}, skipped")
            scan = [i for i in scan if i in solutions]
    else:
        scan = sorted(solutions)

    # Score every requested case from its STORED positions (no optimizer run).
    rows = []
    print(f"Scoring {len(scan)} stored case(s)...")
    for idx in scan:
        sample = ev.dataset[idx]
        inputs, labels = sample["input"], sample["label"]
        area_target, b2b, p2b, pins, constraints = inputs
        bc = int((area_target != -1).sum().item())
        baseline, target_pos = ev._extract_baseline(idx, labels, b2b, p2b, pins, bc)
        positions = [tuple(float(v) for v in p) for p in solutions[idx]["positions"]]
        m = evaluate_solution({"positions": positions, "runtime": 1.0}, baseline,
                              constraints, b2b, p2b, pins, area_target, target_pos,
                              median_runtime=1.0)
        qcost = compute_cost(m.hpwl_gap, m.area_gap, m.violations_relative, 1.0, m.is_feasible)
        rows.append(dict(id=idx, bc=bc, cost=qcost, feasible=m.is_feasible,
                         hpwl=m.hpwl_gap, area=m.area_gap, vrel=m.violations_relative,
                         bnd=m.boundary_violations, grp=m.grouping_violations,
                         mib=m.mib_violations, positions=positions,
                         target=target_pos, constraints=constraints[:bc]))

    if args.worst is not None and not args.ids:
        render = sorted(rows, key=lambda r: -r["cost"])[:args.worst]
    else:
        render = rows
    print(f"Rendering {len(render)} case(s) -> {out_dir}/")
    print(f"  {'id':>4} {'blk':>4} {'cost':>7} {'hpwl':>6} {'area':>6} {'vrel':>6}  bnd/grp/mib")
    for r in render:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        gt = [tuple(float(v) for v in r["target"][i]) for i in range(r["bc"])] \
            if r["target"] is not None else r["positions"]
        title = (f"case {r['id']}  {r['bc']} blocks  cost={r['cost']:.2f}  "
                 f"HPWL_gap={r['hpwl']:.2f} Area_gap={r['area']:.2f} "
                 f"V_rel={r['vrel']:.3f}\nviolations  bnd={r['bnd']} "
                 f"grp={r['grp']} mib={r['mib']}  "
                 f"{'FEASIBLE' if r['feasible'] else 'INFEASIBLE'}")
        _draw(axes[0], r["positions"], r["constraints"], r["bc"], "OUR SOLUTION — " + title, cmap)
        _draw(axes[1], gt, r["constraints"], r["bc"], "GROUND TRUTH", cmap)
        fig.tight_layout()
        fp = out_dir / f"case_{r['id']:03d}.png"
        fig.savefig(fp, dpi=130); plt.close(fig)
        print(f"  {r['id']:>4} {r['bc']:>4} {r['cost']:>7.2f} {r['hpwl']:>6.2f} "
              f"{r['area']:>6.2f} {r['vrel']:>6.3f}  {r['bnd']}/{r['grp']}/{r['mib']}"
              f"   -> {fp.name}")

    # Legend / how to read the images.
    print("\nLegend: cluster blocks share a colour | fixed='///' preplaced='xxx' | "
          "boundary edge GREEN=touches required side, RED=violates ('L/R/T/B'=required) | "
          "red dotted box = a grouping group split into >1 component (grouping violation) | "
          "gray dashed = bounding box.")


if __name__ == "__main__":
    main()
