#!/usr/bin/env python3
"""
Render every case through the four stages of the pipeline, side by side.

  STAGE 1  GNN                single-shot GNN prediction, area sizing, MIB
                              unification.  Blocks overlap freely here.
  STAGE 2  ELECTROSTATIC      density repulsion + boundary / cluster / netlist
                              attraction on a shrinking canvas.  With cap_end
                              below 1.0 this no longer separates blocks -- it
                              COMPACTS them, and that is what the legalizer
                              actually wants from it.
  STAGE 3  LEGALIZER          stage3_legalizer: bottom row, side towers, MaxRects
                              interior fill, top row, monotonic safety net.
                              The only stage that guarantees zero overlap.
  GT                          the dataset's own floorplan, for reference.

Colours match viz_frame.py: fill by cluster, hatch for fixed / preplaced,
red outline where a block overlaps another (stages 1-2) or misses the boundary
edge it is coded for (stage 3 / GT).

  python viz_stages.py                 # all 100 cases
  python viz_stages.py --ids 35,90,94
"""
import argparse
import json
import sys
from pathlib import Path

CONTEST_DIR = Path(__file__).resolve().parent


def _decode(c):
    c = int(c)
    return "".join(k for k, b in (("L", c & 1), ("R", c & 2),
                                  ("T", c & 4), ("B", c & 8)) if b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="", help="comma list; empty = all 100")
    ap.add_argument("--out-dir", default=str(CONTEST_DIR / "viz_stages"))
    ap.add_argument("--data-path", default=str(CONTEST_DIR.parent))
    args = ap.parse_args()

    sys.path.insert(0, str(CONTEST_DIR))
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp

    import stage2_electrostatic as LD
    import stage3_legalizer as FP
    import my_optimizer as M
    from iccad2026_evaluate import ContestEvaluator

    SNAP = {}
    orig_ld, orig_fp = LD.stage2_electrostatic, FP.stage3_legalizer

    def spy_ld(n, P, ipp, **kw):
        SNAP["raw"] = P[:n].copy()                    # before any spreading
        return orig_ld(n, P, ipp, **kw)

    def spy_fp(n, P, ipp, cons, util=None, **kw):
        SNAP["push"] = P[:n].copy()                   # after the field settles
        return orig_fp(n, P, ipp, cons, util, **kw)

    LD.stage2_electrostatic = spy_ld
    M.stage2_electrostatic = spy_ld
    FP.stage3_legalizer = spy_fp
    M.stage3_legalizer = spy_fp

    ev = ContestEvaluator(data_path=args.data_path, verbose=False)
    ev._load_dataset()
    opt = M.MyOptimizer()
    ids = [int(s) for s in args.ids.split(",") if s.strip()] or list(range(100))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cmap = plt.cm.tab20

    for idx in ids:
        rec = ev.dataset[idx]
        area, b2b, p2b, pins, cons = rec["input"]
        bc = int((area != -1).sum().item())
        tp = rec.get("target_positions")
        SNAP.clear()
        final = opt.solve(bc, area, b2b, p2b, pins, cons, tp)
        final = [tuple(float(v) for v in p) for p in final][:bc]

        poly = rec["label"][0]                        # ground truth polygons
        gt = []
        for i in range(bc):
            xs = [float(v) for v in poly[i][:, 0]]
            ys = [float(v) for v in poly[i][:, 1]]
            gt.append((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))

        panels = [
            ([tuple(float(v) for v in p) for p in SNAP["raw"]], "STAGE 1  GNN", True),
            ([tuple(float(v) for v in p) for p in SNAP["push"]],
             "STAGE 2  electrostatic", True),
            (final, "STAGE 3  legalizer", False),
            (gt, "ground truth", False),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(26, 7))
        for ax, (pos, name, mark_overlap) in zip(axes, panels):
            x0 = min(p[0] for p in pos); y0 = min(p[1] for p in pos)
            x1 = max(p[0] + p[2] for p in pos); y1 = max(p[1] + p[3] for p in pos)

            bad = set()
            ov_area = 0.0
            if mark_overlap:
                for i in range(bc):
                    for j in range(i + 1, bc):
                        ox = min(pos[i][0] + pos[i][2], pos[j][0] + pos[j][2]) \
                            - max(pos[i][0], pos[j][0])
                        oy = min(pos[i][1] + pos[i][3], pos[j][1] + pos[j][3]) \
                            - max(pos[i][1], pos[j][1])
                        if ox > 1e-6 and oy > 1e-6:
                            bad.add(i); bad.add(j); ov_area += ox * oy

            gviol = 0
            for g in sorted({int(c) for c in cons[:bc, 3]} - {0}):
                mem = [i for i in range(bc) if int(cons[i, 3]) == g]
                if len(mem) < 2:
                    continue
                par = list(range(len(mem)))

                def find(a):
                    while par[a] != a:
                        par[a] = par[par[a]]; a = par[a]
                    return a
                for u in range(len(mem)):
                    for v in range(u + 1, len(mem)):
                        i, j = mem[u], mem[v]
                        ox = min(pos[i][0] + pos[i][2], pos[j][0] + pos[j][2]) \
                            - max(pos[i][0], pos[j][0])
                        oy = min(pos[i][1] + pos[i][3], pos[j][1] + pos[j][3]) \
                            - max(pos[i][1], pos[j][1])
                        if (ox > 1e-6 and oy >= -1e-6) or (oy > 1e-6 and ox >= -1e-6):
                            ra, rb = find(u), find(v)
                            if ra != rb:
                                par[ra] = rb
                gviol += len({find(u) for u in range(len(mem))}) - 1

            nbad = 0
            for i in range(bc):
                x, y, w, h = pos[i]
                g = int(cons[i, 3])
                req = _decode(cons[i, 4])
                hatch = "xxx" if int(cons[i, 1]) else ("///" if int(cons[i, 0]) else None)
                face = cmap((g * 3) % 20) if g > 0 else \
                    ("wheat" if int(cons[i, 1]) else
                     ("#cfe0f0" if req else "#eeeeee"))
                if mark_overlap:
                    edge = "red" if i in bad else "#8fa8bd"
                    lw = 1.4 if i in bad else 0.5
                else:
                    ok = all([("L" not in req or abs(x - x0) <= 1e-6),
                              ("R" not in req or abs(x + w - x1) <= 1e-6),
                              ("B" not in req or abs(y - y0) <= 1e-6),
                              ("T" not in req or abs(y + h - y1) <= 1e-6)]) if req else True
                    nbad += 0 if (ok or not req) else 1
                    edge = ("limegreen" if ok else "red") if req else \
                        ("darkviolet" if g > 0 else "#bbbbbb")
                    lw = 2.2 if req else (1.6 if g > 0 else 0.5)
                ax.add_patch(mp.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                                          linewidth=lw, hatch=hatch,
                                          alpha=0.55 if mark_overlap else 0.85))
                ax.text(x + w / 2, y + h / 2, str(i), ha="center", va="center", fontsize=4.5)

            blk = sum(p[2] * p[3] for p in pos)
            util = blk / max((x1 - x0) * (y1 - y0), 1e-9)
            sub = f"{x1-x0:.0f} x {y1-y0:.0f}   util={util:.0%}   grp-viol={gviol}"
            sub += f"   overlap={100*ov_area/blk:.0f}%" if mark_overlap \
                else f"   off-edge={nbad}"
            ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                      edgecolor="black", linewidth=1.0, linestyle="--"))
            ax.set_title(f"{name}\n{sub}", fontsize=9)
            ax.set_xlim(x0 - 0.04 * (x1 - x0), x1 + 0.04 * (x1 - x0))
            ax.set_ylim(y0 - 0.04 * (y1 - y0), y1 + 0.04 * (y1 - y0))
            ax.set_aspect("equal"); ax.tick_params(labelsize=5)

        fig.suptitle(f"case {idx}  ({bc} blocks)", fontsize=13)
        fp = out / f"stages_{idx:03d}.png"
        fig.tight_layout(); fig.savefig(fp, dpi=95); plt.close(fig)
        print(f"case {idx:3d} -> {fp.name}")


if __name__ == "__main__":
    main()
