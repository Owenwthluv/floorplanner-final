#!/usr/bin/env python3
"""
Side-by-side: our packed result against the dataset's ground-truth floorplan.

Both panels use the same colour scheme as viz_frame.py -- fill by cluster,
hatch for fixed/preplaced, edge colour by whether a boundary block reaches the
edge it is coded for -- so the two can be read against each other directly.

  python viz_gt.py --case 35 --solutions gnn_density_pack3_solutions.json
"""
import argparse
import json
from pathlib import Path

CONTEST_DIR = Path(__file__).resolve().parent


def _decode(c):
    c = int(c)
    return "".join(k for k, b in (("L", c & 1), ("R", c & 2),
                                  ("T", c & 4), ("B", c & 8)) if b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=35)
    ap.add_argument("--solutions", default="gnn_density_pack3_solutions.json")
    ap.add_argument("--out-dir", default=str(CONTEST_DIR / "viz_steps"))
    ap.add_argument("--data-path", default=str(CONTEST_DIR.parent))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    from iccad2026_evaluate import ContestEvaluator

    ev = ContestEvaluator(data_path=args.data_path, verbose=False)
    ev._load_dataset()
    rec = ev.dataset[args.case]
    area, b2b, p2b, pins, cons = rec["input"]
    bc = int((area != -1).sum().item())

    with open(args.solutions) as f:
        sols = {int(s["test_id"]): s for s in json.load(f).get("solutions", [])}
    ours = [tuple(float(v) for v in p) for p in sols[args.case]["positions"]][:bc]

    # ground truth: one closed 5-point polygon per block -> its bounding box
    poly = rec["label"][0]
    gt = []
    for i in range(bc):
        xs = [float(v) for v in poly[i][:, 0]]
        ys = [float(v) for v in poly[i][:, 1]]
        gt.append((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))

    fixed, pre = cons[:, 0], cons[:, 1]
    mib, clust, bound = cons[:, 2], cons[:, 3], cons[:, 4]
    cmap = plt.cm.tab20

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    for ax, pos, name in ((axes[0], ours, "ours"), (axes[1], gt, "ground truth")):
        x0 = min(p[0] for p in pos); y0 = min(p[1] for p in pos)
        x1 = max(p[0] + p[2] for p in pos); y1 = max(p[1] + p[3] for p in pos)
        tol = 1e-6
        bnd_ids = [i for i in range(bc) if int(bound[i])]

        # cluster components, to colour split groups red
        split = set()
        gviol = 0
        for g in sorted({int(c) for c in clust[:bc]} - {0}):
            mem = [i for i in range(bc) if int(clust[i]) == g]
            par = list(range(len(mem)))

            def find(a):
                while par[a] != a:
                    par[a] = par[par[a]]; a = par[a]
                return a
            for u in range(len(mem)):
                for v in range(u + 1, len(mem)):
                    i, j = mem[u], mem[v]
                    ox = min(pos[i][0] + pos[i][2], pos[j][0] + pos[j][2]) - max(pos[i][0], pos[j][0])
                    oy = min(pos[i][1] + pos[i][3], pos[j][1] + pos[j][3]) - max(pos[i][1], pos[j][1])
                    if (ox > 1e-6 and oy >= -1e-6) or (oy > 1e-6 and ox >= -1e-6):
                        ra, rb = find(u), find(v)
                        if ra != rb:
                            par[ra] = rb
            comps = len({find(u) for u in range(len(mem))})
            gviol += comps - 1
            if comps > 1:
                split.add(g)

        nbad = 0
        for i in range(bc):
            x, y, w, h = pos[i]
            g, m = int(clust[i]), int(mib[i])
            req = _decode(bound[i])
            hatch = "xxx" if int(pre[i]) else ("///" if int(fixed[i]) else None)
            if g > 0:
                face, alpha = cmap((g * 3) % 20), 0.85
            elif req:
                face, alpha = ("wheat" if int(pre[i]) else "#cfe0f0"), 0.85
            else:
                face, alpha = "#eeeeee", 0.5
            if req:
                ok = all([("L" not in req or abs(x - x0) <= tol),
                          ("R" not in req or abs(x + w - x1) <= tol),
                          ("B" not in req or abs(y - y0) <= tol),
                          ("T" not in req or abs(y + h - y1) <= tol)])
                nbad += 0 if ok else 1
                edge, lw = ("limegreen" if ok else "red"), 2.4
            elif g > 0:
                edge, lw = ("red" if g in split else "darkviolet"), 2.0
            elif m > 0:
                edge, lw = "darkorange", 1.6
            else:
                edge, lw = "#bbbbbb", 0.5
            ax.add_patch(mp.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                                      linewidth=lw, hatch=hatch, alpha=alpha))
            tag = [str(i)] + ([req] if req else []) + \
                  ([f"g{g}"] if g else []) + ([f"m{m}"] if m else [])
            ax.text(x + w / 2, y + h / 2, "\n".join(tag), ha="center", va="center",
                    fontsize=5.5, color="black" if (req or g or m) else "#888888")

        blk = sum(p[2] * p[3] for p in pos)
        util = blk / max((x1 - x0) * (y1 - y0), 1e-9)
        ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                  edgecolor="black", linewidth=1.1, linestyle="--"))
        ax.set_title(f"{name}   {x1-x0:.0f} x {y1-y0:.0f}   util={util:.0%}   "
                     f"off-edge={nbad}   grp-viol={gviol}", fontsize=10)
        ax.set_xlim(x0 - 0.04 * (x1 - x0), x1 + 0.04 * (x1 - x0))
        ax.set_ylim(y0 - 0.04 * (y1 - y0), y1 + 0.04 * (y1 - y0))
        ax.set_aspect("equal"); ax.tick_params(labelsize=6)

    fig.suptitle(f"case {args.case}  ({bc} blocks)", fontsize=12)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fp = out / f"compare_case{args.case:03d}.png"
    fig.tight_layout(); fig.savefig(fp, dpi=130); plt.close(fig)
    print(f"-> {fp}")


if __name__ == "__main__":
    main()
