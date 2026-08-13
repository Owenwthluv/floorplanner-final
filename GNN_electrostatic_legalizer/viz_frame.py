#!/usr/bin/env python3
"""
Clean render of STEP-1 frame construction: boundary blocks solid (green if they
touch the frame edge they require, red otherwise), interior blocks faint (they
are untouched at their density positions), and the frame box W x H dashed.

Usage:
  python viz_frame.py gnn_frame_only_solutions.json --ids 0,2,3,29,69
"""
import argparse
import json
from pathlib import Path

CONTEST_DIR = Path(__file__).resolve().parent


def _decode(c):
    c = int(c)
    return "".join(k for k, b in (("L", c & 1), ("R", c & 2), ("T", c & 4), ("B", c & 8)) if b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--ids", default="0,2,3,29,69")
    ap.add_argument("--out-dir", default=str(CONTEST_DIR / "viz_frame"))
    ap.add_argument("--data-path", default=str(CONTEST_DIR.parent))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    from iccad2026_evaluate import ContestEvaluator

    with open(args.results) as f:
        sols = {int(s["test_id"]): s for s in json.load(f).get("solutions", [])}
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ev = ContestEvaluator(data_path=args.data_path, verbose=False); ev._load_dataset()
    cmap = plt.cm.tab20

    for idx in [int(s) for s in args.ids.split(",") if s.strip()]:
        if idx not in sols:
            continue
        inp = ev.dataset[idx]["input"]
        area, b2b, p2b, pins, cons = inp
        bc = int((area != -1).sum().item())
        pos = [tuple(float(v) for v in p) for p in sols[idx]["positions"]]
        fixed = cons[:, 0]; pre = cons[:, 1]; clust = cons[:, 3]; bound = cons[:, 4]

        bnd_ids = [i for i in range(bc) if int(bound[i]) != 0]
        # frame MARKS = median coordinate of each edge's blocks (robust to a
        # preplaced block left sticking out by method B).
        import statistics as _st

        def _med(vals):
            return _st.median(vals) if vals else None
        mL = _med([pos[i][0] for i in bnd_ids if "L" in _decode(bound[i])])
        mR = _med([pos[i][0] + pos[i][2] for i in bnd_ids if "R" in _decode(bound[i])])
        mB = _med([pos[i][1] for i in bnd_ids if "B" in _decode(bound[i])])
        mT = _med([pos[i][1] + pos[i][3] for i in bnd_ids if "T" in _decode(bound[i])])
        fx0 = mL if mL is not None else min(pos[i][0] for i in bnd_ids)
        fy0 = mB if mB is not None else min(pos[i][1] for i in bnd_ids)
        fx1 = mR if mR is not None else max(pos[i][0] + pos[i][2] for i in bnd_ids)
        fy1 = mT if mT is not None else max(pos[i][1] + pos[i][3] for i in bnd_ids)
        tol = 0.02 * max(fx1 - fx0, fy1 - fy0)

        def on_frame(i):
            x, y, w, h = pos[i]
            return (abs(x - fx0) <= tol or abs(x + w - fx1) <= tol or
                    abs(y - fy0) <= tol or abs(y + h - fy1) <= tol)

        mib = cons[:, 2]
        # grouping violations per cluster: connected components - 1
        def _components(members):
            par = list(range(len(members)))
            def find(a):
                while par[a] != a:
                    par[a] = par[par[a]]; a = par[a]
                return a
            for u in range(len(members)):
                for v in range(u + 1, len(members)):
                    i, j = members[u], members[v]
                    ox = min(pos[i][0] + pos[i][2], pos[j][0] + pos[j][2]) - max(pos[i][0], pos[j][0])
                    oy = min(pos[i][1] + pos[i][3], pos[j][1] + pos[j][3]) - max(pos[i][1], pos[j][1])
                    if (ox > 1e-6 and oy >= -1e-6) or (oy > 1e-6 and ox >= -1e-6):
                        ra, rb = find(u), find(v)
                        if ra != rb:
                            par[ra] = rb
            return len({find(u) for u in range(len(members))})
        split_groups, grp_viol = set(), 0
        for g in sorted({int(c) for c in clust[:bc]} - {0}):
            mem = [i for i in range(bc) if int(clust[i]) == g]
            c = _components(mem)
            grp_viol += c - 1
            if c > 1:
                split_groups.add(g)
        mib_viol = 0
        for m in sorted({int(v) for v in mib[:bc]} - {0}):
            mem = [i for i in range(bc) if int(mib[i]) == m]
            mib_viol += len({(round(pos[i][2], 4), round(pos[i][3], 4)) for i in mem}) - 1

        fig, ax = plt.subplots(figsize=(9, 8))

        def draw(i, boundary):
            """one block: fill = cluster colour, hatch = fixed/preplaced,
            edge = boundary satisfaction (boundary blocks) or cluster state."""
            x, y, w, h = pos[i]
            g, m = int(clust[i]), int(mib[i])
            req = _decode(bound[i])
            hatch = "xxx" if int(pre[i]) else ("///" if int(fixed[i]) else None)
            if g > 0:
                face, alpha = cmap((g * 3) % 20), 0.85
            elif boundary:
                face, alpha = ("wheat" if int(pre[i]) else "#cfe0f0"), 0.85
            elif on_frame(i):
                face, alpha = "#c9efe6", 0.8            # interior pulled onto the edge
            else:
                face, alpha = "#eeeeee", 0.5
            if boundary:
                ok = all([("L" not in req or abs(x - fx0) <= tol),
                          ("R" not in req or abs(x + w - fx1) <= tol),
                          ("B" not in req or abs(y - fy0) <= tol),
                          ("T" not in req or abs(y + h - fy1) <= tol)])
                edge, lw = ("limegreen" if ok else "red"), 2.6
            elif g > 0:
                edge, lw = ("red" if g in split_groups else "darkviolet"), 2.0
            elif m > 0:
                edge, lw = "darkorange", 1.8
            elif on_frame(i):
                edge, lw = "teal", 2.0
            else:
                edge, lw = "#bbbbbb", 0.5
            ax.add_patch(mp.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                                      linewidth=lw, hatch=hatch, alpha=alpha))
            tag = [str(i)]
            if req:
                tag.append(req)
            if g > 0:
                tag.append(f"g{g}")
            if m > 0:
                tag.append(f"m{m}")
            if not boundary and not req and g == 0 and m == 0 and on_frame(i):
                tag.append("fill")
            ax.text(x + w / 2, y + h / 2, "\n".join(tag), ha="center", va="center",
                    fontsize=6 if (boundary or g > 0 or m > 0) else 5,
                    color="black" if (boundary or g > 0 or m > 0) else "#888888")

        for i in range(bc):                              # interior first (behind)
            if int(bound[i]) == 0:
                draw(i, False)
        for i in bnd_ids:
            draw(i, True)
        ax.add_patch(mp.Rectangle((fx0, fy0), fx1 - fx0, fy1 - fy0, fill=False,
                                  edgecolor="black", linewidth=1.2, linestyle="--"))
        npp = sum(1 for i in bnd_ids if int(pre[i]))
        nbad = sum(1 for i in bnd_ids
                   if not all([("L" not in _decode(bound[i]) or abs(pos[i][0] - fx0) <= tol),
                               ("R" not in _decode(bound[i]) or abs(pos[i][0] + pos[i][2] - fx1) <= tol),
                               ("B" not in _decode(bound[i]) or abs(pos[i][1] - fy0) <= tol),
                               ("T" not in _decode(bound[i]) or abs(pos[i][1] + pos[i][3] - fy1) <= tol)]))
        nov = 0
        for u in range(len(bnd_ids)):
            for v in range(u + 1, len(bnd_ids)):
                i, j = bnd_ids[u], bnd_ids[v]
                oxx = min(pos[i][0] + pos[i][2], pos[j][0] + pos[j][2]) - max(pos[i][0], pos[j][0])
                oyy = min(pos[i][1] + pos[i][3], pos[j][1] + pos[j][3]) - max(pos[i][1], pos[j][1])
                if oxx > tol and oyy > tol:
                    nov += 1
        blk_area = sum(p[2] * p[3] for p in pos[:bc])
        util = blk_area / max((fx1 - fx0) * (fy1 - fy0), 1e-9)
        ax.set_title(f"case {idx}  frame {fx1-fx0:.0f} x {fy1-fy0:.0f}  util={util:.0%}  "
                     f"boundary={len(bnd_ids)} (pp={npp})  off-edge={nbad}  "
                     f"grp-viol={grp_viol}  mib-viol={mib_viol}  ov={nov}", fontsize=9)
        ax.legend(handles=[
            mp.Patch(facecolor="#cfe0f0", edgecolor="limegreen", lw=2, label="boundary OK"),
            mp.Patch(facecolor="#cfe0f0", edgecolor="red", lw=2, label="boundary violated"),
            mp.Patch(facecolor=cmap(3), edgecolor="darkviolet", lw=2, label="cluster (intact)"),
            mp.Patch(facecolor=cmap(6), edgecolor="red", lw=2, label="cluster (split)"),
            mp.Patch(facecolor="#eeeeee", edgecolor="darkorange", lw=2, label="MIB group"),
            mp.Patch(facecolor="#eeeeee", edgecolor="#bbbbbb", label="free interior"),
        ], loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)
        ax.set_xlim(fx0 - (fx1 - fx0) * 0.05, fx1 + (fx1 - fx0) * 0.05)
        ax.set_ylim(fy0 - (fy1 - fy0) * 0.05, fy1 + (fy1 - fy0) * 0.05)
        ax.set_aspect("equal"); ax.tick_params(labelsize=6)
        fp = out / f"frame_{idx:03d}.png"
        fig.tight_layout(); fig.savefig(fp, dpi=130); plt.close(fig)
        print(f"case {idx:3d}: boundary={len(bnd_ids)} preplaced={npp} off-edge={nbad} -> {fp.name}")


if __name__ == "__main__":
    main()
