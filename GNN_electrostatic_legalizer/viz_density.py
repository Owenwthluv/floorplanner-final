#!/usr/bin/env python3
"""
Render the layout as the density stage leaves it -- i.e. what the packer
actually receives.  Overlapping pairs are outlined in red, so it is visible at a
glance how much (or how little) the electrostatic spreading actually untangled.

  python viz_density.py --case 94 --caps 3.0,0.5
"""
import argparse
import os
import sys
from pathlib import Path

CONTEST_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=94)
    ap.add_argument("--caps", default="3.0,0.5", help="cap_end values to compare")
    ap.add_argument("--out-dir", default=str(CONTEST_DIR / "viz_density"))
    args = ap.parse_args()
    caps = [float(c) for c in args.caps.split(",")]

    sys.path.insert(0, str(CONTEST_DIR))
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp

    import stage3_legalizer as FP
    orig = FP.stage3_legalizer
    CAP = {}

    def spy(n, P, ipp, cons, util=None, **kw):
        CAP["P"] = P[:n].copy()
        CAP["pp"] = [bool(ipp[i]) for i in range(n)]
        CAP["clust"] = [int(cons[i, 3]) for i in range(n)]
        CAP["n"] = n
        return orig(n, P, ipp, cons, util, **kw)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(caps), figsize=(7.5 * len(caps), 8))
    axes = [axes] if len(caps) == 1 else list(axes)
    cmap = plt.cm.tab20

    for ax, cap in zip(axes, caps):
        os.environ["LD_CAP_END"] = str(cap)
        FP.stage3_legalizer = spy
        import my_optimizer as M
        M.stage3_legalizer = spy
        import iccad2026_evaluate as E
        sys.argv = ["x", "--evaluate", "my_optimizer.py",
                    "--test-id", str(args.case)]
        try:
            E.main()
        except SystemExit:
            pass

        P, n = CAP["P"], CAP["n"]
        pp, clust = CAP["pp"], CAP["clust"]
        # every block that overlaps at least one other
        bad = set()
        ov_area = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                ox = min(P[i, 0] + P[i, 2], P[j, 0] + P[j, 2]) - max(P[i, 0], P[j, 0])
                oy = min(P[i, 1] + P[i, 3], P[j, 1] + P[j, 3]) - max(P[i, 1], P[j, 1])
                if ox > 1e-6 and oy > 1e-6:
                    bad.add(i); bad.add(j); ov_area += ox * oy
        blk = float(sum(P[i, 2] * P[i, 3] for i in range(n)))
        x0 = min(P[i, 0] for i in range(n)); y0 = min(P[i, 1] for i in range(n))
        x1 = max(P[i, 0] + P[i, 2] for i in range(n))
        y1 = max(P[i, 1] + P[i, 3] for i in range(n))

        for i in range(n):
            x, y, w, h = P[i]
            g = clust[i]
            face = cmap((g * 3) % 20) if g > 0 else ("wheat" if pp[i] else "#dfe8f2")
            edge = "red" if i in bad else "#7f9bb5"
            ax.add_patch(mp.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                                      linewidth=1.6 if i in bad else 0.6, alpha=0.55))
            ax.text(x + w / 2, y + h / 2, str(i), ha="center", va="center", fontsize=5)
        ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                  edgecolor="black", linewidth=1.0, linestyle="--"))
        ax.set_title(f"cap_end={cap}   span {x1-x0:.0f} x {y1-y0:.0f}\n"
                     f"{len(bad)}/{n} blocks overlapping   "
                     f"overlap area = {100*ov_area/blk:.0f}% of block area",
                     fontsize=10)
        ax.set_xlim(x0 - 0.04 * (x1 - x0), x1 + 0.04 * (x1 - x0))
        ax.set_ylim(y0 - 0.04 * (y1 - y0), y1 + 0.04 * (y1 - y0))
        ax.set_aspect("equal"); ax.tick_params(labelsize=6)

    fig.suptitle(f"case {args.case} -- what the packer receives from the density stage",
                 fontsize=12)
    fp = out / f"density_case{args.case:03d}.png"
    fig.tight_layout(); fig.savefig(fp, dpi=130); plt.close(fig)
    print(f"-> {fp}")


if __name__ == "__main__":
    main()
