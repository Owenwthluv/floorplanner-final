#!/usr/bin/env python3
"""
Replay the interior packing one block at a time.

Each frame shows what the packer knew at that moment: the blocks already on the
floor (solid), the block it has just placed (thick red), and every MAXRECTS free
rectangle it was choosing between (faint dashed).  Watching the free-rect set
shrink is the quickest way to see whether a pocket -- under a preplaced block,
under a bridge -- was ever offered as a candidate.

Usage:
  python viz_steps.py --case 99                     # every step
  python viz_steps.py --case 99 --every 5 --gif     # every 5th, plus an animation
  python viz_steps.py --case 99 --sheet             # one contact sheet instead
"""
import argparse
import os
import sys
from pathlib import Path

CONTEST_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=99)
    ap.add_argument("--out-dir", default=str(CONTEST_DIR / "viz_steps"))
    ap.add_argument("--every", type=int, default=1, help="render every Nth step")
    ap.add_argument("--gif", action="store_true", help="also write steps.gif")
    ap.add_argument("--sheet", action="store_true", help="one grid image instead")
    ap.add_argument("--max-frames", type=int, default=400)
    args = ap.parse_args()

    os.environ.setdefault("PACK_INTERIOR", "maxrects")
    sys.path.insert(0, str(CONTEST_DIR))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp

    import stage3_legalizer as FP

    steps = []

    def hook(block_id, placed, free, occ, W):
        steps.append({
            "id": int(block_id),
            "placed": tuple(float(v) for v in placed),      # (x0, x1, y0, y1)
            "free": [tuple(float(v) for v in r) for r in free],   # (x0, y0, x1, y1)
            "occ": [tuple(float(v) for v in r) for r in occ],     # (x0, x1, y0, y1)
            "W": float(W),
        })
    FP.STEP_HOOK = hook

    import iccad2026_evaluate as E
    sys.argv = ["x", "--evaluate", "my_optimizer.py", "--test-id", str(args.case)]
    try:
        E.main()
    except SystemExit:
        pass
    FP.STEP_HOOK = None

    if not steps:
        print("no steps captured -- is PACK_INTERIOR=maxrects and the case interior non-empty?")
        return
    print(f"captured {len(steps)} placements")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("step_*.png"):
        f.unlink()

    W = steps[-1]["W"]
    hi = max(max(r[3] for r in s["occ"]) for s in steps)
    idxs = list(range(0, len(steps), max(1, args.every)))
    if idxs[-1] != len(steps) - 1:
        idxs.append(len(steps) - 1)
    idxs = idxs[:args.max_frames]

    def draw(ax, s, k):
        # Free rectangles the packer could choose from.  MAXRECTS rects overlap
        # each other by construction, so they are drawn as OUTLINES -- stacking
        # translucent fills just turns the whole frame into one blue smear.
        for x0, y0, x1, y1 in s["free"]:
            ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                                      edgecolor="#2b7fd4", alpha=0.30,
                                      linewidth=0.5, linestyle=(0, (2, 2))))
        cur = s["placed"]
        for x0, x1, y0, y1 in s["occ"]:                 # everything already down
            is_cur = abs(x0 - cur[0]) < 1e-9 and abs(y0 - cur[2]) < 1e-9 \
                and abs(x1 - cur[1]) < 1e-9 and abs(y1 - cur[3]) < 1e-9
            if is_cur:
                continue
            ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#cfe0f0",
                                      edgecolor="#7f9bb5", linewidth=0.8, alpha=0.9))
        ax.add_patch(mp.Rectangle((cur[0], cur[2]), cur[1] - cur[0], cur[3] - cur[2],
                                  facecolor="#ffb3b3", edgecolor="red", linewidth=2.0))
        ax.text((cur[0] + cur[1]) / 2, (cur[2] + cur[3]) / 2, str(s["id"]),
                ha="center", va="center", fontsize=6)
        ax.add_patch(mp.Rectangle((0, 0), W, hi, fill=False, edgecolor="black",
                                  linewidth=1.0, linestyle="--"))
        ax.set_xlim(-0.03 * W, 1.03 * W)
        ax.set_ylim(-0.03 * hi, 1.03 * hi)
        ax.set_aspect("equal")
        ax.set_title(f"step {k + 1}/{len(steps)}  block {s['id']}  "
                     f"free rects={len(s['free'])}", fontsize=8)
        ax.tick_params(labelsize=5)

    if args.sheet:
        import math
        cols = min(6, len(idxs))
        rows = math.ceil(len(idxs) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.2 * rows))
        axes = [axes] if rows * cols == 1 else list(axes.flat)
        for ax, k in zip(axes, idxs):
            draw(ax, steps[k], k)
        for ax in axes[len(idxs):]:
            ax.axis("off")
        fp = out / f"sheet_case{args.case:03d}.png"
        fig.tight_layout(); fig.savefig(fp, dpi=110); plt.close(fig)
        print(f"-> {fp}")
        return

    files = []
    for k in idxs:
        fig, ax = plt.subplots(figsize=(6, 6 * hi / max(W, 1e-9)))
        draw(ax, steps[k], k)
        fp = out / f"step_{k:04d}.png"
        fig.tight_layout(); fig.savefig(fp, dpi=110); plt.close(fig)
        files.append(fp)
    print(f"-> {len(files)} frames in {out}")

    if args.gif:
        try:
            from PIL import Image
        except ImportError:
            print("(no Pillow -- skipping the gif)")
            return
        ims = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
        gp = out / f"steps_case{args.case:03d}.gif"
        ims[0].save(gp, save_all=True, append_images=ims[1:], duration=220, loop=0)
        print(f"-> {gp}")


if __name__ == "__main__":
    main()
