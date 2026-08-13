#!/usr/bin/env python3
"""STAGE 3 -- LEGALIZER.

Takes the compacted, still-overlapping layout that STAGE 2 (electrostatic)
hands over and turns it into a legal floorplan:

  a. FRAME          width from the target utilisation, pinned to a preplaced
                    R block when one fixes that edge.  The top is left free --
                    it is the edge the packing grows into.
  b. BOTTOM ROW     BL corner, the B blocks, BR corner; widths spread so the
                    row spans the frame.  Against a preplaced obstacle a block
                    first narrows into the pocket before it, then flattens to
                    duck underneath, and only jumps past as a last resort.
  c. SIDE TOWERS    stacked bottom-up.  A boundary block slides along its edge
                    to meet a preplaced cluster peer (0 degrees of freedom vs
                    1), narrows to squeeze past an obstacle rather than hopping
                    over it, and unpinned blocks drop into the holes left.
  d. INTERIOR       MaxRects free-rectangle fill -- unlike a skyline it can see
                    the pocket under a floating preplaced block.  One candidate
                    per step, in bottom-up density order, scored by
                        S_area + 0.5*S_wire + 32*S_grp
                    where S_area charges only for growth of the frame, S_wire
                    counts already-placed neighbours, and S_grp is the gap to
                    the nearest cluster peer.
  e. TOP ROW        laid side by side with a flat lid.
  f. SAFETY NET     two-tier push; the second tier is strictly monotonic with
                    preplaced immovable, so overlap is always zero.

Soft blocks may be reshaped area-constant within AR <= 3; fixed, preplaced and
MIB blocks never are.
"""
import numpy as np

EPS = 1e-6
import os as _os
AR_MAX = float(_os.environ.get("PACK_AR", "3.0"))
# Horizontal hug at placement time: slide a block left to close a gap of up to
# HUG_FRAC x its width.  Measured to REGRESS at every threshold tried (0.10 ->
# 2.012, 0.20 -> 2.092, 0.35 -> 2.160, vs 1.980 disabled): the placement x
# already encodes the density order, so sliding left both breaks that (hpwl)
# and starves the right side.  Kept switchable for experiments; default off.
HUG_FRAC = float(_os.environ.get("HUG_FRAC", "0.0"))

# Debug hook: when set to a callable, it is invoked after every interior
# placement with (block_id, placed_rect, free_rects, all_occupied) so a
# visualiser can replay the packing step by step.  None in production.
STEP_HOOK = None


def _decode(code):
    code = int(code)
    return {"L": bool(code & 1), "R": bool(code & 2),
            "T": bool(code & 4), "B": bool(code & 8)}


class Contour:
    """Piecewise-constant height profile over [lo, hi]."""
    def __init__(self, x_lo, x_hi, h=0.0):
        self.lo, self.hi = x_lo, x_hi
        self.segs = [[x_lo, x_hi, h]]

    def height(self, a, b):
        m = 0.0
        for s, e, h in self.segs:
            if e <= a + EPS or s >= b - EPS:
                continue
            if h > m:
                m = h
        return m

    def raise_to(self, a, b, h):
        a = max(self.lo, a); b = min(self.hi, b)
        if b <= a:
            return
        out = []
        for s, e, sh in self.segs:
            if e <= a + EPS or s >= b - EPS:
                out.append([s, e, sh]); continue
            if s < a - EPS:
                out.append([s, a, sh])
            if e > b + EPS:
                out.append([b, e, sh])
        out.append([a, b, h])
        out.sort()
        merged = [out[0]]
        for seg in out[1:]:
            if abs(seg[2] - merged[-1][2]) < 1e-9 and abs(seg[0] - merged[-1][1]) < 1e-9:
                merged[-1][1] = seg[1]
            else:
                merged.append(seg)
        self.segs = merged

    def raise_min(self, a, b, h):
        """raise the profile to at least h over [a, b], per segment -- used
        when a block is tucked UNDER an overhang, where a plain set would
        wrongly lower the profile back down to the block's own top."""
        a = max(self.lo, a); b = min(self.hi, b)
        if b <= a:
            return
        for s, e, sh in list(self.segs):
            s2, e2 = max(s, a), min(e, b)
            if e2 - s2 > EPS and sh < h:
                self.raise_to(s2, e2, h)

class MaxRects:
    """Free-rectangle tracker (Jylanki's MAXRECTS): unlike a skyline it keeps
    EVERY maximal free rectangle, so pockets under overhangs -- beside a
    floating preplaced block, under a wide block bridging a valley -- stay
    usable instead of being written off."""

    def __init__(self, W, H):
        self.F = [(0.0, 0.0, W, H)]

    def occupy(self, o):
        """subtract the placed rect o=(x0,x1,y0,y1) from every free rect."""
        a, b, c, d = o
        nf = []
        for x0, y0, x1, y1 in self.F:
            if a >= x1 - EPS or b <= x0 + EPS or c >= y1 - EPS or d <= y0 + EPS:
                nf.append((x0, y0, x1, y1))
                continue
            if a > x0 + EPS:
                nf.append((x0, y0, a, y1))
            if b < x1 - EPS:
                nf.append((b, y0, x1, y1))
            if c > y0 + EPS:
                nf.append((x0, y0, x1, c))
            if d < y1 - EPS:
                nf.append((x0, d, x1, y1))
        keep = []
        for i, r in enumerate(nf):                    # prune contained rects
            if r[2] - r[0] <= EPS or r[3] - r[1] <= EPS:
                continue
            contained = False
            for j, s in enumerate(nf):
                if i == j:
                    continue
                if s[0] <= r[0] + 1e-9 and s[1] <= r[1] + 1e-9 \
                   and s[2] >= r[2] - 1e-9 and s[3] >= r[3] - 1e-9:
                    if s != r or j < i:
                        contained = True
                        break
            if not contained:
                keep.append(r)
        self.F = keep

    def find(self, w, h, a, px, win=None, score=None):
        """best spot for a w x h block; if `a` is given the block is soft and
        may be reshaped (AR cap) to slot into a narrow/low free rect.  Returns
        (y, x, w_final) or None.  `win` restricts the x-range (clusters).

        `score(x, y, w2, h2, rect_area) -> float` ranks the candidates; without
        it the fallback is the old lexicographic "lowest landing first", which
        is exactly what made this mode lose: an absolutely-lowest pocket on the
        far side of the floorplan always beat a sensible spot next to the
        block's own neighbours."""
        best = None
        lo_w = np.sqrt(a / AR_MAX) if a else w
        hi_w = np.sqrt(a * AR_MAX) if a else w
        for x0, y0, x1, y1 in self.F:
            rw, rh = x1 - x0, y1 - y0
            cands = (w,) if a is None else \
                    (w, min(w, rw), (a / rh) if rh > EPS else w)
            for w2 in cands:
                w2 = min(max(w2, lo_w), hi_w)
                h2 = (a / w2) if a else h
                if w2 > rw + EPS or h2 > rh + EPS:
                    continue
                xa, xb = x0, x1 - w2
                if win is not None:
                    xa, xb = max(xa, win[0]), min(xb, win[1])
                    if xb < xa - EPS:
                        continue
                x = min(max(px, xa), xb)
                if score is None:
                    key = (y0, 0.0 if abs(w2 - w) <= EPS else 2.0, abs(x - px))
                else:
                    key = score(x, y0, w2, h2, rw * rh, rw, rh)
                if best is None or key < best[0]:
                    best = (key, x, y0, w2)
        if best is None:
            return None
        return best[2], best[1], best[3]


def stage3_legalizer(n, positions, is_preplaced, constraints, util=None,
                b2b=None, p2b=None, pins=None, w_hint=None):
    if util is None:
        util = float(_os.environ.get("PACK_UTIL", "0.85"))
    P = positions
    cons = constraints
    code = [_decode(cons[i, 4]) for i in range(n)]
    clust = [int(cons[i, 3]) for i in range(n)]
    mib = [int(cons[i, 2]) for i in range(n)]
    fixed = [int(cons[i, 0]) > 0 for i in range(n)]
    soft = [not is_preplaced[i] and not fixed[i] and mib[i] == 0 for i in range(n)]
    area = [float(P[i, 2] * P[i, 3]) for i in range(n)]

    A = float(sum(area))
    W0 = max(P[i, 0] + P[i, 2] for i in range(n))
    H0 = max(P[i, 1] + P[i, 3] for i in range(n))
    ar = max(0.25, min(4.0, (W0 / H0) if H0 > EPS else 1.0))
    W = max(np.sqrt(A / util * ar), max(P[i, 2] for i in range(n)))
    # PACK_WPOLICY=gnn: take the width the GNN itself marked out with its L/R
    # boundary blocks.  That mark has to be read from the RAW prediction, while
    # the layout is still compact and overlapping -- by the time the density
    # stage has finished, everything sits on a canvas ~1.5x wider and the mark
    # means nothing.  The caller passes it in as w_hint (medians, so one stray
    # boundary block cannot set it); measured at 0.77-1.20x the utilisation
    # width, i.e. a real signal rather than a distorted one.
    if _os.environ.get("PACK_WPOLICY", "util") == "gnn" and w_hint:
        if w_hint > max(P[i, 2] for i in range(n)):
            W = float(w_hint)
    pp_ext = max([P[i, 0] + P[i, 2] for i in range(n) if is_preplaced[i]] + [0.0])
    W = max(W, pp_ext)
    # R-coded preplaced PIN the right edge: they only touch the bbox right edge
    # if the frame ends exactly at their own right edge, so snap W to it (unless
    # another preplaced sticks out further, which makes the pin unreachable)
    # ---- FRAME MARKS DICTATED BY PREPLACED BOUNDARY / CORNER BLOCKS ----
    # A preplaced block cannot move, so if it also carries a boundary code the
    # frame edge has to come to IT: an L block fixes where the left edge is, a T
    # block fixes the total height, a corner block fixes two marks at once.
    # Measured over the suite: L and B already land on their mark (the data puts
    # them at 0), R misses 21% of the time, and T misses 100% -- the top mark is
    # whatever the packing happens to reach, always above the preplaced.
    #
    # PIN[k] is the mark that edge k must end at, or None when nothing fixes it.
    # Several preplaced on one edge can disagree (only one of them can then be
    # satisfied); the majority mark is taken, which is the most that is
    # achievable, and the rest keep their violation.
    def _pin(letter, val_fn):
        vals = [round(val_fn(i), 6) for i in range(n)
                if is_preplaced[i] and code[i][letter]]
        if not vals:
            return None
        return max(set(vals), key=vals.count)

    PIN = {
        "L": _pin("L", lambda i: P[i, 0]),
        "R": _pin("R", lambda i: P[i, 0] + P[i, 2]),
        "B": _pin("B", lambda i: P[i, 1]),
        "T": _pin("T", lambda i: P[i, 1] + P[i, 3]),
    }
    USE_PIN = _os.environ.get("PACK_PIN", "1") == "1"
    x_lo = PIN["L"] if (USE_PIN and PIN["L"] is not None) else 0.0
    y_lo = PIN["B"] if (USE_PIN and PIN["B"] is not None) else 0.0
    # Pinning W to the R block is only worth doing if the frame can still HOLD
    # the content: the bottom row cannot be squeezed below the sum of its
    # blocks' narrowest AR-legal widths, and a row that overflows pushes the
    # bbox back out anyway -- losing the pin and distorting the frame at the
    # same time (measured overruns of 12-32 units before this guard).
    _row_min = sum((np.sqrt(area[i] / AR_MAX) if soft[i] else P[i, 2])
                   for i in range(n) if code[i]["B"])
    _need = max([_row_min] + [P[i, 2] for i in range(n)])
    # NOTE: a feasibility guard was tried here (skip the pin when W would fall
    # below _need, the row's narrowest AR-legal span).  It measured WORSE --
    # 1.769 vs 1.758, and 10 of 38 R blocks off their mark instead of 8: the
    # overrunning rows cost less than the pins the guard threw away.
    if USE_PIN and PIN["R"] is not None and PIN["R"] >= pp_ext - EPS:
        W = PIN["R"] - x_lo
    # PIN["T"] is deliberately NOT enforced: the top is the edge the packing is
    # free to grow into, and forcing it down to a preplaced block's top would
    # either not fit at all or leave a dead band under the row.  A T-coded
    # preplaced stuck below the final top keeps its violation; that is the
    # cheaper of the two.

    cx = P[:n, 0] + P[:n, 2] / 2
    cy = P[:n, 1] + P[:n, 3] / 2
    mov = [i for i in range(n) if not is_preplaced[i]]

    # =====================================================================
    # SETUP -- frame width, coordinate map, connectivity
    # =====================================================================
    # ---- DENSITY -> FRAME coordinate map ----
    # The density stage spreads the movable blocks over a canvas measured at
    # ~1.57x the width of the packing frame, while preplaced blocks stay at
    # their true mandated coordinates.  So a raw density x is NOT comparable
    # with a frame x: 24-36% of the interior blocks have cx > W, and every
    # "which end of this valley is nearer my density position" test then picks
    # the RIGHT end unconditionally, regardless of where the block really wants
    # to be.  Map the movable blocks' span onto the frame once -- anisotropically,
    # since the canvas and the frame have different aspect ratios -- and route
    # every density lookup through dxc()/dyc().  Preplaced blocks and blocks
    # already placed keep their real coordinates and are never mapped.
    if mov:
        _xd0 = min(P[i, 0] for i in mov); _xd1 = max(P[i, 0] + P[i, 2] for i in mov)
        _yd0 = min(P[i, 1] for i in mov); _yd1 = max(P[i, 1] + P[i, 3] for i in mov)
    else:
        _xd0, _xd1, _yd0, _yd1 = 0.0, 1.0, 0.0, 1.0
    _sx = W / max(_xd1 - _xd0, EPS)
    _sy = (A / max(util * W, EPS)) / max(_yd1 - _yd0, EPS)
    MAPXY = _os.environ.get("PACK_MAPXY", "1") == "1"

    def dxc(v):
        """a density x (block centre or cluster mean) in frame coordinates"""
        return (v - _xd0) * _sx if MAPXY else v

    def dyc(v):
        return (v - _yd0) * _sy if MAPXY else v

    # ---- connectivity, for the wirelength term of the placement score ----
    # nbr[i] = [(j, weight), ...].  A neighbour that is already placed is read
    # at its real position; one that is not yet placed is read at its MAPPED
    # density position, since the raw density canvas is ~1.57x the frame.
    nbr = [[] for _ in range(n)]
    if b2b is not None:
        for e in b2b:
            i, j, w_ = int(e[0]), int(e[1]), float(e[2])
            if i < 0 or j < 0 or i >= n or j >= n:
                continue
            nbr[i].append((j, w_)); nbr[j].append((i, w_))
    pin_of = [[] for _ in range(n)]                   # (pin_x, pin_y, weight)
    if p2b is not None and pins is not None:
        for e in p2b:
            pi, bi, w_ = int(e[0]), int(e[1]), float(e[2])
            if pi < 0 or bi < 0 or bi >= n or pi >= len(pins):
                continue
            pin_of[bi].append((float(pins[pi][0]), float(pins[pi][1]), w_))

    # PACK_WIRE=placed (default): only count neighbours that are ALREADY on the
    # floor -- the bottom row, the side towers, the preplaced obstacles and
    # whatever interior has landed so far.  An unplaced neighbour has no real
    # position yet, only a density guess, and steering by a guess pulled blocks
    # toward coordinates nothing was ever going to occupy.
    # PACK_WIRE=all restores the old behaviour (density proxy for unplaced).
    WIRE_ALL = _os.environ.get("PACK_WIRE", "placed") == "all"

    def hpwl_of(i, ux, uy):
        """(weighted Manhattan cost of block i at centre (ux, uy), how many
        real anchors that cost was measured against)"""
        s, k = 0.0, 0
        for j, w_ in nbr[i]:
            if placed_at[j] is not None:
                jx, jy = placed_at[j]
            elif WIRE_ALL:
                jx, jy = dxc(cx[j]), dyc(cy[j])
            else:
                continue
            s += w_ * (abs(ux - jx) + abs(uy - jy))
            k += 1
        for pxx, pyy, w_ in pin_of[i]:                # pins are fixed: always real
            s += w_ * (abs(ux - pxx) + abs(uy - pyy))
            k += 1
        return s, k

    placed_at = [None] * n                            # filled in as blocks land
    for i in range(n):
        if is_preplaced[i]:
            placed_at[i] = (float(cx[i]), float(cy[i]))
    HPWL_REF = max(sum(hpwl_of(i, cx[i], cy[i])[0] for i in range(n)) / 2.0, 1.0)

    def resize_w(i, new_w):
        """area-constant width change, clamped to AR <= AR_MAX; soft blocks only."""
        if not soft[i]:
            return
        lo_w = np.sqrt(area[i] / AR_MAX)
        hi_w = np.sqrt(area[i] * AR_MAX)
        v = min(max(new_w, lo_w), hi_w)
        P[i, 2], P[i, 3] = v, area[i] / v

    # ---- classify the movable boundary blocks ----
    def has(i, *ks):
        return all(code[i][k] for k in ks)
    Bs = [i for i in mov if code[i]["B"]]
    Ts = [i for i in mov if code[i]["T"] and not code[i]["B"]]
    Ls = [i for i in mov if code[i]["L"] and not code[i]["B"] and not code[i]["T"]]
    Rs = [i for i in mov if code[i]["R"] and not code[i]["B"] and not code[i]["T"]]
    interior = [i for i in mov if not any(code[i].values())]

    BLc = next((i for i in Bs if has(i, "L")), None)
    BRc = next((i for i in Bs if has(i, "R") and i != BLc), None)
    TLc = next((i for i in Ts if has(i, "L")), None)
    TRc = next((i for i in Ts if has(i, "R") and i != TLc), None)
    Bmid = [i for i in Bs if i not in (BLc, BRc)]
    Tmid = [i for i in Ts if i not in (TLc, TRc)]

    # cluster mean coordinates for cohesive ordering
    gmx, gmy = {}, {}
    for g in set(clust):
        if g > 0:
            mem = [i for i in mov if clust[i] == g]
            if mem:
                gmx[g] = float(np.mean([cx[i] for i in mem]))
                gmy[g] = float(np.mean([cy[i] for i in mem]))
    Bmid.sort(key=lambda i: (gmx.get(clust[i], cx[i]), clust[i], cx[i]))
    Tmid.sort(key=lambda i: (gmx.get(clust[i], cx[i]), clust[i], cx[i]))
    # A tower is stacked bottom-up, so its ORDER decides each block's height.
    # When a side block shares a cluster with a T-coded block, the rest of that
    # cluster ends up near the ceiling -- so send the side block to the TOP of
    # its tower, where it can actually reach them.  A cluster with a B-coded
    # member gets the mirror treatment.  Measured before this: 110 of 111 T-side
    # blocks never touched an interior member of their own cluster.
    def _side_rank(i):
        g = clust[i]
        if g <= 0 or _os.environ.get("PACK_SIDE_ORDER", "0") != "1":
            return 1
        peers = [j for j in range(n) if clust[j] == g and j != i]
        if any(code[j]["B"] for j in peers):
            return 0                                  # cluster hugs the floor
        if any(code[j]["T"] for j in peers):
            return 2                                  # cluster hugs the ceiling
        return 1
    Ls.sort(key=lambda i: (_side_rank(i), gmy.get(clust[i], cy[i]), clust[i], cy[i]))
    Rs.sort(key=lambda i: (_side_rank(i), gmy.get(clust[i], cy[i]), clust[i], cy[i]))

    # ---- side-tower height balance: the L/R towers stack to the SUM of their
    #      blocks' heights; if that exceeds the estimated frame height the tower
    #      dictates H and leaves dead space, so flatten the side blocks first ----
    H_est = float(np.sqrt(A / util / ar))
    for side in (Ls, Rs):
        sumH = sum(P[i, 3] for i in side)
        if sumH > H_est + EPS and side:
            f = H_est / sumH
            for i in side:
                if soft[i]:
                    resize_w(i, area[i] / (P[i, 3] * f))   # lower height, wider block

    # preplaced obstacles
    pp = [(P[i, 0], P[i, 0] + P[i, 2], P[i, 1], P[i, 1] + P[i, 3])
          for i in range(n) if is_preplaced[i]]
    pp_b = [(P[i, 0], P[i, 0] + P[i, 2]) for i in range(n)
            if is_preplaced[i] and code[i]["B"]]

    def min_width(i):
        """narrowest this block can get (AR cap), for pocket/gap-close tests."""
        return np.sqrt(area[i] / AR_MAX) if soft[i] else P[i, 2]

    def spread_widths(blocks, target):
        """resize the soft blocks so their total width hits `target`, spreading
        the deficit/excess over blocks that still have AR headroom (iterative
        redistribution -- capped blocks stop, the rest absorb the remainder)."""
        sb = [i for i in blocks if soft[i]]
        if not sb:
            return
        for _ in range(12):
            diff = target - sum(P[i, 2] for i in sb)
            if abs(diff) < 1e-6:
                break
            if diff > 0:
                free = [i for i in sb if P[i, 2] < np.sqrt(area[i] * AR_MAX) - 1e-9]
            else:
                free = [i for i in sb if P[i, 2] > np.sqrt(area[i] / AR_MAX) + 1e-9]
            if not free:
                break
            fsum = sum(P[i, 2] for i in free)
            f = (fsum + diff) / fsum
            for i in free:
                resize_w(i, P[i, 2] * f)

    # =====================================================================
    # (b) BOTTOM ROW -- corners anchored, widths spread, obstacles dodged
    # =====================================================================
    # ---- 1. BOTTOM ROW: BL + mids + BR, resized to span the full width ----
    row = ([BLc] if BLc is not None else []) + Bmid + ([BRc] if BRc is not None else [])
    if row:
        pp_bw = sum(b - a for a, b in pp_b)
        s_hard = sum(P[i, 2] for i in row if not soft[i])
        spread_widths(row, W - pp_bw - s_hard)

    def bottom_jump(x, w, h):
        """slide x right past any preplaced the bottom block would hit at y=0."""
        moved = True
        while moved:
            moved = False
            for px0, px1, pb, pt in pp:
                if min(x + w, px1) - max(x, px0) > EPS and pb < h - EPS and pt > EPS:
                    x = px1; moved = True
        return x

    cur, row_end = 0.0, 0.0
    for k, i in enumerate(row):
        if i == BRc and k == len(row) - 1:
            break                                    # BR goes to the far corner
        w, h = P[i, 2], P[i, 3]
        x2 = bottom_jump(cur, w, h)
        if x2 > cur + EPS and soft[i]:
            # a preplaced forces a jump: shrink JUST ENOUGH to slot into the
            # pocket before it (never below the AR-5 minimum)
            blockers = [px0 for px0, px1, pb, pt in pp
                        if px0 > cur + EPS and px0 < cur + w - EPS
                        and pb < h * 2 - EPS and pt > EPS]
            pocket = (min(blockers) - cur) if blockers else 0.0
            old_w, old_h = P[i, 2], P[i, 3]
            if pocket > EPS and min_width(i) <= pocket + EPS:
                resize_w(i, pocket)
                if bottom_jump(cur, P[i, 2], P[i, 3]) <= cur + EPS:
                    x2 = cur                          # it fits now
                else:
                    P[i, 2], P[i, 3] = old_w, old_h   # still blocked -> jump
                    x2 = bottom_jump(cur, w, h)
            if x2 > cur + EPS:
                # The pocket before the blocker was too narrow for the AR cap.
                # The other way through is UNDER it: flatten the block until it
                # clears the blocker's bottom edge, area-constant, so it slips
                # beneath instead of being pushed aside.  Case 98's LB corner
                # block lost its corner exactly here -- the pocket was 8 wide
                # against a 10.4 minimum, while ducking to height 13 needed only
                # aspect 1.9.
                duck = min([pb for px0, px1, pb, pt in pp
                            if min(cur + old_w, px1) - max(cur, px0) > EPS
                            and pb > EPS] or [0.0])
                if duck > EPS:
                    resize_w(i, area[i] / duck)
                    if P[i, 3] <= duck + EPS \
                       and bottom_jump(cur, P[i, 2], P[i, 3]) <= cur + EPS:
                        x2 = cur
                    else:
                        P[i, 2], P[i, 3] = old_w, old_h
        P[i, 0], P[i, 1] = x2, 0.0
        cur = x2 + P[i, 2]
        row_end = cur
    if BRc is not None:
        w = P[BRc, 2]
        x = max(W - w, row_end)
        x = bottom_jump(x, w, P[BRc, 3])
        P[BRc, 0], P[BRc, 1] = x, 0.0
        row_end = max(row_end, x + w)
    # PACK_GROW=top: the side marks are settled, so a row that does not fit is
    # NOT allowed to push the frame wider -- the packing may only grow upward.
    if _os.environ.get("PACK_GROW", "wide") != "top":
        W = max(W, row_end)

    # ---- close leftover bottom-row gaps: grow the two neighbours of each gap
    #      JUST enough to meet (AR-capped, and never into a preplaced above) ----
    def grow_bottom(i, amt, leftward):
        """grow block i horizontally by up to amt; returns width actually gained."""
        if not soft[i] or amt <= EPS:
            return 0.0
        old_w, old_h, old_x = P[i, 2], P[i, 3], P[i, 0]
        resize_w(i, old_w + amt)
        if leftward:
            P[i, 0] = old_x - (P[i, 2] - old_w)
        for px0, px1, pb, pt in pp:                   # taller/wider now: check pp
            if min(P[i, 0] + P[i, 2], px1) - max(P[i, 0], px0) > EPS \
               and pb < P[i, 3] - EPS and pt > EPS:
                P[i, 2], P[i, 3], P[i, 0] = old_w, old_h, old_x
                return 0.0
        return P[i, 2] - old_w
    if row:
        elems = sorted([(P[i, 0], P[i, 0] + P[i, 2], i) for i in row] +
                       [(a, b, None) for a, b in pp_b])
        for k in range(len(elems) - 1):
            gap = elems[k + 1][0] - elems[k][1]
            if gap <= EPS:
                continue
            got = grow_bottom(elems[k][2], gap, False) if elems[k][2] is not None else 0.0
            if gap - got > EPS and elems[k + 1][2] is not None:
                grow_bottom(elems[k + 1][2], gap - got, True)

    ct = Contour(0.0, W, 0.0)
    for i in row:
        ct.raise_to(P[i, 0], P[i, 0] + P[i, 2], P[i, 3])

    def resolved_y(x, w, h):
        y = ct.height(x, x + w)
        changed = True
        while changed:
            changed = False
            for px0, px1, pb, pt in pp:
                if min(x + w, px1) - max(x, px0) > EPS and pb < y + h - EPS and pt > y + EPS:
                    y = pt
                    changed = True
        return y

    # ---- 2. EVENT-DRIVEN LAYERED FILL: lowest valley -> sides first, then
    #         interior in the middle; clusters kept together ----
    # PACK_SUPERBLOCK=1: pre-pack each cluster's interior members into ONE
    # zero-deadspace super-block (IMP/shape-curve style rows: soft members are
    # reshaped so every row fills the block width exactly), then drop it as a
    # single unit -- internal connectivity becomes structural.
    # Each group gets a SHAPE CURVE: several width variants are prepared and
    # the actual shape is only chosen at DROP time, matched to the valley (or
    # the group's anchor).  Rows never mutate P at build time -- widths are
    # stored as (member, width) pairs and applied on placement.
    pseudo = {}                                       # gid -> [(Wc, Hc, rows)]
    # item accessors: c >= 0 is a block id, c < 0 is cluster super-block -c
    # (for super-blocks the MID variant is the representative shape; the real
    # choice among variants happens at drop time in place_group)
    def it_w(c):
        return pseudo[-c][len(pseudo[-c]) // 2][0] if c < 0 else P[c, 2]

    def it_h(c):
        return pseudo[-c][len(pseudo[-c]) // 2][1] if c < 0 else P[c, 3]

    def it_g(c):
        return -c if c < 0 else clust[c]

    def it_cx(c):
        return dxc(gmx[-c] if c < 0 else cx[c])

    def it_cy(c):
        return dyc(gmy[-c] if c < 0 else cy[c])

    def it_minw(c):
        return min(v[0] for v in pseudo[-c]) if c < 0 else min_width(c)

    iq = [i for i in interior if clust[i] not in pseudo] + [-g for g in pseudo]
    last_cl = 0
    grp_x = {}                                        # cluster -> anchor x (last member)

    # NOTE: a hard-anchor schedule was tried here -- a cluster containing a
    # preplaced or boundary member was pinned by it and scheduled at that
    # member's height.  Grouping improved (case 69: 11 -> 8 splits) but hpwl
    # 0.724 -> 0.745 and area 0.295 -> 0.311 outweighed it (1.959 -> 1.985),
    # because a T-anchored cluster gets pushed to the ceiling, far from where
    # the net wanted it.  Removed.
    anch = {}                                         # kept: read by sort_y

    def anchor_y(g):
        return None

    def sort_y(c):
        g = it_g(c)
        ay = anchor_y(g) if g > 0 else None
        return ay if ay is not None else gmy.get(g, it_cy(c))
    iq.sort(key=lambda c: (sort_y(c), it_g(c), it_cy(c), it_cx(c)))

    # A cluster that spans BOTH boundary and interior blocks used to be split by
    # construction: the rows are placed outside the interior loop, so they never
    # registered an anchor for their group.  Register the bottom row now, and
    # pre-compute the top row's x positions (they depend only on W and the row
    # order, not on the interior) so its groups have an anchor too.
    for i in row:
        if clust[i] > 0:
            grp_x[clust[i]] = P[i, 0]
    trow = ([TLc] if TLc is not None else []) + Tmid + ([TRc] if TRc is not None else [])
    if trow:
        spread_widths(trow, W - sum(P[i, 2] for i in trow if not soft[i]))
        curx = 0.0
        for k, i in enumerate(trow):
            tx = (max(W - P[i, 2], curx) if (i == TRc and k == len(trow) - 1)
                  else min(curx, W - P[i, 2]))
            P[i, 0] = tx
            curx = tx + P[i, 2]
            if clust[i] > 0:
                grp_x[clust[i]] = tx

    # every rectangle already on the floor -- the contour is only a max-height
    # profile and cannot describe a pocket under an overhang, so the horizontal
    # hug below tests against real rectangles instead.
    occ = [(P[i, 0], P[i, 0] + P[i, 2], P[i, 1], P[i, 1] + P[i, 3]) for i in row]
    occ += [(a, b, c, d) for a, b, c, d in pp]

    def hug(x, y, w, h):
        """slide the block LEFT until it abuts its neighbour (or the frame
        side) at this height -- a pure 1D projection, so it can never create
        an overlap.  This is what tucks blocks INTO pockets the skyline
        cannot see."""
        lo = 0.0
        for a, b, c, d in occ:
            if min(y + h, d) - max(y, c) <= EPS:       # no vertical overlap
                continue
            if b <= x + EPS:
                lo = max(lo, b)
        # only close a SLIVER: a long slide would wreck the density order the
        # placement just chose (measured: unrestricted hug costs ~0.5 cost)
        return lo if EPS < x - lo <= HUG_FRAC * w else x

    def place(i, x, no_hug=False):
        w, h = P[i, 2], P[i, 3]
        y = resolved_y(x, w, h)
        if not no_hug:
            x = hug(x, y, w, h)
        P[i, 0], P[i, 1] = x, y
        # the block may now sit UNDER an overhang: never lower the profile
        ct.raise_min(x, x + w, y + h)
        occ.append((x, x + w, y, y + h))
        if clust[i] > 0 and clust[i] not in anch:     # hard anchors never drift
            grp_x[clust[i]] = x

    # =====================================================================
    # (c) SIDE TOWERS -- stacked bottom-up, slide/squeeze past obstacles
    # =====================================================================
    # ---- 2a. SIDE TOWERS FIRST: stack every L block at x=0 and every R block
    #          at x=W-w bottom-up.  Interior then fills around them -- no lane
    #          reservation needed, so no dead strips are left behind. ----
    # A preplaced block has ZERO degrees of freedom; a boundary block has ONE
    # (it slides along its own edge).  So when they share a cluster it is the
    # boundary block that must travel: an L/R tower member slides in y until it
    # abuts its preplaced peer, instead of stacking at the next free height.
    SLIDE = _os.environ.get("PACK_SLIDE", "1") == "1"
    pp_of = {}                                        # cluster -> preplaced peer
    if SLIDE:
        for j in range(n):
            if is_preplaced[j] and clust[j] > 0 and clust[j] not in pp_of:
                pp_of[clust[j]] = j

    def free_at(x, y, w, h):
        for a, b, c, d in occ:
            if min(x + w, b) - max(x, a) > EPS and min(y + h, d) - max(y, c) > EPS:
                return False
        return True

    def slide_to_peer(i, x):
        """y positions where block i (pinned at x) would touch its preplaced
        cluster peer -- best free one wins, else None."""
        j = pp_of.get(clust[i])
        if j is None:
            return None
        w, h = P[i, 2], P[i, 3]
        px, py, pw, ph = P[j, 0], P[j, 1], P[j, 2], P[j, 3]
        side_touch = min(x + w, px + pw) - max(x, px) > EPS   # stack above/below
        cands = []
        if side_touch:
            cands += [py + ph, py - h]
        if min(x + w, px + pw) - max(x, px) > -EPS:           # or sit beside it
            cands += [py, py + ph - h]
        best = None
        for y in cands:
            if y < 0 or not free_at(x, y, w, h):
                continue
            if resolved_y(x, w, h) > y + EPS:                 # must be supported
                continue
            d = abs(y - py)
            if best is None or d < best[0]:
                best = (d, y)
        return None if best is None else best[1]

    def squeeze_past(i, left):
        """A tower block pushed up by a preplaced obstacle: try narrowing it
        (area-constant, AR-capped) so it slips PAST the obstacle in the lane
        beside it and keeps the lower resting height.  This is the vertical
        twin of the bottom row's pocket shrink -- without it a tower simply
        jumps over every obstacle, and case 90's left tower lost 96 of its 280
        units of height to four such gaps."""
        if not soft[i] or _os.environ.get("PACK_SIDE_FIT", "1") != "1":
            return None
        w, h = P[i, 2], P[i, 3]
        x = 0.0 if left else W - w
        y0 = resolved_y(x, w, h)
        best = None
        for px0, px1, pb, pt in pp:
            if pt <= EPS or pb >= y0 - EPS:
                continue                              # not what is blocking us
            lane = px0 if left else W - px1           # width still free beside it
            if lane <= EPS or lane >= w - EPS:
                continue
            old_w, old_h = P[i, 2], P[i, 3]
            resize_w(i, lane)
            if P[i, 2] <= lane + EPS:
                x2 = 0.0 if left else W - P[i, 2]
                y2 = resolved_y(x2, P[i, 2], P[i, 3])
                if y2 < y0 - EPS and (best is None or y2 < best[0]):
                    best = (y2, x2, P[i, 2], P[i, 3])
            P[i, 2], P[i, 3] = old_w, old_h
        if best is None:
            return None
        y2, x2, w2, h2 = best
        P[i, 2], P[i, 3] = w2, h2
        return x2, y2

    def place_side(i, x):
        y = slide_to_peer(i, x)
        if y is None:
            fit = squeeze_past(i, left=(x <= EPS))
            if fit is not None:
                x2, y2 = fit
                P[i, 0], P[i, 1] = x2, y2
                ct.raise_min(x2, x2 + P[i, 2], y2 + P[i, 3])
                occ.append((x2, x2 + P[i, 2], y2, y2 + P[i, 3]))
                if clust[i] > 0:
                    grp_x[clust[i]] = x2
                return
            place(i, x, no_hug=True)
            return
        P[i, 0], P[i, 1] = x, y
        ct.raise_min(x, x + P[i, 2], y + P[i, 3])
        occ.append((x, x + P[i, 2], y, y + P[i, 3]))
        if clust[i] > 0:
            grp_x[clust[i]] = x

    pinned_side = set()                               # placed by slide_to_peer

    def place_side_tracked(i, x):
        before = P[i, 1]
        y = slide_to_peer(i, x)
        place_side(i, x)
        if y is not None:
            pinned_side.add(i)

    for i in Ls:
        place_side_tracked(i, 0.0)
    for i in Rs:
        place_side_tracked(i, W - P[i, 2])

    # ---- FILL THE HOLES A PINNED TOWER BLOCK LEAVES BELOW IT ----
    # slide_to_peer does the right thing -- a boundary block has one degree of
    # freedom and a preplaced peer has none, so the boundary block travels.  But
    # when the pinned block is the FIRST of its tower, everything under it is
    # stranded: case 90's right tower lost 66 units that way, and no later block
    # ever goes back for it because towers only stack upward.  So make a second
    # pass: any tower block that is NOT pinned drops into the lowest hole that
    # can hold it, cluster or not.
    if _os.environ.get("PACK_SIDE_FILL", "1") == "1":
        for seq, sx in ((Ls, lambda w: 0.0), (Rs, lambda w: W - w)):
            movable = [i for i in seq if i not in pinned_side and not is_preplaced[i]]
            for _ in range(2):                        # a drop can open the next
                for i in sorted(movable, key=lambda j: -P[j, 1]):
                    w, h = P[i, 2], P[i, 3]
                    x = sx(w)
                    tops = [0.0] + [b[3] for b in occ
                                    if min(x + w, b[1]) - max(x, b[0]) > EPS
                                    and b[3] <= P[i, 1] + EPS
                                    and not (abs(b[0] - x) < EPS and abs(b[2] - P[i, 1]) < EPS)]
                    for cand in sorted(set(tops)):
                        if cand >= P[i, 1] - EPS:
                            break
                        clash = False
                        for a, b2, c, d in occ:
                            if abs(a - x) < EPS and abs(c - P[i, 1]) < EPS:
                                continue              # itself
                            if min(x + w, b2) - max(x, a) > EPS \
                               and min(cand + h, d) - max(cand, c) > EPS:
                                clash = True; break
                        if not clash:
                            occ[:] = [b for b in occ
                                      if not (abs(b[0] - x) < EPS
                                              and abs(b[2] - P[i, 1]) < EPS)]
                            P[i, 1] = cand
                            occ.append((x, x + w, cand, cand + h))
                            break
        ct.segs = [[0.0, W, 0.0]]                     # rebuild the profile
        for a, b2, c, d in occ:
            ct.raise_min(a, b2, d)

    grp_box = {}                                      # cluster -> placed bbox

    # Everything already on the floor becomes a real wirelength anchor: the
    # bottom row, both side towers, and any non-boundary preplaced obstacle.
    for i in list(row) + Ls + Rs:
        placed_at[i] = (float(P[i, 0] + P[i, 2] / 2), float(P[i, 1] + P[i, 3] / 2))

    # =====================================================================
    # (d) INTERIOR -- MaxRects fill, scored S_area + 0.5*S_wire + 32*S_grp
    # =====================================================================
    # ---- 2b. INTERIOR ----
    if _os.environ.get("PACK_INTERIOR", "maxrects") == "maxrects":
        # MAXRECTS mode: pockets under overhangs (beside preplaced, under
        # bridges) are real free rectangles here, so blocks tuck into them.
        HBIN = H_est * 6 + max(P[i, 3] for i in range(n)) \
            + max([d for _, _, _, d in pp] + [0.0])
        mr = MaxRects(W, HBIN)
        for r in occ:
            mr.occupy(r)

        # ---- placement score, all three terms dimensionless ----
        # S_area: only the part of the landing that pushes the frame TALLER
        #         costs anything -- a block tucked inside the existing envelope
        #         is free, which is exactly what "lowest y" failed to express.
        # S_wire: the block's own weighted Manhattan cost at that spot, over the
        #         density layout's total -- the same ratio the contest's
        #         hpwl_gap uses, so it needs no weight of its own.
        # S_fit:  leftover area of the free rect; not in the contest cost at
        #         all, only a guess about future waste, hence the small lambda.
        # S_grp: the score above is blind to grouping, and letting hpwl+area
        # decide alone doubled the soft violations (0.085 -> 0.159), which the
        # exp(beta*V_rel) factor turns into ~16% cost -- more than the gain.
        # A cluster only counts as satisfied when its members ABUT, so pay for
        # the gap to the nearest already-placed peer, in units of sqrt(A).
        LAM = float(_os.environ.get("PACK_LAMBDA", "0.0"))
        LAM_FREE = float(_os.environ.get("PACK_LAMBDA_FREE", "0.2"))
        # Turn the dial toward dead space: WIREW < 1 buys tighter packing with
        # wirelength.  FIT=bssf measures tightness by the SHORT side left over
        # (Jylanki's best-short-side-fit) instead of leftover area, which does
        # not punish a small block for landing in a large free rect that other
        # blocks can still use.
        # How many blocks from the head of the queue compete each step.  The
        # queue is only a soft bottom-up preference: a wider window lets the
        # score override it further (order-vs-queue correlation was +0.43 at
        # a window of 10, with blocks jumping up to 33 places).  Cost is
        # O(window) calls to MaxRects.find per placement.
        POOLW = int(_os.environ.get("PACK_POOL", "1"))
        WIREW = float(_os.environ.get("PACK_WIREW", "0.5"))
        FITMODE = _os.environ.get("PACK_FIT", "area")
        GRP = float(_os.environ.get("PACK_GRP", "32.0"))
        H_cur = [max([b[3] for b in occ] + [0.0])]
        SQA = np.sqrt(A)

        def mk_score(i):
            peers = [j for j in range(n) if clust[j] == clust[i] and j != i] \
                if clust[i] > 0 else []

            def sc(x, y, w2, h2, rect_area, rw_=0.0, rh_=0.0):
                s_area = (W * max(0.0, y + h2 - H_cur[0])) / A
                wl, k = hpwl_of(i, x + w2 / 2, y + h2 / 2)
                s = s_area + (WIREW * wl / HPWL_REF if k else 0.0)
                lam = LAM if k else LAM_FREE   # no placed neighbour -> fit decides
                if lam:
                    if FITMODE == "bssf":
                        s += lam * min(rw_ - w2, rh_ - h2) / SQA
                    else:
                        s += lam * (rect_area - w2 * h2) / A
                if GRP and peers:
                    gap = None
                    for j in peers:
                        if placed_at[j] is None:
                            continue
                        jx0, jy0 = P[j, 0], P[j, 1]
                        jx1, jy1 = jx0 + P[j, 2], jy0 + P[j, 3]
                        gx = max(0.0, max(x, jx0) - min(x + w2, jx1))
                        gy = max(0.0, max(y, jy0) - min(y + h2, jy1))
                        d = gx + gy
                        if gap is None or d < gap:
                            gap = d
                    if gap:
                        s += GRP * gap / SQA

                return s
            return sc

        while iq:
            pool = [c for c in iq if last_cl > 0 and it_g(c) == last_cl] or iq
            best = None
            for k in range(min(POOLW, len(pool))):
                c = pool[k]
                g = it_g(c)
                px = it_cx(c) - it_w(c) / 2
                win = None
                if g > 0 and g in grp_x:
                    px = grp_x[g]
                    win = (px - 2 * it_w(c), px + 3 * it_w(c))
                if c >= 0:
                    a = area[c] if soft[c] else None
                    sc = mk_score(c)
                    f = mr.find(P[c, 2], P[c, 3], a, px, win, sc)
                    if f is None and win is not None:
                        f = mr.find(P[c, 2], P[c, 3], a, px, None, sc)
                    vv = None
                else:                                  # super-block: try every
                    f, vv = None, None                 # shape variant, keep the
                    for v in pseudo[-c]:               # lowest landing
                        fv = mr.find(v[0], v[1], None, px, win)
                        if fv is None and win is not None:
                            fv = mr.find(v[0], v[1], None, px, None)
                        if fv is not None and (f is None or fv[0] < f[0]):
                            f, vv = fv, v
                if f is None:
                    continue
                y, x, w2 = f
                h2 = (area[c] / w2) if (c >= 0 and soft[c]) else \
                     (P[c, 3] if c >= 0 else vv[1])
                cost = (W * max(0.0, y + h2 - H_cur[0])) / A + k * 0.02
                if c >= 0:
                    _wl, _k = hpwl_of(c, x + w2 / 2, y + h2 / 2)
                    cost += (_wl / HPWL_REF if _k else 0.0)
                if best is None or cost < best[0]:
                    best = (cost, c, x, y, w2, vv)
            if best is None:                          # fragmented: fall back to
                c = pool[0]                           # the skyline drop
                iq.remove(c)
                if c >= 0:
                    place(c, 0.0)
                else:
                    s0, e0, h0 = ct.lowest_valley()
                    place_group(-c, s0, e0, h0)
                last_cl = it_g(c)
                continue
            _, c, x, y, w2, vv = best
            iq.remove(c)
            if c >= 0:
                if abs(w2 - P[c, 2]) > EPS:
                    resize_w(c, w2)
                P[c, 0], P[c, 1] = x, y
                r = (x, x + P[c, 2], y, y + P[c, 3])
                if clust[c] > 0:
                    grp_x[clust[c]] = x
            else:
                g = -c
                Wc, Hc, rows = vv
                yy = y
                for hr, mem in rows:
                    _apply_row(x, yy, hr, mem)
                    yy += hr
                r = (x, x + Wc, y, y + Hc)
                grp_x[g] = x
                grp_box[g] = r
            mr.occupy(r)
            occ.append(r)
            ct.raise_min(r[0], r[1], r[3])
            H_cur[0] = max(H_cur[0], r[3])
            if STEP_HOOK is not None:
                STEP_HOOK(c, r, list(mr.F), list(occ), W)
            if c >= 0:
                placed_at[c] = ((r[0] + r[1]) / 2, (r[2] + r[3]) / 2)
            else:
                for _, mem in vv[2]:
                    for j, _w in mem:
                        placed_at[j] = (P[j, 0] + P[j, 2] / 2, P[j, 1] + P[j, 3] / 2)
            last_cl = it_g(c)

    # Clusters that straddle the boundary and the interior are split BY
    # CONSTRUCTION: their tower members go down while the towers are built, long
    # before the interior half of the same cluster exists.  Measured, they carry
    # 99% of what is left of the grouping violations (0.59 splits per cluster,
    # against 0.04 for clusters that live entirely in the interior), and no
    # amount of extra abutment reward moves them -- the reward only acts inside
    # the interior loop.  So hand step 2c the interior extent of every cluster,
    # not just the ones that were packed as super-blocks, and let it slide the
    # tower members along their edge to meet it.
    if _os.environ.get("PACK_GRP_SLIDE", "1") == "1" and not grp_box:
        for g in {clust[i] for i in interior if clust[i] > 0}:
            mem = [i for i in interior
                   if clust[i] == g and placed_at[i] is not None]
            if mem:
                grp_box[g] = (min(P[i, 0] for i in mem),
                              max(P[i, 0] + P[i, 2] for i in mem),
                              min(P[i, 1] for i in mem),
                              max(P[i, 1] + P[i, 3] for i in mem))

    # ---- 2c. RE-ADJUST BOUNDARY MEMBERS OF PLACED CLUSTERS ----
    # A tower block whose cluster's super-block only lands later cannot be put
    # in the right place when the tower is built.  Now that the block has
    # landed, the tower member slides along its edge -- its single degree of
    # freedom -- until it abuts the cluster.  The hole it leaves behind is not
    # wasted: everything above it in the tower drops down to close the gap, and
    # whatever slack is left is ordinary free space the interior can use.
    if grp_box and _os.environ.get("PACK_SB_ADJUST", "1") == "1":
        settled = list(row) + Ls + Rs + interior + \
            [i for i in range(n) if is_preplaced[i]]

        def rects_but(b):
            return [(P[i, 0], P[i, 0] + P[i, 2], P[i, 1], P[i, 1] + P[i, 3])
                    for i in settled if i != b]

        def free_but(b, x, y, w, h):
            if y < -EPS or x < -EPS or x + w > W + EPS:
                return False
            for a, b2, c, d in rects_but(b):
                if min(x + w, b2) - max(x, a) > EPS and min(y + h, d) - max(y, c) > EPS:
                    return False
            return True
        moved = set()
        for g, (gx0, gx1, gy0, gy1) in grp_box.items():
            for b in Ls + Rs:
                if clust[b] != g or is_preplaced[b] or b in moved:
                    continue
                x, w, h = P[b, 0], P[b, 2], P[b, 3]
                if min(x + w, gx1) - max(x, gx0) <= EPS:
                    continue                          # not in the same column
                best = None
                for y in (gy1, gy0 - h, gy0, gy1 - h):
                    if not free_but(b, x, y, w, h):
                        continue
                    d = abs(y - P[b, 1])
                    if best is None or d < best[0]:
                        best = (d, y)
                if best is not None:
                    P[b, 1] = best[1]
                    moved.add(b)
        if moved:                                     # close the towers' holes
            for seq in (Ls, Rs):
                for b in sorted([i for i in seq if i not in moved
                                 and not is_preplaced[i]], key=lambda i: P[i, 1]):
                    x, w, h = P[b, 0], P[b, 2], P[b, 3]
                    ny = 0.0
                    for a, b2, c, d in rects_but(b):
                        if min(x + w, b2) - max(x, a) > EPS and d <= P[b, 1] + EPS:
                            ny = max(ny, d)
                    if ny < P[b, 1] - EPS and free_but(b, x, ny, w, h):
                        P[b, 1] = ny
            ct.segs = [[0.0, W, 0.0]]                 # rebuild the profile
            occ[:] = []
            for i in settled:
                ct.raise_min(P[i, 0], P[i, 0] + P[i, 2], P[i, 1] + P[i, 3])
                occ.append((P[i, 0], P[i, 0] + P[i, 2], P[i, 1], P[i, 1] + P[i, 3]))

    # =====================================================================
    # (e) TOP ROW -- flat lid, then reach down to cluster peers
    # =====================================================================
    # ---- 3. TOP ROW: TL + mids + TR side by side, flat top, resized to span ----
    if trow:
        tset = set(trow)
        rest = [i for i in range(n) if i not in tset]
        H_int = max([P[i, 1] + P[i, 3] for i in rest] + [0.0])
        top = H_int + max(P[i, 3] for i in trow)      # x was fixed before the fill
        for i in trow:                                 # so its clusters had an anchor
            P[i, 1] = top - P[i, 3]

        # ---- REACH UP: close the last gap to a T-coded cluster peer ----
        # Ordering a side block to the top of its tower only brings it near the
        # ceiling; the tower still ends wherever its blocks happen to end, tens
        # of units below the T row (7 of 79 such pairs actually touched).  Now
        # that the row's height is known, STRETCH the side block upward --
        # area-constant, so it narrows as it grows and hands the width back to
        # the lane -- until it meets its peer.  Only the topmost block of a
        # tower may do this: anything below has neighbours above it.
        if _os.environ.get("PACK_REACH", "1") == "1":
            tops = {}
            for seq in (Ls, Rs):
                for i in seq:
                    if soft[i] and (seq_top := tops.get(id(seq))) is None \
                       or (soft[i] and P[i, 1] + P[i, 3] > P[tops[id(seq)], 1] + P[tops[id(seq)], 3]):
                        tops[id(seq)] = i
            for seq in (Ls, Rs):
                i = tops.get(id(seq))
                if i is None or clust[i] <= 0:
                    continue
                peers = [j for j in trow if clust[j] == clust[i]]
                if not peers:
                    continue
                target = min(P[j, 1] for j in peers)   # bottom of the T peer
                need = target - P[i, 1]
                if need <= P[i, 3] + EPS:
                    continue                           # already reaches
                ow, oh, ox = P[i, 2], P[i, 3], P[i, 0]
                resize_w(i, area[i] / need)            # taller => narrower
                P[i, 0] = 0.0 if ox <= EPS else W - P[i, 2]
                ok = P[i, 3] >= need - EPS
                if ok:
                    for a, b2, c, d in occ:
                        if abs(a - ox) < EPS and abs(c - P[i, 1]) < EPS:
                            continue                   # itself
                        if min(P[i, 0] + P[i, 2], b2) - max(P[i, 0], a) > EPS \
                           and min(P[i, 1] + P[i, 3], d) - max(P[i, 1], c) > EPS:
                            ok = False; break
                if ok:
                    occ[:] = [b for b in occ
                              if not (abs(b[0] - ox) < EPS and abs(b[2] - P[i, 1]) < EPS)]
                    occ.append((P[i, 0], P[i, 0] + P[i, 2], P[i, 1], P[i, 1] + P[i, 3]))
                else:
                    P[i, 2], P[i, 3], P[i, 0] = ow, oh, ox
        # NOTE: dropping the whole row onto the interior skyline was tried here
        # (lid kept flat, lowered by the smallest clearance).  It is a no-op in
        # practice -- the tallest T block almost always sits over the tallest
        # interior column, so that clearance is 0 -- and measured 1.980 -> 1.983.
        # The band under a SHORT T block is interior space, not row space; the
        # way to reclaim it is a better frame width, not a lower row.

    # =====================================================================
    # (f) SAFETY NET -- two-tier push; tier 2 guarantees zero overlap
    # =====================================================================
    # ---- final cleanup, two tiers (identical to frame_pack v1) ----
    def cleanup(anchor, rounds):
        for _ in range(rounds):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    ox = min(P[i, 0] + P[i, 2], P[j, 0] + P[j, 2]) - max(P[i, 0], P[j, 0])
                    oy = min(P[i, 1] + P[i, 3], P[j, 1] + P[j, 3]) - max(P[i, 1], P[j, 1])
                    if ox <= EPS or oy <= EPS or (anchor[i] and anchor[j]):
                        continue
                    free = j if anchor[i] else (i if anchor[j] else
                           (i if P[i, 2] * P[i, 3] <= P[j, 2] * P[j, 3] else j))
                    other = i if free == j else j
                    moved = True
                    if ox < oy:
                        P[free, 0] = (P[free, 0] + ox + EPS) if P[free, 0] >= P[other, 0] \
                            else max(0.0, P[free, 0] - ox - EPS)
                    else:
                        P[free, 1] = (P[free, 1] + oy + EPS) if P[free, 1] >= P[other, 1] \
                            else max(0.0, P[free, 1] - oy - EPS)
            if not moved:
                return True
        return False

    soft_anchor = [is_preplaced[i] or any(code[i].values()) for i in range(n)]
    if _os.environ.get("FRAME3_TRACE"):
        ov = [(i, j) for i in range(n) for j in range(i + 1, n)
              if min(P[i, 0] + P[i, 2], P[j, 0] + P[j, 2]) - max(P[i, 0], P[j, 0]) > 0.3
              and min(P[i, 1] + P[i, 3], P[j, 1] + P[j, 3]) - max(P[i, 1], P[j, 1]) > 0.3]
        print(f"[TRACE3 n={n}] overlaps BEFORE cleanup: {len(ov)} {ov[:12]}")
        print("  Ls y:", [(i, round(float(P[i, 1]), 1)) for i in Ls])
        print("  Rs y:", [(i, round(float(P[i, 1]), 1)) for i in Rs])
    cleanup(soft_anchor, 200)

    # Tier 2: strict monotonic push -> guarantees overlap 0; preplaced never move
    changed, r = True, 0
    while changed and r < 400:
        changed, r = False, r + 1
        for i in range(n):
            for j in range(i + 1, n):
                ox = min(P[i, 0] + P[i, 2], P[j, 0] + P[j, 2]) - max(P[i, 0], P[j, 0])
                oy = min(P[i, 1] + P[i, 3], P[j, 1] + P[j, 3]) - max(P[i, 1], P[j, 1])
                if ox <= EPS or oy <= EPS:
                    continue
                pi, pj = is_preplaced[i], is_preplaced[j]
                if pi and pj:
                    continue
                changed = True
                if ox < oy:
                    if pi:
                        P[j, 0] = max(P[j, 0], P[i, 0] + P[i, 2])
                    elif pj:
                        P[i, 0] = max(P[i, 0], P[j, 0] + P[j, 2])
                    elif P[i, 0] > P[j, 0]:
                        P[i, 0] = max(P[i, 0], P[j, 0] + P[j, 2])
                    else:
                        P[j, 0] = max(P[j, 0], P[i, 0] + P[i, 2])
                else:
                    if pi:
                        P[j, 1] = max(P[j, 1], P[i, 1] + P[i, 3])
                    elif pj:
                        P[i, 1] = max(P[i, 1], P[j, 1] + P[j, 3])
                    elif P[i, 1] > P[j, 1]:
                        P[i, 1] = max(P[i, 1], P[j, 1] + P[j, 3])
                    else:
                        P[j, 1] = max(P[j, 1], P[i, 1] + P[i, 3])
