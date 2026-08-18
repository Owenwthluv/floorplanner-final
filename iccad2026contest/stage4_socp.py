#!/usr/bin/env python3
"""
STAGE 4 -- SOCP POLISH  (UFO / SOPL, Lin & Hung, TCAD 2011, Section IV-B)

The legalizer hands over a floorplan that is already legal: zero overlap, every
hard constraint met, boundary and cluster codes mostly satisfied.  What it is
NOT is tight -- it packs at ~83% utilisation where the ground truth reaches 96%,
because every block was placed once, greedily, and never revisited.

This stage keeps the TOPOLOGY the legalizer found and re-solves the geometry
exactly.  Two blocks that the packing put side by side stay side by side; what
changes is where the edge between them falls and how each block is shaped.  With
the topology frozen the problem is convex, so the answer is the global optimum
for that topology rather than another heuristic pass:

    min   H_now * W + W_now * H                    (linearised bbox area)
    s.t.  x_i + w_i <= x_j          (i,j) in C_h   non-overlap, horizontal
          y_i + h_i <= y_j          (i,j) in C_v   non-overlap, vertical
          0 <= x_i,  x_i + w_i <= W                inside the frame
          0 <= y_i,  y_i + h_i <= H
          w_i * h_i >= A_i                         area preserved
          w_i <= AR*h_i,  h_i <= AR*w_i            aspect ratio

The area constraint is the only non-linear one, and it is exactly a second-order
cone -- this is the identity the paper turns on:

    w*h >= A   <=>   h + w >= || (h - w,  2*sqrt(A)) ||_2

so the whole model is an SOCP and MOSEK solves it directly.

WHERE THIS DIFFERS FROM THE PAPER
The paper has to GUESS the topology: its first stage leaves overlapping circles,
so it recovers relations with Delaunay triangulation and a rule about which of
two circles is more efficiently separated horizontally.  We do not guess.  The
input here is a legal packing, so for every pair the relation that already holds
is read straight off the layout, with the largest margin winning.  The constraint
graphs are exact, and no pair can be missed.

SAFETY
MOSEK is a licensed solver and the license lives on the developer's machine, so
this stage is optional by construction: a missing module, a missing license, a
solver failure, residual overlap, or a result that does not actually score
better all return None, and the caller keeps the layout it already had.
"""
import os as _os
from pathlib import Path

import numpy as np

EPS = 1e-6
REJECT = {}          # why a solve was thrown away, for diagnosis


def _ensure_license():
    """MOSEK auto-discovers ~/mosek/mosek.lic; set the env var too if present."""
    if "MOSEKLM_LICENSE_FILE" in _os.environ:
        return
    cand = Path.home() / "mosek" / "mosek.lic"
    if cand.exists():
        _os.environ["MOSEKLM_LICENSE_FILE"] = str(cand)


def build_constraint_graphs(P, n, reduce_transitive=True):
    """Read C_h and C_v off a legal packing.

    For every pair, whichever separation actually holds becomes the constraint;
    when both hold the one with the larger margin is kept, since that is the
    relation the packing was really relying on.  Overlapping pairs (there should
    be none) fall back to the paper's rule -- separate along whichever axis
    needs less movement.
    """
    x0 = P[:n, 0]; y0 = P[:n, 1]
    x1 = x0 + P[:n, 2]; y1 = y0 + P[:n, 3]
    Ch, Cv = [], []
    for i in range(n):
        for j in range(i + 1, n):
            # gaps along each axis, positive when i is left of / below j
            gh_ij = x0[j] - x1[i]; gh_ji = x0[i] - x1[j]
            gv_ij = y0[j] - y1[i]; gv_ji = y0[i] - y1[j]
            best = max(gh_ij, gh_ji, gv_ij, gv_ji)
            if best < -EPS:                       # overlapping: paper's rule
                dw = min(x1[i], x1[j]) - max(x0[i], x0[j])
                dh = min(y1[i], y1[j]) - max(y0[i], y0[j])
                if dw < dh:
                    (Ch if x0[i] <= x0[j] else Ch).append(
                        (i, j) if x0[i] <= x0[j] else (j, i))
                else:
                    Cv.append((i, j) if y0[i] <= y0[j] else (j, i))
                continue
            if best == gh_ij:
                Ch.append((i, j))
            elif best == gh_ji:
                Ch.append((j, i))
            elif best == gv_ij:
                Cv.append((i, j))
            else:
                Cv.append((j, i))
    if reduce_transitive:
        Ch = _transitive_reduce(Ch, n)
        Cv = _transitive_reduce(Cv, n)
    return Ch, Cv


def _transitive_reduce(E, n):
    """Drop edges implied by a two-hop path.

    An O(n^2) packing produces an edge for every pair, and most of them say
    nothing new: if a is left of b and b is left of c, the a->c edge is already
    enforced.  Removing them shrinks the model by roughly an order of magnitude
    without changing the feasible set at all.
    """
    adj = [set() for _ in range(n)]
    for a, b in E:
        adj[a].add(b)
    keep = []
    for a, b in E:
        if not any(b in adj[m] for m in adj[a] if m != b):
            keep.append((a, b))
    return keep


def stage4_socp(n, positions, is_preplaced, constraints, ar_max=3.0,
                verbose=False, b2b=None, p2b=None, pins=None):
    """Re-solve the geometry of a legal packing.  Returns a new (n,4) array, or
    None when the polish is unavailable or did not help.

    TOLERANCE LADDER.  A tight primal-feasibility tolerance is what keeps the
    solver's residue small, but it is also what makes it give up: at 1e-12,
    26 of 100 cases came back SolutionStatus.Unknown and were thrown away --
    a quarter of the suite skipping the polish entirely, for no reason but an
    over-strict ask.  Nothing downstream needs that tightness any more, because
    stage 5 closes contacts by ASSIGNING coordinates, which repairs a residue
    of any size.  So ask for tight, and settle for what the solver can actually
    deliver.
    """
    # CLARABEL BY DEFAULT.  Both solvers reach the same answer -- 1.1922 against
    # 1.1926 Total Score over the suite, neither meaningfully ahead -- but only
    # one of them runs on a machine that is not this one.  MOSEK needs a licence
    # file, and a submission that assumes it will not find one: stage 4 would
    # return None on every case, silently, and the score would fall back to
    # roughly 1.32 with nothing in the output to say why.  Set
    # SOCP_BACKEND=mosek to use MOSEK where a licence exists.
    back = _os.environ.get("SOCP_BACKEND", "clarabel").lower()
    fn = _solve_clarabel if back == "clarabel" else _solve
    for _t in [float(v) for v in
               _os.environ.get("SOCP_TOL_LADDER", "1e-12,1e-10,1e-8").split(",")]:
        r = fn(n, positions, is_preplaced, constraints, ar_max, verbose, _t,
               b2b=b2b, p2b=p2b, pins=pins)
        if r is not None:
            return r
    return None


def _solve(n, positions, is_preplaced, constraints, ar_max, verbose, tol,
           b2b=None, p2b=None, pins=None):
    try:
        import mosek
        from mosek.fusion import Model, Domain, Expr, Var, ObjectiveSense
    except ImportError:
        return None
    _ensure_license()

    P = np.asarray(positions, dtype=float)[:n].copy()
    cons = constraints
    area = P[:, 2] * P[:, 3]
    fixed = np.array([int(cons[i, 0]) > 0 for i in range(n)])
    prep = np.array([bool(is_preplaced[i]) for i in range(n)])
    mib = np.array([int(cons[i, 2]) for i in range(n)])
    bnd = np.array([int(cons[i, 4]) for i in range(n)])
    # a block may only be reshaped if nothing else pins its shape
    soft = ~prep & ~fixed & (mib == 0)

    x_lo = P[:, 0].min(); y_lo = P[:, 1].min()
    W_now = float((P[:, 0] + P[:, 2]).max() - x_lo)
    H_now = float((P[:, 1] + P[:, 3]).max() - y_lo)
    P[:, 0] -= x_lo; P[:, 1] -= y_lo          # work in frame coordinates

    Ch, Cv = build_constraint_graphs(P, n)
    if verbose:
        print(f"    C_h={len(Ch)}  C_v={len(Cv)} edges")

    try:
        with Model("sopl") as M:
            M.setSolverParam("optimizerMaxTime", 20.0)
            # MOSEK stops when its own tolerances are met, and the default
            # primal-feasibility tolerance leaves an equality like
            # "x_a + w_a = x_b" satisfied to ~9e-6 at coordinates of order 200.
            # The contest stops calling two blocks touching past 1e-6, so that
            # residue alone breaks a cluster contact.  Tightening costs a few
            # extra interior-point iterations and buys three orders of
            # magnitude, which puts the residue back under the threshold.
            # One thread per solve.  MOSEK grabs every core by default, so
            # three variants running "in parallel" just fight over the same
            # eight, and the wall clock does not move.  With each solve pinned
            # to one thread the parallelism moves up a level, where it is the
            # variants -- not the linear algebra -- that overlap.
            try:
                M.setSolverParam("numThreads",
                                 int(_os.environ.get("SOCP_THREADS", "1")))
            except Exception:
                pass
            _tol = tol
            for _pname in ("intpntCoTolPfeas", "intpntCoTolDfeas",
                           "intpntCoTolRelGap", "intpntCoTolMuRed"):
                try:
                    M.setSolverParam(_pname, _tol)
                except Exception:
                    pass
            x = M.variable("x", n, Domain.greaterThan(0.0))
            y = M.variable("y", n, Domain.greaterThan(0.0))
            w = M.variable("w", n, Domain.greaterThan(0.0))
            h = M.variable("h", n, Domain.greaterThan(0.0))
            W = M.variable("W", 1, Domain.greaterThan(0.0))
            H = M.variable("H", 1, Domain.greaterThan(0.0))

            # ================================================================
            # THE WHOLE MODEL IS BUILT IN VECTORS
            # ================================================================
            # Fusion's Python layer is where this stage spent its time: 925
            # separate M.constraint() calls on case 99, 0.32s of the 0.38s
            # total, against 0.055s actually inside the solver.  All of it is
            # Python bytecode, so it is also why running variants on threads
            # bought nothing -- the GIL held them in single file.
            #
            # Every family of constraints below therefore goes in as ONE call
            # over an index array: Var.pick selects the rows, Var.repeat
            # broadcasts a scalar against a vector, and a matrix handed to
            # Domain.inQCone() is read one cone per ROW.  Same model, same
            # answer, a couple of dozen calls instead of a thousand.
            def pick(v, idx):
                return v.pick([int(k) for k in idx])

            idx = np.arange(n)
            i_pre = idx[prep]
            i_hard = idx[~prep & ~soft]            # movable, shape frozen
            i_soft = idx[soft]

            # ---- shapes -------------------------------------------------
            if len(i_pre):
                M.constraint(pick(x, i_pre), Domain.equalsTo(P[i_pre, 0]))
                M.constraint(pick(y, i_pre), Domain.equalsTo(P[i_pre, 1]))
                M.constraint(pick(w, i_pre), Domain.equalsTo(P[i_pre, 2]))
                M.constraint(pick(h, i_pre), Domain.equalsTo(P[i_pre, 3]))
            if len(i_hard):
                M.constraint(pick(w, i_hard), Domain.equalsTo(P[i_hard, 2]))
                M.constraint(pick(h, i_hard), Domain.equalsTo(P[i_hard, 3]))
            if len(i_soft):
                hs, ws = pick(h, i_soft), pick(w, i_soft)
                # w*h >= A, one second-order cone per row
                M.constraint(Expr.hstack(Expr.add(hs, ws), Expr.sub(hs, ws),
                                         Expr.constTerm(2.0 * np.sqrt(area[i_soft]))),
                             Domain.inQCone())
                M.constraint(Expr.sub(ws, Expr.mul(ar_max, hs)),
                             Domain.lessThan(0.0))
                M.constraint(Expr.sub(hs, Expr.mul(ar_max, ws)),
                             Domain.lessThan(0.0))

            # MIB blocks must stay identical to each other
            ma, mb = [], []
            for g in sorted(set(int(v) for v in mib) - {0}):
                mem = [k for k in range(n) if mib[k] == g and not prep[k]]
                ma += mem[:-1]; mb += mem[1:]
            if ma:
                M.constraint(Expr.sub(pick(w, ma), pick(w, mb)),
                             Domain.equalsTo(0.0))
                M.constraint(Expr.sub(pick(h, ma), pick(h, mb)),
                             Domain.equalsTo(0.0))

            # ---- cluster contacts that already exist --------------------
            # Touching edges are legal, so a frozen pair must NOT also be asked
            # for the separation margin below -- the two rows contradict and
            # the model comes back infeasible.
            clu = [int(cons[k, 3]) for k in range(n)]
            abut_h, abut_v = set(), set()
            for g in sorted(set(clu) - {0}):
                mem = [k for k in range(n) if clu[k] == g]
                for a in mem:
                    for b in mem:
                        if a == b:
                            continue
                        if abs(P[a, 0] + P[a, 2] - P[b, 0]) < 1e-4:
                            abut_h.add((a, b))
                        if abs(P[a, 1] + P[a, 3] - P[b, 1]) < 1e-4:
                            abut_v.add((a, b))

            # ---- non-overlap, straight from the packing -----------------
            # SEP defaults to ZERO on purpose.  The evaluator calls a pair
            # overlapping only past 1e-6 and touching from -1e-6, so a gap
            # inside +-1e-6 is simultaneously legal and in contact -- exactly
            # where solver noise lives.  An earlier SEP of 1e-5 pushed every
            # pair OUT of that band: no overlaps, but no contacts either, and
            # the grouping violations doubled.
            SEP = float(_os.environ.get("SOCP_SEP", "0.0"))
            for E, ab, u, sz in ((Ch, abut_h, x, w), (Cv, abut_v, y, h)):
                if not E:
                    continue
                ea = [a for a, b in E]; eb = [b for a, b in E]
                rhs = np.array([0.0 if (a, b) in ab else -SEP for a, b in E])
                M.constraint(Expr.sub(Expr.add(pick(u, ea), pick(sz, ea)),
                                      pick(u, eb)), Domain.lessThan(rhs))

            # ---- frame and boundary codes -------------------------------
            M.constraint(Expr.sub(Expr.add(x, w), Var.repeat(W, n)),
                         Domain.lessThan(0.0))
            M.constraint(Expr.sub(Expr.add(y, h), Var.repeat(H, n)),
                         Domain.lessThan(0.0))
            free = ~prep
            for bit, fn in ((1, "L"), (2, "R"), (4, "T"), (8, "B")):
                sel = idx[free & ((bnd & bit) != 0)]
                if not len(sel):
                    continue
                if fn == "L":
                    M.constraint(pick(x, sel), Domain.equalsTo(0.0))
                elif fn == "B":
                    M.constraint(pick(y, sel), Domain.equalsTo(0.0))
                elif fn == "R":
                    M.constraint(Expr.sub(Expr.add(pick(x, sel), pick(w, sel)),
                                          Var.repeat(W, len(sel))),
                                 Domain.equalsTo(0.0))
                else:
                    M.constraint(Expr.sub(Expr.add(pick(y, sel), pick(h, sel)),
                                          Var.repeat(H, len(sel))),
                                 Domain.equalsTo(0.0))

            # ---- hold the frozen contacts together ----------------------
            # Freezing on ONE axis is not enough: two peers touch along a
            # vertical edge only while they still overlap in y, and nothing
            # else stops the solver sliding one up and the other down until
            # that overlap is gone -- edges aligned, contact lost.  So demand
            # the perpendicular overlap too, at half of what the packing had,
            # which is feasible by construction.
            for ab, u, sz, pu, psz in ((abut_h, x, w, y, h), (abut_v, y, h, x, w)):
                if not ab:
                    continue
                pairs = sorted(ab)
                ea = [a for a, b in pairs]; eb = [b for a, b in pairs]
                M.constraint(Expr.sub(Expr.add(pick(u, ea), pick(sz, ea)),
                                      pick(u, eb)), Domain.equalsTo(0.0))
                pc = 1 if pu is y else 0
                m = 0.5 * np.maximum(0.0, np.minimum(
                    P[ea, pc] + P[ea, pc + 2], P[eb, pc] + P[eb, pc + 2])
                    - np.maximum(P[ea, pc], P[eb, pc]))
                keep = m > EPS
                if keep.any():
                    ka = list(np.array(ea)[keep]); kb = list(np.array(eb)[keep])
                    mm = m[keep]
                    M.constraint(Expr.sub(Expr.add(pick(pu, ka), pick(psz, ka)),
                                          pick(pu, kb)), Domain.greaterThan(mm))
                    M.constraint(Expr.sub(Expr.add(pick(pu, kb), pick(psz, kb)),
                                          pick(pu, ka)), Domain.greaterThan(mm))

            # ---- objective ----------------------------------------------
            # Minimising W*H is not convex; minimising W+H is convex but pulls
            # a tall frame and a wide one equally, which is wrong.  The first
            # order expansion of the area at the current frame, H_now*W +
            # W_now*H, is linear AND weights each edge by how much frame the
            # other direction already has -- so it shrinks whichever side
            # actually costs area.
            # OBJECTIVE.  Two shapes of it, and the difference matters more
            # than it looks.
            #
            # "area": minimise the linearised bbox area, H_now*W + W_now*H.
            # Every scrap of frame is worth taking, including the scraps that
            # can only be had by pulling a cluster apart -- nothing in the model
            # prices contact, so the solver sells it.  Measured: area gap 0.162
            # -> 0.056, grouping violations 186 -> ~490, a bad trade under
            # exp(2*V).
            #
            # "outline" is what the paper actually does: fit inside a target
            # frame, min |W - W_f| + |H - H_f|, and once inside stop caring.
            # One-sided here, so the gradient is exactly zero within the
            # outline and the solver has no reason to disturb a layout that
            # already fits.  W_f is the current frame shrunk by SOCP_FIT.
            grav = float(_os.environ.get("SOCP_GRAV", "1e-3")) \
                * (H_now + W_now) / max(n, 1)
            pull = Expr.mul(grav, Expr.add(Expr.sum(x), Expr.sum(y)))
            if _os.environ.get("SOCP_OBJ", "area") == "outline":
                f = float(_os.environ.get("SOCP_FIT", "0.95"))
                Wf, Hf = W_now * f, H_now * f
                Dx = M.variable("Dx", 1, Domain.greaterThan(0.0))
                Dy = M.variable("Dy", 1, Domain.greaterThan(0.0))
                M.constraint(Expr.sub(Dx.index(0), Expr.sub(W.index(0), Wf)),
                             Domain.greaterThan(0.0))
                M.constraint(Expr.sub(Dy.index(0), Expr.sub(H.index(0), Hf)),
                             Domain.greaterThan(0.0))
                obj = Expr.add(Expr.add(Expr.mul(H_now, Dx.index(0)),
                                        Expr.mul(W_now, Dy.index(0))), pull)
            else:
                obj = Expr.add(Expr.add(Expr.mul(H_now, W.index(0)),
                                        Expr.mul(W_now, H.index(0))), pull)
            M.objective(ObjectiveSense.Minimize, obj)
            M.solve()
            _st = str(M.getPrimalSolutionStatus())
            if _st not in ("SolutionStatus.Optimal", "SolutionStatus.Feasible"):
                REJECT[_st] = REJECT.get(_st, 0) + 1
                return None
            out = np.stack([x.level(), y.level(), w.level(), h.level()], axis=1)
    except Exception as e:
        REJECT['exception: ' + type(e).__name__] = \
            REJECT.get('exception: ' + type(e).__name__, 0) + 1
        return None

    return _finish(out, n, P, area, soft, prep, bnd, ar_max, x_lo, y_lo)


def _finish(out, n, P, area, soft, prep, bnd, ar_max, x_lo, y_lo):
    """Everything both solvers need after they hand back numbers.

    Kept in one place deliberately: these steps are not cosmetic, they are what
    makes a solution the contest will actually accept, and a second backend
    that skipped any of them would fail in ways that look like the solver's
    fault.  A conic solver satisfies constraints to ITS tolerance; the contest
    checks them at 1e-6, and counts two blocks as touching only when their
    coordinates are the same double.
    """
    # ---- SNAP THE SHAPES BACK TO EXACTLY THEIR TARGET AREA ----
    # The cone only bounds the area from below.  A block that is not on the
    # critical path feels no pressure from the objective, so the solver is free
    # to leave it anywhere on or above the cone -- and the contest fails any
    # soft block whose area drifts more than 1% (89 of 100 cases before this).
    #
    # Scaling both sides by sqrt(A/(w*h)) <= 1 lands exactly on the target area,
    # keeps the aspect ratio (so the AR cap still holds), and only ever SHRINKS
    # a block, which cannot create an overlap that was not already there.  The
    # block is anchored to whichever edge it is coded for, so shrinking never
    # pulls it off a boundary it was satisfying.
    for i in range(n):
        if not soft[i]:
            continue
        cur = out[i, 2] * out[i, 3]
        if cur <= area[i] + 1e-12:
            continue
        f = np.sqrt(area[i] / cur)
        nw, nh = out[i, 2] * f, out[i, 3] * f
        c = bnd[i]
        if c & 2:                                  # R: keep the right edge
            out[i, 0] += out[i, 2] - nw
        if c & 4:                                  # T: keep the top edge
            out[i, 1] += out[i, 3] - nh
        out[i, 2], out[i, 3] = nw, nh

    # ---- SNAP THE BOUNDARY BLOCKS ONTO THE FRAME ----
    # MOSEK satisfies "y + h = H" to its own tolerance; at coordinates of order
    # 200 that lands at ~1.3e-6, just past the 1e-6 at which the contest stops
    # calling a block "on the edge".  Case 57 lost 11 boundary blocks that way,
    # every one of them visually flush.  Move them the last micron -- it is
    # below the overlap tolerance, so it cannot create one.
    bx0, by0 = out[:, 0].min(), out[:, 1].min()
    bx1 = (out[:, 0] + out[:, 2]).max(); by1 = (out[:, 1] + out[:, 3]).max()
    SNAP = float(_os.environ.get("SOCP_SNAP", "1e-3"))
    for i in range(n):
        c = bnd[i]
        if not c or prep[i]:
            continue
        if c & 1 and abs(out[i, 0] - bx0) < SNAP:
            out[i, 0] = bx0
        if c & 2 and abs(out[i, 0] + out[i, 2] - bx1) < SNAP:
            out[i, 0] = bx1 - out[i, 2]
        if c & 4 and abs(out[i, 1] + out[i, 3] - by1) < SNAP:
            out[i, 1] = by1 - out[i, 3]
        if c & 8 and abs(out[i, 1] - by0) < SNAP:
            out[i, 1] = by0

    # ---- verify: the solver works to a tolerance, the contest does not ----
    for i in range(n):
        for j in range(i + 1, n):
            ox = min(out[i, 0] + out[i, 2], out[j, 0] + out[j, 2]) \
                - max(out[i, 0], out[j, 0])
            oy = min(out[i, 1] + out[i, 3], out[j, 1] + out[j, 3]) \
                - max(out[i, 1], out[j, 1])
            if ox > 1e-6 and oy > 1e-6:
                REJECT['overlap'] = REJECT.get('overlap', 0) + 1
                return None                        # the evaluator's own test
    if (np.abs(out[:, 2] * out[:, 3] - area) > 0.01 * area)[soft].any():
        REJECT['area 1%'] = REJECT.get('area 1%', 0) + 1
        return None                                # outside the 1% tolerance
    ar = np.maximum(out[soft, 2] / out[soft, 3], out[soft, 3] / out[soft, 2])
    if soft.any() and ar.max() > ar_max + 1e-6:
        REJECT['aspect ratio'] = REJECT.get('aspect ratio', 0) + 1
        return None                                # aspect ratio cap broken
    out[:, 0] += x_lo; out[:, 1] += y_lo
    return out


# =====================================================================
# CLARABEL BACKEND -- the same model, without the licence
# =====================================================================
# MOSEK is commercial, and the licence lives on one developer's machine.  That
# is fine for experiments and not fine for a submission: the graders run the
# solver on their own hardware, where the licence does not exist and this whole
# stage silently returns None.  Clarabel is Apache-2.0, pip-installable, and an
# interior-point method like MOSEK, so it lands in much the same place.
#
# There is no modelling layer here.  Clarabel takes the raw conic form
#
#     minimise  q'z        subject to  A z + s = b,   s in K
#
# so the model is assembled as sparse triplets, which is if anything a
# advantage: building the MOSEK model through Fusion's per-constraint Python
# API cost 85% of that solver's total time, and there is none of it here.
#
# Variable layout:  z = [ x(n) | y(n) | w(n) | h(n) | W | H ]

def _solve_clarabel(n, positions, is_preplaced, constraints, ar_max, verbose, tol,
                    b2b=None, p2b=None, pins=None):
    try:
        import clarabel
        import scipy.sparse as sp
    except ImportError:
        return None

    P = np.asarray(positions, dtype=float)[:n].copy()
    cons = constraints
    area = P[:, 2] * P[:, 3]
    fixed = np.array([int(cons[i, 0]) > 0 for i in range(n)])
    prep = np.array([bool(is_preplaced[i]) for i in range(n)])
    mib = np.array([int(cons[i, 2]) for i in range(n)])
    bnd = np.array([int(cons[i, 4]) for i in range(n)])
    clu = np.array([int(cons[i, 3]) for i in range(n)])
    soft = ~prep & ~fixed & (mib == 0)

    x_lo = P[:, 0].min(); y_lo = P[:, 1].min()
    W_now = float((P[:, 0] + P[:, 2]).max() - x_lo)
    H_now = float((P[:, 1] + P[:, 3]).max() - y_lo)
    P[:, 0] -= x_lo; P[:, 1] -= y_lo

    Ch, Cv = build_constraint_graphs(P, n)
    IX, IY, IW, IH = 0, n, 2 * n, 3 * n
    iW, iH = 4 * n, 4 * n + 1

    # ---- WIRELENGTH IN THE OBJECTIVE -------------------------------
    # The model shapes and places every block and then scores itself on frame
    # area alone -- it never looks at a net.  hpwl_gap is 0.21 against an
    # area_gap of 0.05, so four fifths of what the cost function charges us was
    # invisible to the last solve that could still move anything.
    #
    # The contest's HPWL is weighted Manhattan between block CENTRES, which is
    # a linear program: one auxiliary variable per edge per axis, bounded below
    # by both signs of the difference, and minimised.  Centre is x + w/2, so
    # the shaping variables enter it too -- widening a block drags its centre.
    WIRE = float(_os.environ.get("SOCP_WIRE", "0.03"))
    eb, ep = [], []
    if WIRE > 0.0:
        if b2b is not None and len(b2b):
            for e in np.asarray(b2b, dtype=float):
                i, j, wt = int(e[0]), int(e[1]), float(e[2])
                if 0 <= i < n and 0 <= j < n and i != j and wt > 0:
                    eb.append((i, j, wt))
        if p2b is not None and len(p2b) and pins is not None and len(pins):
            pa = np.asarray(pins, dtype=float)
            for e in np.asarray(p2b, dtype=float):
                pi, bi, wt = int(e[0]), int(e[1]), float(e[2])
                if 0 <= bi < n and 0 <= pi < len(pa) and wt > 0:
                    ep.append((pi, bi, wt))
    # PRUNE THE LIGHT TAIL OF THE NETLIST.
    # Every edge costs two variables and four rows, and case 99 has 10580 of
    # them -- about 42000 rows, most of the model and most of the solve.  But
    # the weight is very unevenly spread: the heaviest 49% of case 99's edges
    # carry 80% of the total weight, so the rest can only move the objective by
    # a rounding error while costing half the solve time.  SOCP_WIRE_KEEP is
    # the fraction of TOTAL WEIGHT to model; 1.0 keeps everything.
    keep = float(_os.environ.get("SOCP_WIRE_KEEP", "1.0"))
    if 0.0 < keep < 1.0 and (eb or ep):
        allw = sorted((w for *_x, w in eb), reverse=True) + \
               sorted((w for *_x, w in ep), reverse=True)
        allw = sorted(allw, reverse=True)
        tot = sum(allw)
        run = 0.0
        cut = allw[-1] if allw else 0.0
        for w in allw:
            run += w
            if run >= keep * tot:
                cut = w
                break
        eb = [e for e in eb if e[2] >= cut]
        ep = [e for e in ep if e[2] >= cut]

    ITB = 4 * n + 2                       # b2b slacks: tx then ty
    ITP = ITB + 2 * len(eb)               # p2b slacks: ux then uy
    NV = ITP + 2 * len(ep)

    eq_r, eq_c, eq_v, eq_b = [], [], [], []
    lp_r, lp_c, lp_v, lp_b = [], [], [], []

    def add(rs, cs, vs, bl, terms, rhs):
        r = len(bl)
        for c, v in terms:
            rs.append(r); cs.append(c); vs.append(v)
        bl.append(rhs)

    def eq(terms, rhs):
        add(eq_r, eq_c, eq_v, eq_b, terms, rhs)

    def le(terms, rhs):
        add(lp_r, lp_c, lp_v, lp_b, terms, rhs)

    # ---- equalities ------------------------------------------------
    for i in range(n):
        if prep[i]:
            eq([(IX + i, 1.0)], P[i, 0]); eq([(IY + i, 1.0)], P[i, 1])
            eq([(IW + i, 1.0)], P[i, 2]); eq([(IH + i, 1.0)], P[i, 3])
        elif not soft[i]:
            eq([(IW + i, 1.0)], P[i, 2]); eq([(IH + i, 1.0)], P[i, 3])
    for g in sorted(set(int(v) for v in mib) - {0}):
        mem = [k for k in range(n) if mib[k] == g and not prep[k]]
        for a, b in zip(mem, mem[1:]):
            eq([(IW + a, 1.0), (IW + b, -1.0)], 0.0)
            eq([(IH + a, 1.0), (IH + b, -1.0)], 0.0)

    abut_h, abut_v = set(), set()
    for g in sorted(set(int(v) for v in clu) - {0}):
        mem = [k for k in range(n) if clu[k] == g]
        for a in mem:
            for b in mem:
                if a == b:
                    continue
                if abs(P[a, 0] + P[a, 2] - P[b, 0]) < 1e-4:
                    abut_h.add((a, b))
                if abs(P[a, 1] + P[a, 3] - P[b, 1]) < 1e-4:
                    abut_v.add((a, b))
    # BOUNDARY AND CLUSTER ARE SOFT CONSTRAINTS IN THE RULES, HARD HERE.
    # That is safe when the input is already a legal packing -- the equalities
    # then agree with the constraint graph read off it.  It is NOT safe when
    # the input still overlaps: the graph says one block is right of another
    # while the boundary says it sits at x=0, the two cannot both hold, and the
    # solver returns PrimalInfeasible.  That is exactly why feeding stage 2
    # straight in gave 7 feasible cases out of 100.
    # SOCP_BND / SOCP_CLU = "off" drops them, so the model stays feasible and
    # the violations are paid for in the score instead of in feasibility.
    _clu_on = _os.environ.get("SOCP_CLU", "hard") != "off"
    _bnd_on = _os.environ.get("SOCP_BND", "hard") != "off"
    if _clu_on:
        for a, b in sorted(abut_h):
            eq([(IX + a, 1.0), (IW + a, 1.0), (IX + b, -1.0)], 0.0)
        for a, b in sorted(abut_v):
            eq([(IY + a, 1.0), (IH + a, 1.0), (IY + b, -1.0)], 0.0)
    for i in range(n):
        if prep[i] or not bnd[i] or not _bnd_on:
            continue
        if bnd[i] & 1:
            eq([(IX + i, 1.0)], 0.0)
        if bnd[i] & 8:
            eq([(IY + i, 1.0)], 0.0)
        if bnd[i] & 2:
            eq([(IX + i, 1.0), (IW + i, 1.0), (iW, -1.0)], 0.0)
        if bnd[i] & 4:
            eq([(IY + i, 1.0), (IH + i, 1.0), (iH, -1.0)], 0.0)

    # ---- inequalities ----------------------------------------------
    SEP = float(_os.environ.get("SOCP_SEP", "0.0"))
    for a, b in Ch:
        if (a, b) in abut_h:
            continue                                   # already an equality
        le([(IX + a, 1.0), (IW + a, 1.0), (IX + b, -1.0)], -SEP)
    for a, b in Cv:
        if (a, b) in abut_v:
            continue
        le([(IY + a, 1.0), (IH + a, 1.0), (IY + b, -1.0)], -SEP)
    for a, b in sorted(abut_h):                        # keep the y-overlap
        m = 0.5 * max(0.0, min(P[a, 1] + P[a, 3], P[b, 1] + P[b, 3])
                      - max(P[a, 1], P[b, 1]))
        if m > EPS:
            le([(IY + b, 1.0), (IY + a, -1.0), (IH + a, -1.0)], -m)
            le([(IY + a, 1.0), (IY + b, -1.0), (IH + b, -1.0)], -m)
    for a, b in sorted(abut_v):                        # keep the x-overlap
        m = 0.5 * max(0.0, min(P[a, 0] + P[a, 2], P[b, 0] + P[b, 2])
                      - max(P[a, 0], P[b, 0]))
        if m > EPS:
            le([(IX + b, 1.0), (IX + a, -1.0), (IW + a, -1.0)], -m)
            le([(IX + a, 1.0), (IX + b, -1.0), (IW + b, -1.0)], -m)
    # ---- WIRE MUST NOT BE BOUGHT WITH FRAME -------------------------
    # With a wirelength term in the objective the solver happily pays for
    # shorter nets by letting the frame grow: hpwl_gap fell 0.2127 -> 0.1845
    # but area_gap rose 0.0512 -> 0.0686 and both of the heaviest cases got
    # worse.  The frame is not the wire's to spend, so nail it down and let the
    # nets be shortened only inside the box the packing already earned.
    if (eb or ep) and _os.environ.get("SOCP_WIRE_CAP", "1") == "1":
        le([(iW, 1.0)], W_now); le([(iH, 1.0)], H_now)

    for i in range(n):
        le([(IX + i, 1.0), (IW + i, 1.0), (iW, -1.0)], 0.0)
        le([(IY + i, 1.0), (IH + i, 1.0), (iH, -1.0)], 0.0)
        le([(IX + i, -1.0)], 0.0); le([(IY + i, -1.0)], 0.0)
        le([(IW + i, -1.0)], 0.0); le([(IH + i, -1.0)], 0.0)
        if soft[i]:
            le([(IW + i, 1.0), (IH + i, -ar_max)], 0.0)
            le([(IH + i, 1.0), (IW + i, -ar_max)], 0.0)

    for k, (i, j, wt) in enumerate(eb):        # t >= |c_i - c_j| on each axis
        for ax, IP, IS in ((0, IX, IW), (1, IY, IH)):
            t = ITB + k + ax * len(eb)
            le([(IP + i, 1.0), (IS + i, 0.5),
                (IP + j, -1.0), (IS + j, -0.5), (t, -1.0)], 0.0)
            le([(IP + j, 1.0), (IS + j, 0.5),
                (IP + i, -1.0), (IS + i, -0.5), (t, -1.0)], 0.0)
    for k, (pi, bi, wt) in enumerate(ep):      # u >= |c_i - pin|, pin constant
        for ax, IP, IS in ((0, IX, IW), (1, IY, IH)):
            u = ITP + k + ax * len(ep)
            pv = float(np.asarray(pins, dtype=float)[pi][ax]) \
                - (x_lo if ax == 0 else y_lo)
            le([(IP + bi, 1.0), (IS + bi, 0.5), (u, -1.0)], pv)
            le([(IP + bi, -1.0), (IS + bi, -0.5), (u, -1.0)], -pv)

    # ---- one second-order cone per soft block ----------------------
    so_r, so_c, so_v, so_b, cone_dims = [], [], [], [], []
    for i in np.arange(n)[soft]:
        r = len(so_b)
        so_r += [r, r]; so_c += [IH + i, IW + i]; so_v += [-1.0, -1.0]
        so_b.append(0.0)                               # s0 = h + w
        so_r += [r + 1, r + 1]; so_c += [IH + i, IW + i]; so_v += [-1.0, 1.0]
        so_b.append(0.0)                               # s1 = h - w
        so_b.append(2.0 * np.sqrt(area[i]))            # s2 = 2*sqrt(A)
        cone_dims.append(3)

    n_eq, n_lp = len(eq_b), len(lp_b)
    rows = np.concatenate([np.array(eq_r, dtype=int),
                           np.array(lp_r, dtype=int) + n_eq,
                           np.array(so_r, dtype=int) + n_eq + n_lp])
    colv = np.concatenate([np.array(eq_c, dtype=int), np.array(lp_c, dtype=int),
                           np.array(so_c, dtype=int)])
    vals = np.concatenate([np.array(eq_v), np.array(lp_v), np.array(so_v)])
    bvec = np.concatenate([np.array(eq_b), np.array(lp_b), np.array(so_b)])
    Amat = sp.csc_matrix((vals, (rows, colv)),
                         shape=(n_eq + n_lp + len(so_b), NV))

    q = np.zeros(NV)
    q[iW] = H_now; q[iH] = W_now
    if eb or ep:
        # Price a unit of wire against a unit of frame, so SOCP_WIRE is a
        # dimensionless trade-off and not a magnitude that has to be retuned
        # per case.
        cx = P[:, 0] + P[:, 2] / 2; cy = P[:, 1] + P[:, 3] / 2
        hp = sum(wt * (abs(cx[i] - cx[j]) + abs(cy[i] - cy[j])) for i, j, wt in eb)
        pa = np.asarray(pins, dtype=float) if ep else None
        hp += sum(wt * (abs(cx[bi] - (pa[pi][0] - x_lo))
                        + abs(cy[bi] - (pa[pi][1] - y_lo)))
                  for pi, bi, wt in ep)
        cf = WIRE * (H_now * W_now) / max(hp, 1e-9)
        for k, (_i, _j, wt) in enumerate(eb):
            q[ITB + k] += cf * wt; q[ITB + len(eb) + k] += cf * wt
        for k, (_p, _b, wt) in enumerate(ep):
            q[ITP + k] += cf * wt; q[ITP + len(ep) + k] += cf * wt
    grav = float(_os.environ.get("SOCP_GRAV", "1e-3")) * (H_now + W_now) / max(n, 1)
    mu = float(_os.environ.get("SOCP_MU", "0.01")) * (H_now + W_now) / max(n, 1)
    q[IX:IX + 2 * n] += grav
    q[IW:IW + 2 * n] += mu

    cones = [clarabel.ZeroConeT(n_eq), clarabel.NonnegativeConeT(n_lp)] + \
            [clarabel.SecondOrderConeT(d) for d in cone_dims]
    st = clarabel.DefaultSettings()
    st.verbose = False
    st.tol_gap_abs = st.tol_gap_rel = tol
    st.tol_feas = tol
    try:
        sol = clarabel.DefaultSolver(sp.csc_matrix((NV, NV)), q, Amat, bvec,
                                     cones, st).solve()
    except Exception as e:
        REJECT['clarabel ' + type(e).__name__] = \
            REJECT.get('clarabel ' + type(e).__name__, 0) + 1
        return None
    if str(sol.status) not in ("Solved", "AlmostSolved",
                               "SolverStatus.Solved", "SolverStatus.AlmostSolved"):
        REJECT['clarabel ' + str(sol.status)] = \
            REJECT.get('clarabel ' + str(sol.status), 0) + 1
        return None
    z = np.asarray(sol.x)
    out = np.stack([z[IX:IX + n], z[IY:IY + n],
                    z[IW:IW + n], z[IH:IH + n]], axis=1)
    return _finish(out, n, P, area, soft, prep, bnd, ar_max, x_lo, y_lo)
