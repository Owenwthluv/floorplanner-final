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

# RUN FROM ANYWHERE.
# The sibling modules below are imported by bare name, which only resolves when
# the interpreter's working directory happens to be this one.  The grader loads
# a submission with spec_from_file_location on an absolute path and need not cd
# here first -- and then every one of these imports raises and the submission
# scores nothing.  Put our own directory on the path first so the file is
# self-locating.
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

from iccad2026_evaluate import FloorplanOptimizer
from model import FloorplanGNN
from ml_utils import build_pyg_graph
from stage2_electrostatic import stage2_electrostatic
from stage3_legalizer import stage3_legalizer

# STAGE 1 checkpoint.  Kept in its own file so it can be swapped
# without touching the code; 13 input channels (see build_pyg_graph).
# Resolved against THIS FILE, not the working directory.  A relative name
# silently misses when the grader runs from elsewhere, and the miss is not an
# error: the optimizer prints a warning and carries on with a RANDOMLY
# INITIALISED network, which is how the server run quietly scored a pipeline
# with no trained stage 1 in it at all.
WEIGHTS = _os.environ.get("GNN_WEIGHTS", "floorplan_gnn_ar9_final.pth")
if not _os.path.isabs(WEIGHTS):
    _cand = _os.path.join(_HERE, WEIGHTS)
    if _os.path.exists(_cand):
        WEIGHTS = _cand


def _score(P, n, cons, e_b2b, e_p2b, pins):
    """The contest cost of a layout, up to the case's own unknown baselines.

    cost = (1 + a*(hpwl/HPWL_gt + area/AREA_gt - 2)) * exp(b*V_rel).  Both
    baselines are constants OF THE CASE, so dropping them leaves a quantity
    that orders two candidates for the SAME case exactly as the real cost does.
    Comparing across cases with it would be meaningless; within one it is exact.

    Grouping is counted with Shapely, as the evaluator counts it, because that
    is the only test that agrees: its tolerance for "touching" is zero, and any
    epsilon-based approximation reports contacts the contest does not.

    Returns the three ingredients rather than a number, because they cannot be
    added until they are on the same scale -- see _pick.
    """
    Q = P[:n]
    cx = Q[:, 0] + Q[:, 2] / 2
    cy = Q[:, 1] + Q[:, 3] / 2
    hpwl = 0.0
    if len(e_b2b):
        a = e_b2b[:, 0].astype(int); b = e_b2b[:, 1].astype(int)
        hpwl += float((e_b2b[:, 2] * (np.abs(cx[a] - cx[b])
                                      + np.abs(cy[a] - cy[b]))).sum())
    if len(e_p2b):
        q = e_p2b[:, 0].astype(int); b = e_p2b[:, 1].astype(int)
        hpwl += float((e_p2b[:, 2] * (np.abs(pins[q, 0] - cx[b])
                                      + np.abs(pins[q, 1] - cy[b]))).sum())
    x0, y0 = Q[:, 0].min(), Q[:, 1].min()
    x1 = (Q[:, 0] + Q[:, 2]).max(); y1 = (Q[:, 1] + Q[:, 3]).max()
    bnd = cons[:n, 4].astype(int)
    clu = cons[:n, 3].astype(int)
    mib = cons[:n, 2].astype(int)
    v = int(((bnd & 1).astype(bool) & (np.abs(Q[:, 0] - x0) > 1e-6)).sum()
            + ((bnd & 2).astype(bool) & (np.abs(Q[:, 0] + Q[:, 2] - x1) > 1e-6)).sum()
            + ((bnd & 4).astype(bool) & (np.abs(Q[:, 1] + Q[:, 3] - y1) > 1e-6)).sum()
            + ((bnd & 8).astype(bool) & (np.abs(Q[:, 1] - y0) > 1e-6)).sum())
    n_soft = int((bnd != 0).sum())
    try:
        from shapely.geometry import box
        from shapely.ops import unary_union
    except ImportError:
        box = None
    for g in set(int(t) for t in clu) - {0}:
        mem = [i for i in range(n) if clu[i] == g]
        if len(mem) < 2:
            continue
        n_soft += len(mem) - 1
        if box is None:
            continue
        u = unary_union([box(Q[i, 0], Q[i, 1], Q[i, 0] + Q[i, 2],
                             Q[i, 1] + Q[i, 3]) for i in mem])
        if u.geom_type == "MultiPolygon":
            v += len(u.geoms) - 1
    for g in set(int(t) for t in mib) - {0}:
        mem = [i for i in range(n) if mib[i] == g]
        if len(mem) < 2:
            continue
        n_soft += len(mem) - 1
        v += len({(round(float(Q[i, 2]), 4), round(float(Q[i, 3]), 4))
                  for i in mem}) - 1
    return hpwl, float((x1 - x0) * (y1 - y0)), v / max(n_soft, 1)


# ONE FAILURE DISABLES FORKING FOR THE WHOLE RUN.
# The watchdog turns a hung child into a fallback, but only for that case: if
# the grader's machine deadlocks every fork -- a real possibility, since torch
# maps libomp here and forking a process with a live OpenMP pool can leave the
# child holding a lock nobody will release -- we would pay the timeout a
# hundred times over.  So the first failure of any kind latches this off and
# the rest of the run goes straight to threads.
_FORK_OFF = [False]


def _fork_map(fn, args):
    """Run fn over args in forked children.  Returns None if fork is unusable.

    ProcessPoolExecutor cannot be used here: it pickles the callable BY NAME,
    and the contest evaluator imports the optimizer with
    spec_from_file_location("optimizer_module", ...) without registering it in
    sys.modules, so that lookup fails in the child.  The pool then raised, the
    code fell back to threads, and the whole speedup silently did not happen --
    on the evaluator only, which is the one place it had to.

    Forking directly sidesteps it: the child inherits fn as a live closure and
    nothing about it is ever pickled.  Only the result travels, through a temp
    file rather than a pipe so a large payload cannot deadlock on a full buffer.
    """
    import os as _o, pickle as _p, tempfile as _t
    if not hasattr(_o, "fork") or _FORK_OFF[0]:
        return None
    kids = []
    try:
        for a in args:
            fd, path = _t.mkstemp(prefix="fpvar")
            _o.close(fd)
            pid = _o.fork()
            if pid == 0:                            # child
                try:
                    with open(path, "wb") as f:
                        _p.dump(fn(a), f)
                except BaseException:
                    try:
                        _o.unlink(path)
                    except OSError:
                        pass
                _o._exit(0)
            kids.append((pid, path))
    except OSError:
        for pid, path in kids:                      # fork ran out: clean up
            try:
                _o.waitpid(pid, 0)
            except OSError:
                pass
            try:
                _o.unlink(path)
            except OSError:
                pass
        _FORK_OFF[0] = True
        return None
    # WATCHDOG.  This process has libomp mapped (torch pulls it in) and forking
    # a process whose OpenMP or BLAS pool has live threads can leave the child
    # holding a lock no thread survived to release -- it then blocks forever.
    # We have not seen it here in 500 forks, but "not seen on this machine" is
    # not a property of the grader's machine.  A hung child would take the
    # whole submission down, which is far worse than being slow, so every child
    # gets a deadline: miss it and they are killed and the caller falls back to
    # threads, which is the old behaviour and always correct.
    import signal as _sg, time as _tm
    limit = float(_os.environ.get("PACK_FORK_TIMEOUT", "60"))
    deadline = _tm.monotonic() + limit
    pending = {pid for pid, _ in kids}
    while pending and _tm.monotonic() < deadline:
        for pid in list(pending):
            try:
                done, _st = _o.waitpid(pid, _o.WNOHANG)
            except OSError:
                pending.discard(pid); continue
            if done:
                pending.discard(pid)
        if pending:
            _tm.sleep(0.002)
    if pending:                                     # timed out: abandon them
        for pid in pending:
            try:
                _o.kill(pid, _sg.SIGKILL); _o.waitpid(pid, 0)
            except OSError:
                pass
        for _pid, path in kids:
            try:
                _o.unlink(path)
            except OSError:
                pass
        _FORK_OFF[0] = True        # see the note by _FORK_OFF
        return None
    out = []
    for pid, path in kids:
        try:
            with open(path, "rb") as f:
                out.append(_p.load(f))
        except BaseException:
            out.append(None)
        finally:
            try:
                _o.unlink(path)
            except OSError:
                pass
    if any(o is None for o in out):
        _FORK_OFF[0] = True
        return None
    return out


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
        # FIX: forcing a hard MIB member's exact (w,h) onto every soft
        # sibling regardless of the sibling's own target area guarantees an
        # area-tolerance violation (infeasible) once they differ by >1%.
        # Held-out check on floorset_lite/ (disjoint from the 100 visible
        # cases) found this broke 5/5 sampled cases; invisible on the
        # visible set. Only reuse (fw, fh) when it's within tolerance of the
        # sibling's own area; otherwise use the hard member's aspect ratio
        # applied to the sibling's own area.
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
                            a = float(area_targets[idx]) if area_targets[idx] > 0 else 1.0
                            if fw is not None and abs(float(fw) * float(fh) - a) / max(a, 1e-6) <= 0.01:
                                positions[idx, 2], positions[idx, 3] = fw, fh
                            elif fw is not None:
                                ar = max(0.5, min(2.0, float(fw) / float(fh)))
                                positions[idx, 2] = math.sqrt(a * ar)
                                positions[idx, 3] = a / positions[idx, 2]
                            else:
                                positions[idx, 2] = math.sqrt(a * avg_ar)
                                positions[idx, 3] = a / positions[idx, 2]

        cons_np = constraints.cpu().numpy()
        # boundary mark as the GNN drew it, BEFORE density spreading
        _cd = cons_np[:, 4].astype(int)
        _l = [positions[i, 0] for i in range(block_count) if _cd[i] & 1]
        _r = [positions[i, 0] + positions[i, 2]
              for i in range(block_count) if _cd[i] & 2]
        w_hint = float(np.median(_r) - np.median(_l)) if (_l and _r) else None
        # Diagnostic seams.  One slot each, overwritten per case, so they
        # cost a single array copy and cannot grow.  They are what let the
        # stages be compared against ground truth without re-running anything.
        globals()["LAST_STAGE1"] = positions[:block_count].copy()
        stage2_electrostatic(block_count, positions, is_preplaced_arr,
                         area_targets=area_targets.cpu().numpy(), constraints=cons_np,
                         b2b_edges=b2b_connectivity.cpu().numpy(),
                         cap_start=float(_os.environ.get("LD_CAP_START", "6.0")),
                         cap_end=float(_os.environ.get("LD_CAP_END", "1.0")),
                         anneal_frac=float(_os.environ.get("LD_ANNEAL", "0.6")),
                         # 900 -> 450: profiling showed this stage as the
                         # single biggest cost in solve() (case 99: 1.26s of
                         # 3.47s), and 900 rounds is more than stage3 needs
                         # from it (a density-ordered starting layout, not a
                         # converged one). Swept 900/450/350/225 on the full
                         # suite, runtime-adjusted: 450 is the sweet spot --
                         # lower saves more runtime but quality creeps back
                         # up on the heavily-weighted large cases.
                         rounds=int(_os.environ.get("LD_ROUNDS", "450")),
                         lr=float(_os.environ.get("LD_LR", "1.0")),
                         w_battr=float(_os.environ.get("LD_BATTR", "3.0")),
                         w_cattr=float(_os.environ.get("LD_CATTR", "2.0")),
                         w_conn=float(_os.environ.get("LD_CONN", "1.0")),
                         w_grav=float(_os.environ.get("LD_GRAV", "0.0")),
                         p2b_edges=p2b_connectivity.cpu().numpy(),
                         pins_pos=pins_pos.cpu().numpy(),
                         w_pin=float(_os.environ.get("LD_PIN", "1.5")),
                         mass_exp=float(_os.environ.get("LD_MASS", "0.0")))
        # =================================================================
        # STAGES 3-5, SWEPT OVER FRAME WIDTHS
        # =================================================================
        # The frame width comes out of a target utilisation and an aspect ratio
        # clamped to [0.25, 4.0].  That is one guess, and measured against the
        # ground truth it lands anywhere from 0.64x to 1.48x the right answer --
        # so on some cases the packing is solving the wrong problem before it
        # starts, and nothing downstream can recover a frame that was never
        # shaped right.
        #
        # So try several widths and keep whichever actually scores best.  The
        # contest's baselines are unknown, but they are CONSTANTS OF THE CASE,
        # so they cancel when two candidates for the same case are compared --
        # this ranking is exact, not a heuristic, and it uses the evaluator's
        # own predicates (Shapely for grouping, 1e-6 for edges).
        base = positions.copy()
        # Diagnostic seam: the global placement, before any legalizing.
        # Its HPWL is the wirelength the TOPOLOGY is worth, separate from what
        # packing later spends -- the two are worth telling apart.
        globals()["LAST_STAGE2"] = base.copy()
        # WIDTH SWEEP DEFAULT: single scale, not five.
        # Each scale re-runs the full stage3+refine+SOCP pipeline, and
        # PACK_WDEDUP only collapses scales that land on the identical
        # achievable frame -- most cases still pay for all 5. Runtime-
        # adjusted against real per-case median runtime, one scale beats
        # both 5-scale and a 2-scale compromise despite worse raw quality
        # alone. Override with PACK_WSWEEP to restore the sweep.
        scales = [float(v) for v in
                  _os.environ.get("PACK_WSWEEP", "1.0").split(",")]
        # CENTRE THE SWEEP ON WHAT THE BOTTOM ROW CAN ACTUALLY REACH.
        #
        # The scales above multiply sqrt(A/util * ar), a formula the bottom row
        # frequently overrules: on case 98 it asks for 138.5 and the row comes
        # back needing 173.3, so every candidate below 1.25x was asking for a
        # frame the row cannot fit in and silently got 173.3 instead.  Half the
        # sweep was spent re-measuring the same layout.
        #
        # One cheap probe fixes it.  Run the row once, read the width it
        # actually needed, and if that exceeds what the formula asked, re-centre
        # the scales on the achievable value.  Costs one extra pack of the
        # bottom row, not a whole variant.
        if _os.environ.get("PACK_WPROBE", "1") == "1" and len(scales) > 1:
            import stage3_legalizer as _s3
            _s3.ROW_NEED.clear(); _s3.ASKED_W.clear()
            _probe = base.copy()
            try:
                stage3_legalizer(block_count, _probe, is_preplaced_arr, cons_np,
                                 b2b=e_b2b, p2b=e_p2b, pins=pin_np,
                                 w_hint=w_hint, w_scale=1.0)
            except Exception:
                pass
            if _s3.ROW_NEED and _s3.ASKED_W:
                need = _s3.ROW_NEED[-1]; asked = _s3.ASKED_W[-1]
                if need > asked * 1.02:
                    scales = sorted({round(s_ * need / asked, 4) for s_ in scales})
                # DROP THE SCALES THAT ASK FOR THE SAME FRAME.
                # The formula's width is a request; `W = max(W, row_end)` is
                # the answer, and when the bottom row is the wider of the two
                # every scale below it gets the same frame and therefore the
                # same layout.  Case 99 ran all five widths to five identical
                # results: 1.67s of work done five times, 5.42s of wall clock
                # for it because the threads contend, and against the contest's
                # own median that alone costs 0.133 of Total Score -- more than
                # every quality change in this session put together.
                if _os.environ.get("PACK_WDEDUP", "1") == "1":
                    seen = {}
                    for s_ in scales:
                        seen.setdefault(round(max(asked * s_, need), 6), s_)
                    scales = sorted(seen.values())
        # A case whose top edge is nailed down by a preplaced block does not
        # need its width guessed at all -- the height is given, so the width
        # follows from the area.  Offer that as an extra candidate (w_scale 0
        # is the mode flag) wherever such a block exists; the scorer keeps it
        # only if it really is better.
        # A negative scale means "pack the transposed problem".  Offer it only
        # where the top edge is actually pinned by a preplaced block, since
        # that is the case the packer is aimed the wrong way for.  The scorer
        # keeps it only if it really wins.
        if _os.environ.get("PACK_TPIN", "1") == "1" and len(scales) > 1:
            if any(int(cons_np[i, 1]) and int(cons_np[i, 4]) & 4
                   for i in range(block_count)):
                scales = scales + [-float(v) for v in
                                   _os.environ.get("PACK_TSWEEP", "1.0,0.9,1.1").split(",")]
        e_b2b = b2b_connectivity.cpu().numpy()
        e_p2b = p2b_connectivity.cpu().numpy()
        pin_np = pins_pos.cpu().numpy()
        e_b2b = e_b2b[e_b2b[:, 0] != -1] if len(e_b2b) else e_b2b
        e_p2b = e_p2b[e_p2b[:, 0] != -1] if len(e_p2b) else e_p2b

        def _transpose(Q, C, pins2):
            """Reflect the whole problem across the diagonal.

            The packer has one hard assumption: the WIDTH is given and the
            HEIGHT is what grows.  It lays the bottom row across the full
            width, raises the two side towers, and fills upward.  When a
            preplaced block pins the TOP edge, that assumption is backwards --
            the height is the given one and the width is free -- and no amount
            of choosing W better can fix a packer aimed the wrong way.

            Transposing swaps the two roles instead.  A block's (x, y, w, h)
            becomes (y, x, h, w) and its boundary code rotates with it, L<->B
            and R<->T, so the pinned top arrives as a pinned RIGHT edge --
            which the packer already honours, since PIN["R"] snaps W to it.
            The real left edge arrives as the bottom row, and growth runs along
            what used to be the width.  Nothing inside the packer changes.
            """
            Q2 = Q.copy()
            Q2[:, [0, 1, 2, 3]] = Q[:, [1, 0, 3, 2]]
            C2 = C.copy()
            b = C[:, 4].astype(int)
            C2[:, 4] = ((b & 1) << 3) | ((b & 8) >> 3) | \
                       ((b & 2) << 1) | ((b & 4) >> 1)
            return Q2, C2, pins2[:, [1, 0]] if len(pins2) else pins2

        # The contest allows a soft block's realised area to sit within 1% of
        # its target, and stage 5 spends a slice of that to close the last
        # sliver of a split cluster.  It needs the TARGET, not the realised
        # area, or the budget would drift with every earlier reshape.
        _area_tgt = np.asarray(area_targets, dtype=float)[:block_count].ravel()

        def _variant(sc):
            P = base.copy()
            flip = (sc < 0)
            cons_v, pins_v = cons_np, pin_np
            if flip:
                sc = -sc
                P, cons_v, pins_v = _transpose(P, cons_np, pin_np)
            # PACK_SKIP3: hand the OVERLAPPING global placement straight to the
            # SOCP instead of legalizing it first.  This is the pipeline three
            # papers converge on -- PeF, UFO, and the Voronoi two-stage method
            # all read a relative-position matrix off the global placement and
            # let one convex solve do the legalizing.  Ours instead reads the
            # matrix off a packing the greedy legalizer already committed to,
            # so the solver can only polish a topology it never got to choose.
            # build_constraint_graphs already handles overlapping pairs, so the
            # only thing standing between us and that pipeline is this call.
            if _os.environ.get("PACK_SKIP3", "0") != "1":
                stage3_legalizer(block_count, P, is_preplaced_arr, cons_v,
                                 b2b=e_b2b, p2b=e_p2b, pins=pins_v,
                                 w_hint=w_hint, w_scale=sc)
            if _os.environ.get("PACK_REFINE", "1") == "1":
                from refine_clusters import refine_clusters
                refine_clusters(P, block_count, cons_v, is_preplaced_arr,
                                areas=_area_tgt)
            if _os.environ.get("PACK_SOCP", "1") == "1":
                from stage4_socp import stage4_socp
                q = stage4_socp(block_count, P, is_preplaced_arr, cons_v,
                                ar_max=float(_os.environ.get("PACK_AR", "3.0")),
                                b2b=e_b2b, p2b=e_p2b, pins=pins_v)
                if q is not None:
                    P[:block_count] = q
                    if _os.environ.get("PACK_REFINE", "1") == "1":
                        from refine_clusters import refine_clusters
                        refine_clusters(P, block_count, cons_v, is_preplaced_arr,
                                areas=_area_tgt)
            if flip:                                   # back to real coordinates
                P = P[:, [1, 0, 3, 2]].copy()
            return P

        if len(scales) == 1:
            positions[:] = _variant(scales[0])
        else:
            # Run the widths concurrently.  Threads rather than processes: the
            # variants share the read-only inputs, so there is nothing to
            # pickle and no interpreter to start, and the expensive half of
            # each variant is MOSEK, which releases the GIL while it solves.
            # Every variant works on its own copy of the layout and touches no
            # shared state, so the only thing that has to be serialised is the
            # comparison at the end.
            # THE COMMENT ABOVE WAS TRUE OF MOSEK AND IS NOT TRUE NOW.
            # With Clarabel the solve is 0.36s of a 0.98s variant; the other
            # 0.62s is the legalizer and the model build, both pure Python
            # holding the GIL.  So five threads do not cost 1x, they cost about
            # 4x: case 99 measured 1.67s for one width and 5.42s for five.
            # Against the contest's own median runtime that pushes case 99 --
            # weight 0.63 of the whole Total -- off the 0.7 floor and costs
            # 0.133 of Total Score.  Processes actually run them at once.
            #
            # _variant is a closure and cannot be pickled, so the pool is
            # forked: the child inherits it through the fork and only an index
            # crosses the boundary.
            import concurrent.futures as _cf
            _mode = _os.environ.get("PACK_PARALLEL", "proc")
            outs = None
            if _mode == "proc":
                outs = _fork_map(_variant, scales)
            if outs is None:
                if _mode != "0":
                    with _cf.ThreadPoolExecutor(max_workers=len(scales)) as ex:
                        outs = list(ex.map(_variant, scales))
                else:
                    outs = [_variant(sc) for sc in scales]
            # The contest cost is 1 + a*(hpwl/HPWL_gt + area/AREA_gt - 2), and
            # the two gaps are each divided by their OWN baseline before they
            # are added.  Adding the raw numbers instead lets the larger one
            # decide everything: hpwl runs to hundreds of thousands, bbox area
            # to tens of thousands, so area barely registered and case 12 was
            # handed a frame 4% bigger for a wirelength gain that does not pay
            # for it.  The baselines are unknown but constant per case, so the
            # best value of each among the candidates stands in for them --
            # that keeps the two terms commensurate, which is all the ranking
            # needs.
            m = [_score(P, block_count, cons_np, e_b2b, e_p2b, pin_np)
                 for P in outs]
            h_ref = max(min(t[0] for t in m), 1e-9)
            a_ref = max(min(t[1] for t in m), 1e-9)
            best = min(range(len(outs)),
                       key=lambda k: (1.0 + 0.5 * (m[k][0] / h_ref
                                                   + m[k][1] / a_ref))
                       * math.exp(2.0 * m[k][2]))
            positions[:] = outs[best]

        return [(float(p[0]), float(p[1]), float(p[2]), float(p[3])) for p in positions]
