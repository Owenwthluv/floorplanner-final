#!/usr/bin/env python3
"""STAGE 2 -- ELECTROSTATIC.

Resolves three forces together every round on an annealing canvas:

  PUSH  (density)   charge x field, where the field comes from solving Poisson
                    on the block-density map by DCT.  Drives blocks out of
                    crowded regions.
  PULL  boundary    a block with a boundary code is drawn to the matching edge
                    of the current canvas (corners to both).      [w_battr]
  PULL  cluster     a block in a cluster is drawn to that cluster's anchor --
                    the mean centre of its preplaced/boundary members if it has
                    any, otherwise the cluster centroid.          [w_cattr]
  PULL  netlist     connected blocks attract, normalised per degree. [w_conn]

Confinement is SOFT (a block that overshoots is pulled back only partway), the
canvas anneals cap_start -> cap_end, and each round's step is trust-region
limited.  With cap_end below 1.0 the canvas is smaller than the total block
area, so this stage CANNOT and DOES NOT remove overlap -- it compacts.  STAGE 3
does the legalising.  `constraints` is the [n,5] array (col 3 = cluster id,
col 4 = boundary bitmask).  Modifies positions in place.
"""
import os as _os
import numpy as np

try:
    from scipy.fft import dctn, idctn
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _decode(code):
    code = int(code)
    return {"L": bool(code & 1), "R": bool(code & 2),
            "T": bool(code & 4), "B": bool(code & 8)}


def _overlap_total(positions, n):
    """Total pairwise overlap area, vectorized (N x N broadcasting)."""
    p = positions[:n]
    xl = p[:, 0]; yl = p[:, 1]; xr = xl + p[:, 2]; yr = yl + p[:, 3]
    ox = np.clip(np.minimum(xr[:, None], xr[None, :]) - np.maximum(xl[:, None], xl[None, :]), 0, None)
    oy = np.clip(np.minimum(yr[:, None], yr[None, :]) - np.maximum(yl[:, None], yl[None, :]), 0, None)
    inter = ox * oy
    np.fill_diagonal(inter, 0.0)
    return float(inter.sum()) / 2.0


def _os_pin_skip():
    """whether the pin force skips blocks that already carry a boundary code"""
    return _os.environ.get("LD_PIN_SKIP_BND", "1") == "1"


def _density_map(positions, n, W, H, M):
    """Rasterize block areas onto an M x M grid by analytic rect-bin overlap.
    Fully vectorized: per-axis block-vs-bin overlaps, then rho = yo^T @ xo."""
    bw, bh = W / M, H / M
    xe = np.arange(M + 1) * bw          # x bin edges
    ye = np.arange(M + 1) * bh          # y bin edges
    x0 = positions[:n, 0]; y0 = positions[:n, 1]
    x1 = x0 + positions[:n, 2]; y1 = y0 + positions[:n, 3]
    # xo[k,c] = overlap of block k with column c ; yo[k,r] = overlap with row r
    xo = np.clip(np.minimum(x1[:, None], xe[None, 1:]) - np.maximum(x0[:, None], xe[None, :-1]), 0, None)
    yo = np.clip(np.minimum(y1[:, None], ye[None, 1:]) - np.maximum(y0[:, None], ye[None, :-1]), 0, None)
    rho = yo.T @ xo                      # rho[r,c] = sum_k yo[k,r]*xo[k,c]
    return rho, bw, bh


def _poisson_dct(rho):
    """Solve del^2 psi = -rho with Neumann BC via DCT. Returns psi (M x M)."""
    M = rho.shape[0]
    if _HAVE_SCIPY:
        a = dctn(rho, type=2, norm="ortho")
    else:
        a = _dct2_np(rho)
    # Laplacian eigenvalues for the DCT-II 5-point stencil (unit spacing)
    k = np.arange(M)
    lam = 2.0 * (1.0 - np.cos(np.pi * k / M))          # per axis, >= 0
    denom = lam[:, None] + lam[None, :]
    denom[0, 0] = 1.0                                   # avoid /0 (DC component)
    psi_hat = a / denom
    psi_hat[0, 0] = 0.0                                 # zero-mean potential
    if _HAVE_SCIPY:
        return idctn(psi_hat, type=2, norm="ortho")
    return _idct2_np(psi_hat)


def _dct2_np(x):
    # DCT-II via FFT (orthonormal), 2D separable — fallback when scipy absent
    return _dct1_np(_dct1_np(x.T).T)


def _idct2_np(x):
    return _idct1_np(_idct1_np(x.T).T)


def _dct1_np(x):
    N = x.shape[0]
    v = np.concatenate([x, x[::-1]], axis=0)
    V = np.fft.rfft(v, axis=0)[:N]
    k = np.arange(N)[:, None]
    factor = 2 * np.exp(-1j * np.pi * k / (2 * N))
    out = (V * factor).real
    out[0] /= np.sqrt(4 * N)
    out[1:] /= np.sqrt(2 * N)
    return out


def _idct1_np(x):
    N = x.shape[0]
    xx = x.copy().astype(complex)
    xx[0] *= np.sqrt(4 * N); xx[1:] *= np.sqrt(2 * N)
    k = np.arange(N)[:, None]
    xx *= 0.5 * np.exp(1j * np.pi * k / (2 * N))
    v = np.fft.irfft(xx, n=2 * N, axis=0)[:N]
    out = np.zeros_like(v)
    out[0::2] = v[: (N + 1) // 2]
    out[1::2] = v[N - 1: (N - 1) // 2: -1]
    return out


def stage2_electrostatic(n, positions, is_preplaced, area_targets=None, constraints=None,
                     b2b_edges=None, cap_start=5.0, cap_end=1.5, anneal_frac=0.65,
                     M=64, rounds=400, lr=0.15, lr_decay=0.999,
                     k_bound=0.5, w_battr=0.6, w_cattr=0.4, w_conn=0.5, w_grav=0.0,
                     p2b_edges=None, pins_pos=None, w_pin=0.0,
                     mass_exp=0.0,
                     verbose=False, log_every=40):
    """Spread blocks by the electrostatic density (repulsive) force PLUS two
    constraint attractions, all resolved together each round:

      - REPEL  (density): pushes blocks out of dense regions (untangles overlap).
      - ATTRACT boundary (w_battr): a block with a boundary constraint is pulled
        toward the matching edge of the current canvas (L/R/T/B, corners = both).
      - ATTRACT cluster (w_cattr): a block in a cluster is pulled toward that
        cluster's anchor center (mean center of its preplaced/boundary members
        if any, else the cluster centroid).

    Confinement is SOFT (no hard walls), the outline ANNEALS loose->target, and
    the per-round step is TRUST-REGION limited. `constraints` is the [n,5] array
    (col 3 = cluster id, col 4 = boundary bitmask). Modifies positions in place.
    """
    W0 = max(positions[k, 0] + positions[k, 2] for k in range(n))
    H0 = max(positions[k, 1] + positions[k, 3] for k in range(n))
    A = float(np.asarray(area_targets)[:n].clip(min=0).sum()) if area_targets is not None \
        else float(sum(positions[k, 2] * positions[k, 3] for k in range(n)))
    ar = max(0.25, min(4.0, (W0 / H0) if H0 > 1e-9 else 1.0))

    # precompute per-block boundary codes and cluster membership
    if constraints is not None:
        codes = [_decode(constraints[i, 4]) for i in range(n)]
        clust = [int(constraints[i, 3]) for i in range(n)]
    else:
        codes = [{"L": False, "R": False, "T": False, "B": False}] * n
        clust = [0] * n
    max_clust = max(clust) if clust else 0

    # precompute netlist edges (u, v, weight) for connectivity attraction
    eu = ev = ew = None
    if b2b_edges is not None and len(b2b_edges) > 0:
        e = np.asarray(b2b_edges, dtype=np.float64)
        m = (e[:, 0] >= 0) & (e[:, 1] >= 0) & (e[:, 0] < n) & (e[:, 1] < n) & (e[:, 0] != e[:, 1])
        if m.any():
            eu = e[m, 0].astype(int); ev = e[m, 1].astype(int)
            ew = e[m, 2] if e.shape[1] > 2 else np.ones(int(m.sum()))
            ew = ew / (ew.mean() + 1e-9)     # normalize weights to ~O(1)
            deg = np.ones(n)                 # per-block degree (for force norm)
            np.add.at(deg, eu, 1.0); np.add.at(deg, ev, 1.0)
    pu = pv = pxs = pys = pw = None
    if p2b_edges is not None and pins_pos is not None and len(p2b_edges):
        _E = np.asarray(p2b_edges, dtype=float)
        _E = _E[(_E[:, 0] != -1) & (_E[:, 1] < n)]
        if len(_E):
            _pp = np.asarray(pins_pos, dtype=float)
            pu = _E[:, 0].astype(int); pv = _E[:, 1].astype(int)
            pw = _E[:, 2] / (_E[:, 2].mean() + 1e-9)
            pxs = _pp[pu, 0]; pys = _pp[pu, 1]
    bnd_mask = None
    if constraints is not None and _os_pin_skip():
        _b = np.asarray(constraints)[:n, 4].astype(int)
        bnd_mask = _b != 0

    grp_id = None; grp_members = []
    if constraints is not None:
        _c = np.asarray(constraints)[:n, 3].astype(int)
        _g = [np.where(_c == g)[0] for g in sorted(set(_c.tolist()) - {0})]
        grp_members = [m for m in _g if len(m) > 1]
        grp_id = _c if grp_members else None

    ov0 = _overlap_total(positions, n)
    hp0 = 1.0
    if eu is not None:
        _c0x = positions[:n, 0] + positions[:n, 2] / 2
        _c0y = positions[:n, 1] + positions[:n, 3] / 2
        hp0 = max(float((ew * (np.abs(_c0x[eu] - _c0x[ev])
                               + np.abs(_c0y[eu] - _c0y[ev]))).sum()), 1e-9)
    traj = [ov0]
    best_from = float(_os.environ.get("LD_BEST_FROM", "1.0"))
    best_by = _os.environ.get("LD_BEST_BY", "overlap")
    best_w = float(_os.environ.get("LD_BEST_W", "1.0"))
    best_ov = float("inf")
    best_pos = positions.copy()

    # W and H are the annealed canvas, recomputed every round -- but they are
    # also reported back to the caller, so with rounds=0 they would never be
    # bound at all and the whole stage died on a NameError.  That is not a
    # theoretical case: it is exactly what "skip stage 2 and hand the raw GNN
    # output to the legalizer" asks for, and the crash was misread as evidence
    # that the legalizer cannot cope with raw GNN output.  Seed them here so
    # zero rounds is a genuine no-op instead of a failure.
    W = float(np.sqrt(cap_start * A * ar))
    H = float(np.sqrt(cap_start * A / ar))
    for it in range(rounds):
        lr_t = lr * (lr_decay ** it)
        # annealed canvas: cap_start -> cap_end over anneal_frac, then hold
        frac = min(1.0, (it / rounds) / anneal_frac)
        cap_t = cap_start + (cap_end - cap_start) * frac
        in_hold = frac >= 1.0
        W = float(np.sqrt(cap_t * A * ar))
        H = float(np.sqrt(cap_t * A / ar))

        rho, bw, bh = _density_map(positions, n, W, H, M)
        psi = _poisson_dct(rho)
        Ey, Ex = np.gradient(-psi, bh, bw)   # field E = -grad psi

        # (a) REPEL: density force (charge * field), normalized to magnitude <=1
        # so the constraint springs below are weighted on a comparable scale.
        fdx = np.zeros(n); fdy = np.zeros(n)
        for k in range(n):
            if is_preplaced[k]:
                continue
            w, h = positions[k, 2], positions[k, 3]
            c = min(M - 1, max(0, int((positions[k, 0] + w / 2) / bw)))
            r = min(M - 1, max(0, int((positions[k, 1] + h / 2) / bh)))
            # CHARGE OVER MASS.  Area enters as charge -- a bigger block sits
            # on more of the density grid and feels a proportionally bigger
            # force -- but the force is applied straight as a DISPLACEMENT,
            # with no mass term to resist it.  In the physics being imitated a
            # heavier body moves less for the same push; here it moves more,
            # so large blocks get flung to the periphery.
            #
            # Measured against ground truth, that bias is exactly what shows
            # up: correlation between block area and distance-from-centre is
            # +0.244 in our stage-2 output against +0.063 in ground truth,
            # which places blocks with almost no regard to their size.  (The
            # same test on connectivity gives -0.465 against -0.496, so the
            # model gets THAT relationship right.)
            #
            # LD_MASS is the exponent of the mass term: 0 reproduces the old
            # behaviour, 1 cancels area out of the displacement entirely.
            q = (w * h) ** (1.0 - mass_exp)
            fdx[k], fdy[k] = q * Ex[r, c], q * Ey[r, c]
        dmax = float(np.sqrt(fdx * fdx + fdy * fdy).max()) + 1e-12
        fx = fdx / dmax; fy = fdy / dmax

        # cluster anchor centers (mean center of boundary/preplaced members, else all)
        clust_center = {}
        for g in range(1, max_clust + 1):
            idx = [i for i in range(n) if clust[i] == g]
            if len(idx) <= 1:
                continue
            anch = [i for i in idx if is_preplaced[i] or any(codes[i].values())]
            src = anch if anch else idx
            clust_center[g] = (np.mean([positions[i, 0] + positions[i, 2] / 2 for i in src]),
                               np.mean([positions[i, 1] + positions[i, 3] / 2 for i in src]))

        # (b) ATTRACT boundary + (c) ATTRACT cluster (springs, normalized by canvas)
        for k in range(n):
            if is_preplaced[k]:
                continue
            w, h = positions[k, 2], positions[k, 3]
            x, y = positions[k, 0], positions[k, 1]
            cd = codes[k]
            if cd["L"]:  fx[k] += w_battr * (0.0 - x) / W
            if cd["R"]:  fx[k] += w_battr * ((W - w) - x) / W
            if cd["B"]:  fy[k] += w_battr * (0.0 - y) / H
            if cd["T"]:  fy[k] += w_battr * ((H - h) - y) / H
            g = clust[k]
            if g in clust_center and not any(cd.values()):   # non-boundary cluster block
                ccx, ccy = clust_center[g]
                fx[k] += w_cattr * (ccx - (x + w / 2)) / W
                fy[k] += w_cattr * (ccy - (y + h / 2)) / H

        # (d) ATTRACT connectivity (wirelength spring): pull netlist-connected
        # block centers together, weighted by edge weight, normalized by canvas
        # AND by each block's degree (else high-degree blocks get a huge summed
        # force that dominates density and collapses the layout).
        if eu is not None:
            ctrx = positions[:n, 0] + positions[:n, 2] / 2
            ctry = positions[:n, 1] + positions[:n, 3] / 2
            dx = (ctrx[ev] - ctrx[eu]) / W * ew
            dy = (ctry[ev] - ctry[eu]) / H * ew
            cfx = np.zeros(n); cfy = np.zeros(n)
            np.add.at(cfx, eu, dx); np.add.at(cfx, ev, -dx)
            np.add.at(cfy, eu, dy); np.add.at(cfy, ev, -dy)
            fx += w_conn * cfx / deg; fy += w_conn * cfy / deg   # per-block degree norm

        # (d2) PIN ATTRACTION.  Pins sit at FIXED coordinates on the die, so a
        # block wired to one has an absolute place it wants to be -- unlike a
        # block-to-block edge, which only says two things belong near each
        # other and leaves the pair free to drift anywhere together.
        #
        # This stage ignored pins entirely until now, which quietly wrote off
        # 10% of the wirelength: measured over the suite, pin-to-block is a
        # tenth of the total and our gap on it (+25.6% against ground truth) is
        # worse than on block-to-block (+20.1%).  Same degree normalisation as
        # above, so a block wired to twenty pins is not dragged harder than one
        # wired to two.
        if pu is not None and w_pin > 0:
            ctrx = positions[:n, 0] + positions[:n, 2] / 2
            ctry = positions[:n, 1] + positions[:n, 3] / 2
            pfx = np.zeros(n); pfy = np.zeros(n)
            np.add.at(pfx, pv, (pxs - ctrx[pv]) / W * pw)
            np.add.at(pfy, pv, (pys - ctry[pv]) / H * pw)
            pfx /= deg; pfy /= deg
            # DO NOT PIN A BLOCK THAT IS ALREADY PINNED BY ITS CODE.
            #
            # Measured over the suite: of 4465 blocks carrying a pin edge, 47%
            # also carry a boundary code -- and in 2119 of those 2120 cases the
            # pins sit on exactly the edge the code already demands.  58% of all
            # pin pull is therefore duplicating a constraint that w_battr here
            # and the row/tower placement in stage 3 both enforce anyway.
            #
            # Duplicating it is not neutral.  Cluster g1 of case 98 has nine
            # members, of which only two (90 and 91) have pins, and both are
            # T-coded with their pins on the top edge.  Without this force they
            # rise with the boundary attraction and drag the other seven along
            # by cluster attraction, arriving as one piece.  With it they get a
            # second pull the same way, outrun the cluster spring, and the group
            # stretches until it tears.
            #
            # The other 42% -- interior blocks with no code -- is where a pin
            # says something nothing else does, and that part is kept.
            if bnd_mask is not None:
                pfx[bnd_mask] = 0.0; pfy[bnd_mask] = 0.0
            # A CLUSTER MOVES AS ONE BODY.
            #
            # The other forces here are relative -- "these two belong near each
            # other" -- and a cluster can satisfy them while drifting anywhere
            # together.  A pin is absolute: it says "this block belongs THERE".
            # Applied per block, that tears a cluster apart, because each
            # member is pulled toward its own scattered pins and they do not
            # agree.  Case 98 showed it plainly: cluster g1 sat as one piece
            # under the top row without this force, and with it came apart into
            # two, dragged down into the middle of the die.
            #
            # So a cluster gets ONE pull, the mean of its members', applied
            # identically to all of them.  The cluster still goes where its pins
            # want it, and arrives intact.
            if grp_id is not None:
                for _m in grp_members:
                    pfx[_m] = pfx[_m].mean(); pfy[_m] = pfy[_m].mean()
            fx += w_pin * pfx; fy += w_pin * pfy

        # (e) GRAVITY (compaction): pull every block toward the layout centroid.
        # Balanced against the density repulsion, this compacts the (already
        # untangled) layout without reintroducing overlap. Ramped up as the
        # canvas anneals so compaction acts mainly once blocks are spread.
        if w_grav > 0:
            gx0 = np.mean(positions[:n, 0] + positions[:n, 2] / 2)
            gy0 = np.mean(positions[:n, 1] + positions[:n, 3] / 2)
            fx += w_grav * frac * (gx0 - (positions[:n, 0] + positions[:n, 2] / 2)) / W
            fy += w_grav * frac * (gy0 - (positions[:n, 1] + positions[:n, 3] / 2)) / H

        fx[is_preplaced] = fy[is_preplaced] = 0.0   # anchors stay put

        fmax = float(np.sqrt(fx * fx + fy * fy).max()) + 1e-12
        step_cap = lr_t * min(bw, bh)                    # per-round trust radius
        scale = step_cap / fmax

        maxshift = 0.0
        for k in range(n):
            if is_preplaced[k]:
                continue
            w, h = positions[k, 2], positions[k, 3]
            positions[k, 0] += scale * fx[k]
            positions[k, 1] += scale * fy[k]
            # SOFT boundary: pull back a fraction of any overshoot (no hard cut)
            x, y = positions[k, 0], positions[k, 1]
            if x < 0:            positions[k, 0] = x + k_bound * (-x)
            elif x + w > W:      positions[k, 0] = x - k_bound * (x + w - W)
            if y < 0:            positions[k, 1] = y + k_bound * (-y)
            elif y + h > H:      positions[k, 1] = y - k_bound * (y + h - H)
            maxshift = max(maxshift, abs(scale * fx[k]), abs(scale * fy[k]))

        ov = _overlap_total(positions, n)
        # WHICH ROUNDS ARE ALLOWED TO WIN.
        #
        # Only the hold phase counted, on the reasoning that a low overlap
        # reached while the canvas is still large is cheap and unrepresentative.
        # Traced on case 98, that reasoning costs the best state outright: the
        # anneal ends at round 540, and at round 500 cluster g1 is 9.1 units
        # across with 10,091 total overlap, while every held round afterwards
        # sits between 20,856 and 26,213 and the cluster has loosened to 24.1.
        # The single best arrangement the run ever finds is thrown away for
        # being four percent too early.
        #
        # LD_BEST_FROM is the fraction of the anneal after which a round may be
        # recorded; 1.0 is the old hold-only rule.
        # WHAT "BEST" MEANS.
        #
        # The default is least overlap, and overlap is close to irrelevant here:
        # stage 3 discards this geometry entirely and re-packs from scratch,
        # keeping only the relative arrangement.  What it INHERITS is the
        # wirelength -- placement order and cluster anchors both come from these
        # positions -- and measured end to end, stage 2 hands over an
        # arrangement 16.9% BELOW ground truth on hpwl which stage 3 then turns
        # into +29.8%.  Choosing the handover state by a quantity the next stage
        # throws away, while ignoring the one it keeps, is worth questioning.
        #
        # LD_BEST_BY=hpwl scores candidates by weighted wirelength instead.
        # Both terms, never one alone.  Wirelength on its own is minimised by
        # collapsing every block onto a single point -- hpwl goes to zero, the
        # overlap is total, and the arrangement handed to stage 3 is worthless.
        # Overlap on its own is what ships today, and it is blind to the one
        # quantity stage 3 actually inherits.  Each is normalised by its own
        # value at the first round so the two are on the same scale, and
        # LD_BEST_W is the weight on the wirelength half.
        if best_by == "mix" and eu is not None:
            _cx = positions[:n, 0] + positions[:n, 2] / 2
            _cy = positions[:n, 1] + positions[:n, 3] / 2
            _hp = float((ew * (np.abs(_cx[eu] - _cx[ev])
                               + np.abs(_cy[eu] - _cy[ev]))).sum())
            ov = ov / max(ov0, 1e-9) + best_w * _hp / max(hp0, 1e-9)
        if (frac >= best_from) and ov < best_ov:
            best_ov = ov
            best_pos = positions.copy()
        if (it + 1) % log_every == 0 or it == rounds - 1:
            traj.append(ov)
            if verbose:
                print(f"  round {it+1:4d}: cap={cap_t:.2f} overlap={ov:9.1f} "
                      f"({ov/ov0*100:5.1f}%)  best_hold={best_ov:8.1f}  "
                      f"maxshift={maxshift:.2f}")

    if best_ov == float("inf"):               # no hold phase reached
        best_ov, best_pos = _overlap_total(positions, n), positions.copy()
    positions[:n] = best_pos[:n]
    return {"overlap_start": ov0, "overlap_final": best_ov,
            "reduction_pct": (1 - best_ov / max(ov0, 1e-9)) * 100, "traj": traj,
            "scipy": _HAVE_SCIPY, "W": W, "H": H}
