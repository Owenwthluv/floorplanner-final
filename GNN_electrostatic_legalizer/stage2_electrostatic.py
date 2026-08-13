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

    ov0 = _overlap_total(positions, n)
    traj = [ov0]
    best_ov = float("inf")
    best_pos = positions.copy()

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
            q = w * h
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
        if in_hold and ov < best_ov:          # only trust the hold phase
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
