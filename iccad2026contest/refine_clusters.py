#!/usr/bin/env python3
"""
CLUSTER CLOSURE -- slide split cluster members into EXACT contact.

A cluster scores as satisfied only when its members form one connected group,
and the evaluator decides that with a Shapely union:

    unary_union(group_polys).geom_type == 'MultiPolygon'  ->  violation

which means the tolerance for "touching" is exactly ZERO.  Two blocks 1e-12
apart are two polygons.  This is the fact every earlier attempt here got wrong:
the packer, and MOSEK far more so, leave contacts a hair open, and a hair is
enough.  Even the legalizer's own output loses ~31 contacts that way.

So closure has to be arithmetic, not geometric.  Moving a block "by the gap"
does not work -- `x + gap` lands within an ulp of the target, and an ulp is a
MultiPolygon.  The block is ASSIGNED the neighbour's edge:

    P[hi, axis] = P[lo, axis] + P[lo, size]

Shapely then builds the lower block's far edge from the very same expression,
so the two coordinates are the same double and the union is a single polygon.
That is why the upper block is always the one that moves; pushing the lower one
instead needs `P[hi, axis] - P[lo, size]`, which is not exact in reverse, so it
is only taken when it happens to come out exact and skipped otherwise.

Constraints respected while sliding:
  * preplaced blocks never move -- no degrees of freedom;
  * a boundary block slides only ALONG its edge, never off it;
  * a move is rejected unless the destination is clear of every other block;
  * nothing leaves the bounding box, so the frame area does not change;
  * no shape is touched, so areas, aspect ratios and MIB groups are unaffected.
"""
import os as _os
import numpy as np

TOL = 1e-6


def _components(P, mem):
    """Cluster members grouped exactly as the evaluator groups them."""
    try:
        from shapely.geometry import box
        from shapely.ops import unary_union
    except ImportError:
        return _components_fallback(P, mem)
    polys = {i: box(P[i, 0], P[i, 1], P[i, 0] + P[i, 2], P[i, 1] + P[i, 3])
             for i in mem}
    u = unary_union(list(polys.values()))
    parts = list(u.geoms) if u.geom_type == "MultiPolygon" else [u]
    out = [[] for _ in parts]
    for i in mem:
        c = polys[i].representative_point()
        for k, part in enumerate(parts):
            if part.contains(c) or part.intersects(c):
                out[k].append(i)
                break
    return [c for c in out if c]


def _components_fallback(P, mem):
    """Union-find on exact edge coincidence, for a run without Shapely."""
    par = list(range(len(mem)))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for u in range(len(mem)):
        for v in range(u + 1, len(mem)):
            i, j = mem[u], mem[v]
            ox = min(P[i, 0] + P[i, 2], P[j, 0] + P[j, 2]) - max(P[i, 0], P[j, 0])
            oy = min(P[i, 1] + P[i, 3], P[j, 1] + P[j, 3]) - max(P[i, 1], P[j, 1])
            if (ox > 0.0 and oy >= 0.0) or (oy > 0.0 and ox >= 0.0):
                ra, rb = find(u), find(v)
                if ra != rb:
                    par[ra] = rb
    groups = {}
    for u in range(len(mem)):
        groups.setdefault(find(u), []).append(mem[u])
    return list(groups.values())


def compact(P, n, prep, bnd, axis, origin):
    """Pull every free block back along `axis` until it rests on something.

    Each block is ASSIGNED the far edge of whatever stops it -- `P[m, axis] +
    P[m, size]`, the very expression Shapely uses to build that block's edge --
    so the contact is exact rather than merely close, and the union of two
    touching blocks is one polygon.  Adding a gap-sized delta instead lands an
    ulp short, which is still two polygons.

    Blocks are processed in order along the axis, so a block always settles
    against neighbours that have already settled.  Moving a block back onto its
    blockers cannot create an overlap, so the layout stays legal without any
    checking.  Preplaced blocks are immovable, and a boundary block is frozen on
    the axis its edge lives on -- pulling an R block left would take it off the
    right edge it is required to touch.
    """
    size = axis + 2
    pa = 1 - axis
    psize = pa + 2
    for k in sorted(range(n), key=lambda i: P[i, axis]):
        if prep[k]:
            continue
        if axis == 0 and (bnd[k] & 3):
            continue
        if axis == 1 and (bnd[k] & 12):
            continue
        lo = origin
        for m in range(n):
            if m == k:
                continue
            ov = min(P[k, pa] + P[k, psize], P[m, pa] + P[m, psize]) \
                - max(P[k, pa], P[m, pa])
            if ov <= TOL:
                continue                      # nothing in this lane
            e = P[m, axis] + P[m, size]
            if e <= P[k, axis] + TOL and e > lo:
                lo = e
        P[k, axis] = lo


def refine_clusters(P, n, cons, is_preplaced, rounds=40, areas=None):
    """Slide split cluster members into exact contact, in place.  Returns the
    number of blocks moved.

    One round merges at most ONE pair per cluster, because every move changes
    what is reachable for the next one -- a block that had nowhere to go often
    does once its neighbour has shifted.  So `rounds` has to exceed the longest
    merge chain, not the number of clusters: at 6 rounds case 99 stalled at 5
    violations and a further pass took it to 1.  The loop exits as soon as a
    round moves nothing, so a generous cap costs nothing.
    """
    clu = cons[:n, 3].astype(int)
    bnd = cons[:n, 4].astype(int)
    prep = np.array([bool(is_preplaced[i]) for i in range(n)])
    x0 = P[:n, 0].min(); y0 = P[:n, 1].min()
    x1 = (P[:n, 0] + P[:n, 2]).max(); y1 = (P[:n, 1] + P[:n, 3]).max()

    def clear(k, nx, ny, ignore=()):
        for m in range(n):
            if m == k or m in ignore:
                continue
            ox = min(nx + P[k, 2], P[m, 0] + P[m, 2]) - max(nx, P[m, 0])
            oy = min(ny + P[k, 3], P[m, 1] + P[m, 3]) - max(ny, P[m, 1])
            if ox > TOL and oy > TOL:
                return False
        return True

    def movable(k, axis):
        if prep[k]:
            return False
        c = bnd[k]
        if axis == 0 and (c & 1 or c & 2):
            return False
        if axis == 1 and (c & 4 or c & 8):
            return False
        return True

    # OFF by default.  Compacting the whole layout before chasing individual
    # pairs looks like it should help -- many peers would land in contact for
    # free, and the swept-up space would give the closure loop room to work.
    # Measured, it loses twice over.  Wirelength suffers (0.318 -> 0.328),
    # because pulling everything toward the origin throws away the positions
    # the wirelength term chose.  And violations RISE (0.0499 -> 0.0584): a
    # single fixed direction closes the gap on a block's left and opens the one
    # on its right, so every block dragged along abandons a contact behind it.
    # Total Score 1.3040 -> 1.3731 with the polish, 1.3281 -> 1.3448 without.
    import os as _os
    if _os.environ.get("PACK_COMPACT", "0") == "1":
        compact(P, n, prep, bnd, 0, x0)
        compact(P, n, prep, bnd, 1, y0)

    # ---- rank the pieces: everything moves toward the ANCHOR ----
    # Which block travels is not a free choice.  Picking by position, as an
    # earlier version did, happily dragged a block out of a large piece to
    # patch a small one -- one contact gained, another lost, count unchanged.
    # Measured, that churned 1640 blocks across the suite for no net change.
    #
    # The rule that fixes it is the one a disjoint-set merge already implies:
    # the smaller piece joins the larger, never the reverse, and a piece
    # containing something that CANNOT move -- a preplaced block, or a boundary
    # block frozen on the axis in question -- is the anchor that everything
    # else comes to.
    def _rank(comp, axis):
        pinned = sum(1 for k in comp if prep[k] or (
            bnd[k] & (3 if axis == 0 else 12)))
        return (pinned, len(comp))

    def _internal(comp):
        """the contacts inside a piece, as (lo, hi, axis) -- what has to be
        rebuilt after the piece is translated"""
        out = []
        for u in range(len(comp)):
            for v in range(len(comp)):
                if u == v:
                    continue
                a, b = comp[u], comp[v]
                for ax in (0, 1):
                    pa, ps = 1 - ax, 3 - ax
                    ov = min(P[a, pa] + P[a, ps], P[b, pa] + P[b, ps]) \
                        - max(P[a, pa], P[b, pa])
                    if ov > TOL and P[a, ax] + P[a, ax + 2] == P[b, ax]:
                        out.append((a, b, ax))
        return out

    def _translate(comp, axis, delta, anchor, target_coord):
        """Move a whole piece rigidly, then rebuild its own contacts exactly.

        Translating by a delta is not enough: `x + delta` and `(x + w) + delta`
        do not stay equal doubles, so every contact inside the piece would open
        by an ulp -- and an ulp is a split.  So the piece is walked from the
        block that was placed exactly, and each neighbour is ASSIGNED its
        partner's edge again.
        """
        links = _internal(comp)
        keep = {k: (P[k, 0], P[k, 1]) for k in comp}
        for k in comp:
            P[k, axis] += delta
        P[anchor, axis] = target_coord
        seen = {anchor}
        changed = True
        while changed:
            changed = False
            for a, b, ax in links:
                if a in seen and b not in seen:
                    P[b, ax] = P[a, ax] + P[a, ax + 2]
                    seen.add(b); changed = True
                elif b in seen and a not in seen:
                    P[a, ax] = P[b, ax] - P[a, ax + 2]
                    seen.add(a); changed = True
        for k in comp:                              # nothing may collide
            if not clear(k, P[k, 0], P[k, 1], ignore=comp) \
               or P[k, 0] < x0 - TOL or P[k, 1] < y0 - TOL \
               or P[k, 0] + P[k, 2] > x1 + TOL or P[k, 1] + P[k, 3] > y1 + TOL:
                for q, v in keep.items():
                    P[q, 0], P[q, 1] = v
                return False
        return True

    moves = 0
    for _ in range(rounds):
        moved_any = False
        for g in sorted(set(int(v) for v in clu) - {0}):
            mem = [i for i in range(n) if clu[i] == g]
            if len(mem) < 2:
                continue
            comps = _components(P, mem)
            if len(comps) < 2:
                continue
            # merge the least-anchored piece into the best-anchored one
            comps.sort(key=lambda c: (_rank(c, 0)[0] + _rank(c, 1)[0], len(c)),
                       reverse=True)
            main, rest = comps[0], comps[1:]
            done = False
            for comp in rest:
                best = None
                for b in comp:
                    for t in main:
                        for ax in (0, 1):
                            pa, ps = 1 - ax, 3 - ax
                            ov = min(P[b, pa] + P[b, ps], P[t, pa] + P[t, ps]) \
                                - max(P[b, pa], P[t, pa])
                            if ov <= TOL:
                                continue
                            if P[b, ax] >= P[t, ax]:
                                tgt = P[t, ax] + P[t, ax + 2]   # b sits after t
                            else:
                                tgt = P[t, ax] - P[b, ax + 2]   # b sits before t
                            d = tgt - P[b, ax]
                            if best is None or abs(d) < abs(best[0]):
                                best = (d, b, ax, tgt)
                if best is None:
                    continue
                d, b, ax, tgt = best
                # the whole piece must be free to travel on this axis
                if any(prep[k] or (bnd[k] & (3 if ax == 0 else 12))
                       for k in comp):
                    continue
                n_before = len(_components(P, mem))
                if _translate(comp, ax, d, b, tgt):
                    if len(_components(P, mem)) < n_before:
                        moves += len(comp)
                        moved_any = True
                        done = True
                        break
                    # put it back: the merge did not actually reduce the count
                    _translate(comp, ax, -d, b, tgt - d)
            if done:
                continue

            # FALLBACK: shift a single block instead.
            # Moving whole pieces is the principled merge, but it needs the
            # whole piece to be free, and it cannot help when the two pieces
            # are only a rounding error apart -- which is most of what the SOCP
            # leaves behind (gaps around 1e-9).  For those, nudging one block
            # onto its neighbour's edge is enough, and cheap.
            best = None
            for ci in range(len(comps)):
                for cj in range(ci + 1, len(comps)):
                    for i in comps[ci]:
                        for j in comps[cj]:
                            ox = min(P[i, 0] + P[i, 2], P[j, 0] + P[j, 2]) \
                                - max(P[i, 0], P[j, 0])
                            oy = min(P[i, 1] + P[i, 3], P[j, 1] + P[j, 3]) \
                                - max(P[i, 1], P[j, 1])
                            if ox > TOL and oy <= 0.0:
                                axis, gap = 1, -oy
                            elif oy > TOL and ox <= 0.0:
                                axis, gap = 0, -ox
                            else:
                                continue
                            if best is None or gap < best[0]:
                                best = (gap, i, j, axis)
            if best is None:
                continue
            gap, i, j, axis = best
            size = axis + 2
            lo, hi = (i, j) if P[i, axis] < P[j, axis] else (j, i)
            n_before = len(comps)

            def _try(k, nxy):
                if not (clear(k, nxy[0], nxy[1]) and nxy[0] >= x0 - TOL
                        and nxy[1] >= y0 - TOL
                        and nxy[0] + P[k, 2] <= x1 + TOL
                        and nxy[1] + P[k, 3] <= y1 + TOL):
                    return False
                keep = (P[k, 0], P[k, 1])
                P[k, 0], P[k, 1] = nxy
                if len(_components(P, mem)) <= n_before:
                    return True
                P[k, 0], P[k, 1] = keep       # strictly worse -- put it back
                return False

            if movable(hi, axis):
                nxy = [P[hi, 0], P[hi, 1]]
                nxy[axis] = P[lo, axis] + P[lo, size]
                if _try(hi, nxy):
                    moves += 1
                    moved_any = True
                    continue
            if movable(lo, axis):
                nxy = [P[lo, 0], P[lo, 1]]
                nxy[axis] = P[hi, axis] - P[lo, size]
                if nxy[axis] + P[lo, size] == P[hi, axis] and _try(lo, nxy):
                    moves += 1
                    moved_any = True

        if not moved_any:
            break

    moves += _close_by_growth(P, n, cons, is_preplaced, areas)
    return moves


def _close_by_growth(P, n, cons, is_preplaced, areas):
    """Last resort: pay for the last sliver of a gap out of the AREA BUDGET.

    Sliding is the right first move and it is what the passes above do, but a
    block that cannot slide -- boxed in on the far side, or pinned by a peer it
    is already touching -- leaves its cluster split over a gap of a fraction of
    a unit.  Those are not topology failures, they are arithmetic ones, and the
    contest hands us the arithmetic to fix them: a soft block's realised w*h
    only has to be within 1% of its target area, and every block in our layout
    sits at essentially exactly its target.  So the facing edge can simply be
    extended across the gap.

    Half the 1% is spent at most, so the margin survives any later rounding,
    and the extension is verified against every other block and the frame --
    growth that would overlap or leave the outline is refused.
    """
    if areas is None or _os.environ.get("PACK_CLOSE_GROW", "1") != "1":
        return 0
    BUD = float(_os.environ.get("PACK_CLOSE_BUDGET", "0.005"))
    MAXG = float(_os.environ.get("PACK_CLOSE_MAX", "2.0"))
    clu = cons[:n, 3].astype(int)
    prep = np.array([bool(is_preplaced[i]) for i in range(n)])
    fixed = np.array([int(cons[i, 0]) > 0 for i in range(n)])
    mib = np.array([int(cons[i, 2]) for i in range(n)])
    soft = ~prep & ~fixed & (mib == 0)
    tgt = np.asarray(areas, dtype=float)[:n]
    x1 = float((P[:n, 0] + P[:n, 2]).max())
    y1 = float((P[:n, 1] + P[:n, 3]).max())
    grown = 0

    def free(k, box):
        """Is `box` clear of every block except k?"""
        bx0, by0, bx1, by1 = box
        for m in range(n):
            if m == k:
                continue
            if min(bx1, P[m, 0] + P[m, 2]) - max(bx0, P[m, 0]) > TOL and \
               min(by1, P[m, 1] + P[m, 3]) - max(by0, P[m, 1]) > TOL:
                return False
        return True

    for g in sorted(set(clu.tolist()) - {0}):
        mem = [i for i in range(n) if clu[i] == g]
        if len(mem) < 2:
            continue
        for _ in range(len(mem)):
            comps = _components(P, mem)
            if len(comps) < 2:
                break
            # closest AXIALLY aligned cross-component pair
            best = None
            for ca in range(len(comps)):
                for cb in range(ca + 1, len(comps)):
                    for i in comps[ca]:
                        for j in comps[cb]:
                            for ax, sz in ((0, 2), (1, 3)):
                                ot, os_ = 1 - ax, 3 - (sz - 2)
                                ov = min(P[i, ot] + P[i, os_], P[j, ot] + P[j, os_]) \
                                    - max(P[i, ot], P[j, ot])
                                if ov <= TOL:
                                    continue          # no shared face to grow into
                                lo, hi = (i, j) if P[i, ax] <= P[j, ax] else (j, i)
                                d = P[hi, ax] - (P[lo, ax] + P[lo, sz])
                                # ANY positive gap, however small.  TOL here
                                # would have skipped the whole point: a quarter
                                # of the surviving violations are gaps of 1e-10
                                # to 1e-14 -- pairs the geometry meant to abut,
                                # landed a fraction of a picometre apart by the
                                # solver, and Shapely splits at exactly zero.
                                if d <= 0.0 or d > MAXG:
                                    continue
                                if best is None or d < best[0]:
                                    best = (d, lo, hi, ax, sz)
            if best is None:
                break
            d, lo, hi, ax, sz = best
            done = False
            # Grow the LOWER block's far edge forward, else pull the UPPER
            # block's near edge back.  Either way only the facing edge moves.
            for k, forward in ((lo, True), (hi, False)):
                if not soft[k]:
                    continue
                room = (1.0 + BUD) * tgt[k] / max(
                    P[k, 3] if ax == 0 else P[k, 2], TOL) \
                    - (P[k, 2] if ax == 0 else P[k, 3])
                if room < d - TOL:
                    continue                          # budget will not reach
                # ASSIGN the shared edge rather than adding the gap to it.
                # x + w + d and the neighbour's x are different doubles for the
                # same intended number, and Shapely's zero tolerance can tell:
                # the whole repair is worthless unless the two edges come out
                # bit-identical.
                box = list((P[k, 0], P[k, 1],
                            P[k, 0] + P[k, 2], P[k, 1] + P[k, 3]))
                if forward:
                    box[ax + 2] = P[hi, ax]
                else:
                    box[ax] = P[lo, ax] + P[lo, sz]
                if box[0] < -TOL or box[1] < -TOL or \
                   box[2] > x1 + TOL or box[3] > y1 + TOL:
                    continue
                if not free(k, box):
                    continue
                P[k, ax] = box[ax]
                P[k, sz] = box[ax + 2] - box[ax]
                grown += 1
                done = True
                break
            if not done:
                break
    return grown
