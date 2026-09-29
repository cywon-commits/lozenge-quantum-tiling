import numpy as np
from dice_string import Tiling
from qmc_tiling import lieb


def hop(T, mono, e):
    a, b = T.edge_tris[e]; tp = b if a == mono else a
    tpp, ep = T.partner(tp)
    T.removed.add(e); T.removed.remove(ep)
    return tpp


def lap(T, y0, tvec=None):
    """create a monomer pair near height y0, drag one monomer once around the torus along dirv, annihilate"""
    L = T.L
    A1 = np.array([1.0, 0.0]); A2 = np.array([0.5, np.sqrt(3) / 2])
    target = L * (A1 + A2) if tvec is None else tvec
    dirv = target / np.linalg.norm(target); pv = np.array([-dirv[1], dirv[0]])
    e0 = sorted(T.removed, key=lambda e: (abs(T.pos[e[0]] @ pv - y0) + abs(T.pos[e[1]] @ pv - y0), T.pos[e[0]] @ dirv))[0]
    mono, other = T.create_pair(e0); disp = np.zeros(2)
    for step in range(40 * L):
        if np.linalg.norm(disp - target) < 1.8:
            sh = [e for e in T.tri_edges[mono] if other in T.edge_tris[e]]
            if sh:
                T.removed.add(sh[0]); break
        best = None
        for e in T.tri_edges[mono]:
            a, b = T.edge_tris[e]; tp = b if a == mono else a
            if tp == other: continue
            tpp, _ = T.partner(tp); d = T.wrap(T.tri_cent[tpp] - T.tri_cent[mono]); nd = disp + d
            sc = -np.linalg.norm(target - nd) - 1.0 * abs(nd @ pv)
            if best is None or sc > best[0]: best = (sc, e, d)
        mono = hop(T, mono, best[1]); disp = disp + best[2]
    assert not T.check_pairing(), "lap not closed"
    return T


def strip_tiling(L, k, spacing=None):
    T = Tiling(L)
    for n in range(k):
        lap(T, n * spacing)
    return T


if __name__ == "__main__":
    for k in range(0, 13):
        try:
            T = strip_tiling(12, k, spacing=np.sqrt(3) / 2 * 1.0)
            deg = np.bincount(T.bonds().ravel(), minlength=T.N)
            print(k, "S", lieb(T), dict(zip(*np.unique(deg, return_counts=True))))
        except AssertionError as ex:
            print(k, "fail", ex)
