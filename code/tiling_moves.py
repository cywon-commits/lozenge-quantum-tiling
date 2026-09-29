"""Local hexagon flips on lozenge tilings (dice_string.Tiling objects).
A vertex v is flippable when its 6 triangular-lattice edges alternate removed / present.
Flip = swap removed <-> present among those 6 edges (keeps a perfect lozenge tiling,
keeps bipartiteness and the global tilt sector)."""
import numpy as np


def site_edges(T):
    se = [[] for _ in range(T.N)]
    for e in T.edge_tris:
        se[e[0]].append(e); se[e[1]].append(e)
    # order the 6 edges of each site by angle
    out = []
    for v in range(T.N):
        es = se[v]
        ang = []
        for e in es:
            o = e[1] if e[0] == v else e[0]
            d = T.wrap(T.pos[o] - T.pos[v])
            ang.append(np.arctan2(d[1], d[0]))
        out.append([es[k] for k in np.argsort(ang)])
    return out


def flippable(T, SE, v):
    r = [e in T.removed for e in SE[v]]
    return sum(r) == 3 and all(r[k] != r[(k + 1) % 6] for k in range(6))


def flip(T, SE, v):
    for e in SE[v]:
        if e in T.removed:
            T.removed.remove(e)
        else:
            T.removed.add(e)


def degrees(T):
    b = T.bonds()
    return np.bincount(b.ravel(), minlength=T.N)
