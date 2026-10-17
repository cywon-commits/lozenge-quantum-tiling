"""Cluster-expansion features of a periodic hub-free tiling (defect lattice-gas description).
Defects: sites with z != 4 (z = 3 or 5). Curvature k = 4 - z = +-1; sublattice eta = +-1 (Lieb colouring).
Pair features: for every triangular-lattice distance shell up to RMAX^2 and every class
(k k', eta eta') the number of defect pairs per site (periodic lattice sum, images included).
Shell 1 is additionally split by whether the bond is present in the tiling graph."""
import numpy as np, sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from periodic import Cell, DIRS
from collections import deque

RMAX2 = 16
OFFS = [(dx, dy) for dx in range(-5, 6) for dy in range(-5, 6)
        if 0 < dx * dx + dy * dy + dx * dy <= RMAX2]
SHELLS = sorted({dx * dx + dy * dy + dx * dy for dx, dy in OFFS})     # 1,3,4,7,9,12,13,16
CLASSES = [(1, 1), (1, -1), (-1, 1), (-1, -1)]                           # (k k', eta eta')
NAMES = ['n_def', 'm'] + [f'r2={s}|kk={a:+d}|ee={b:+d}' for s in SHELLS for a, b in CLASSES] + \
        [f'nn_bonded|kk={a:+d}' for a in (1, -1)]


def analyse(cell, tiling):
    C = Cell(*cell); N = C.N; rem = set((int(a), int(b)) for a, b in tiling)
    adj = [[] for _ in range(N)]; bond = set()
    for n in range(N):
        i, j = C.site(n)
        for d, (di, dj) in enumerate(DIRS):
            if (n, d) in rem: continue
            m = C.idx(i + di, j + dj); adj[n].append(m); adj[m].append(n)
            bond.add((n, (di, dj))); bond.add((m, (-di, -dj)))
    z = np.array([len(a) for a in adj])
    eta = np.zeros(N, int); eta[0] = 1; q = deque([0])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if eta[v] == 0: eta[v] = -eta[u]; q.append(v)
    return C, z, eta, bond


def features(cell, tiling):
    C, z, eta, bond = analyse(cell, tiling); N = C.N
    k = 4 - z
    assert set(np.unique(z)) <= {3, 4, 5}
    f = np.zeros(len(NAMES))
    f[0] = np.sum(k != 0) / N
    f[1] = abs(np.sum(eta * k)) / 8 / N          # moment m = sum q / 8N  (q = eta k ... up to sign convention)
    D = np.where(k != 0)[0]
    col = {c: t for t, c in enumerate(CLASSES)}
    base = 2
    for n in D:
        i, j = C.site(n)
        for dx, dy in OFFS:
            m2 = C.idx(i + dx, j + dy)
            if k[m2] == 0: continue
            s = dx * dx + dy * dy + dx * dy
            c = col[(k[n] * k[m2], eta[n] * eta[m2])]
            f[base + 4 * SHELLS.index(s) + c] += 0.5 / N
            if s == 1 and (n, (dx, dy)) in bond:
                f[base + 4 * len(SHELLS) + (0 if k[n] * k[m2] == 1 else 1)] += 0.5 / N
    return f
