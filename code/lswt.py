"""Linear spin-wave (harmonic) ground-state energy of the Heisenberg AF  H = J sum S_i.S_j
on an arbitrary bipartite graph (collinear Neel / Lieb-ferrimagnetic reference state).

Holstein-Primakoff about the two-sublattice state gives H2 = (JS/2) a^T [[D, Adj],[Adj, D]] a
(pairing form).  Frequencies  w_n = JS |mu_n|,  mu = eig( L^{1/2} P L^{1/2} ),
L = D - Adj (graph Laplacian), P = diag(+-1) sublattice signs.
E = -J S^2 N_b + (JS/2) ( sum_n |mu_n| - sum_i z_i )
"""
import numpy as np
from collections import deque


def signs(N, bonds):
    adj = [[] for _ in range(N)]
    for i, j in bonds:
        adj[i].append(j); adj[j].append(i)
    c = np.zeros(N, int); seen = np.zeros(N, bool); seen[0] = True; c[0] = 1
    q = deque([0])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True; c[v] = -c[u]; q.append(v)
            elif c[v] == c[u]:
                return None
    return c


def lswt_energy(N, bonds, S=0.5, J=1.0, return_modes=False):
    bonds = np.asarray(bonds)
    P = signs(N, bonds)
    if P is None:
        raise ValueError("graph not bipartite")
    Lap = np.zeros((N, N))
    np.add.at(Lap, (bonds[:, 0], bonds[:, 1]), -1.0)
    np.add.at(Lap, (bonds[:, 1], bonds[:, 0]), -1.0)
    z = -Lap.sum(1)
    Lap[np.arange(N), np.arange(N)] = z
    w, V = np.linalg.eigh(Lap)
    w = np.clip(w, 0, None)
    Lh = (V * np.sqrt(w)) @ V.T
    mu = np.linalg.eigvalsh(Lh @ (P[:, None] * Lh))
    E = -J * S * S * len(bonds) + 0.5 * J * S * (np.abs(mu).sum() - z.sum())
    Sl = abs(P.sum()) / 2 * (2 * S)     # Lieb total spin for spin S
    if return_modes:
        return E, Sl, mu
    return E, Sl
