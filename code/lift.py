"""Lift a lozenge tiling to its stepped surface in Z^3 (height representation) and measure the discrete
Gaussian curvature K_i = 2*pi - (pi/2) z_i (angle deficit: every lozenge is a unit square face).
Triangular-lattice directions -> cube steps:  a1 -> +e_x,  a2-a1 -> +e_y,  -a2 -> +e_z  (p_x + p_y + p_z = 0)."""
import numpy as np, pickle, json
from collections import deque
from dice_string import Tiling
from tiling_moves import degrees
from lswt import signs

STEP = {(1, 0): np.array([1, 0, 0]), (-1, 1): np.array([0, 1, 0]), (0, -1): np.array([0, 0, 1])}
for k in list(STEP): STEP[(-k[0], -k[1])] = -STEP[k]
DIRS6 = list(STEP)


def kept(T, p, d):
    L = T.L; a = T.s(p[0] % L, p[1] % L); b = T.s((p[0] + d[0]) % L, (p[1] + d[1]) % L)
    return tuple(sorted((a, b))) not in T.removed


def lift(T, n0, n1):
    """3D coordinates of all patch points (i,j) with n0<=i,j<n1, by BFS over kept bonds."""
    H = {}; start = (n0, n0); H[start] = np.zeros(3, int); q = deque([start])
    while q:
        p = q.popleft()
        for d in DIRS6:
            r = (p[0] + d[0], p[1] + d[1])
            if not (n0 <= r[0] < n1 and n0 <= r[1] < n1) or not kept(T, p, d): continue
            h = H[p] + STEP[d]
            if r in H: assert (H[r] == h).all(), 'height not single valued'
            else: H[r] = h; q.append(r)
    return H


def faces(T, H):
    """lozenges as 3D quadrilaterals: an up triangle (i,j),(i+1,j),(i,j+1) plus the neighbour across its removed edge."""
    F = []
    for (i, j) in H:
        tri = [(i, j), (i + 1, j), (i, j + 1)]
        if not all(t in H for t in tri): continue
        for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            A, B, Cc = tri[a], tri[b], tri[c]
            d = (B[0] - A[0], B[1] - A[1])
            if not kept(T, A, d):                        # removed edge A-B: fourth vertex = A + B - C
                D = (A[0] + B[0] - Cc[0], A[1] + B[1] - Cc[1])
                if D in H: F.append([H[Cc], H[A], H[D], H[B]])
    return F


def curvature_check(T):
    z = degrees(T); K = 2 * np.pi - 0.5 * np.pi * z
    eta = signs(T.N, T.bonds()); nA, nB = (eta == 1).sum(), (eta == -1).sum()
    if nA > nB: eta = -eta; nA, nB = nB, nA            # eta=+1 on the minority sublattice A
    return dict(sumK=float(K.sum()), NB_minus_NA=int(nB - nA), staggered=float(-(eta * K).sum() / (2 * np.pi)))


def orientation(T):
    """fractions of the three lozenge orientations -> macroscopic surface normal, angle from (111)."""
    L = T.L; cnt = np.zeros(3)
    for e in T.removed:
        d = ((T.ij[e[1]] - T.ij[e[0]] + L // 2) % L - L // 2)
        o = 0 if d[1] == 0 else (1 if d[0] == 0 else 2)
        cnt[o] += 1
    f = cnt / cnt.sum()
    th = np.degrees(np.arccos(np.clip(f.sum() / (np.linalg.norm(f) * np.sqrt(3)), -1, 1)))
    return f.tolist(), float(th)


if __name__ == '__main__':
    S = pickle.load(open('paper/structs_L36.pkl', 'rb')); S['pleated'] = pickle.load(open('paper/pleated_L36.pkl', 'rb')); out = {}
    for k in ('square', 'pleated', 'M6', 'M3', 'M2', 'M23', 'dice'):
        T = Tiling(36); T.removed = S[k]
        c = curvature_check(T); f, th = orientation(T)
        H = lift(T, 0, 14)                               # checks single-valuedness on a patch
        z = degrees(T); u, n = np.unique(z, return_counts=True)
        Kfrac = {int(90 * (4 - a)): float(b / T.N) for a, b in zip(u, n)}
        out[k] = dict(c, orient=f, tilt_from_111=th, K_hist_deg=Kfrac, mean_abs_K_deg=float(np.mean(np.abs(90 * (4 - z)))))
        print(k, out[k], flush=True)
    json.dump(out, open('paper/curvature.json', 'w'), indent=1)
