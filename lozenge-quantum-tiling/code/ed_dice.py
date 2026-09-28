"""
Exact diagonalization of the S=1/2 Heisenberg AF on small lozenge-tiling tori
(dice reference tiling + pairs of frustrated triangles).
Matrix-free Lanczos (ARPACK) in fixed-Sz sectors, numba-parallel matvec.
"""
import numpy as np
from numba import njit, prange
from scipy.sparse.linalg import LinearOperator, eigsh
from math import comb

A1 = np.array([1.0, 0.0]); A2 = np.array([0.5, np.sqrt(3) / 2])


class TorusTiling:
    """triangular-lattice torus with integer (i,j) torus vectors T1,T2 (both on the hub superlattice)"""
    def __init__(self, T1, T2):
        self.T = np.array([T1, T2]).T.astype(float)   # columns
        assert (T1[0] - T1[1]) % 3 == 0 and (T2[0] - T2[1]) % 3 == 0
        self.N = int(round(abs(np.linalg.det(self.T))))
        reps = {}
        R = 3 * max(map(abs, [*T1, *T2]))
        for i in range(-R, R):
            for j in range(-R, R):
                c = self.canon(i, j)
                if c not in reps:
                    reps[c] = len(reps)
        assert len(reps) == self.N, (len(reps), self.N)
        self.idx = reps
        self.ij = np.zeros((self.N, 2), int)
        for (i, j), n in reps.items():
            self.ij[n] = (i, j)
        self.color = (self.ij[:, 0] - self.ij[:, 1]) % 3
        self.pos = self.ij[:, :1] * A1 + self.ij[:, 1:] * A2
        self.tri_sites, cent = [], []
        for (i, j) in [tuple(x) for x in self.ij]:
            for tri in ([(i, j), (i + 1, j), (i, j + 1)], [(i + 1, j), (i, j + 1), (i + 1, j + 1)]):
                self.tri_sites.append([self.s(a, b) for a, b in tri])
                cent.append(np.mean([a * A1 + b * A2 for a, b in tri], 0))
        self.tri_cent = np.array(cent)
        self.edge_tris, self.tri_edges = {}, []
        for t, sts in enumerate(self.tri_sites):
            es = []
            for a in range(3):
                e = tuple(sorted((sts[a], sts[(a + 1) % 3])))
                es.append(e); self.edge_tris.setdefault(e, []).append(t)
            self.tri_edges.append(es)
        assert all(len(v) == 2 for v in self.edge_tris.values()), "torus too small (multi-edges)"
        self.removed = set(e for e in self.edge_tris if {self.color[e[0]], self.color[e[1]]} == {1, 2})

    def canon(self, i, j):
        f = np.linalg.solve(self.T, np.array([i, j], float))
        f = f - np.floor(f + 1e-9)
        v = self.T @ f
        return (int(round(v[0])), int(round(v[1])))

    def s(self, i, j):
        return self.idx[self.canon(i, j)]

    def wrap(self, d):
        M = np.array([A1, A2]).T @ self.T
        best = None
        f0 = np.linalg.solve(M, d)
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                v = M @ (f0 - np.round(f0) + np.array([a, b]))
                if best is None or np.linalg.norm(v) < np.linalg.norm(best):
                    best = v
        return best

    def partner(self, t):
        rem = [e for e in self.tri_edges[t] if e in self.removed]
        if not rem:
            return None, None
        a, b = self.edge_tris[rem[0]]
        return (b if a == t else a), rem[0]

    def monomers(self):
        return [t for t in range(len(self.tri_sites)) if self.partner(t)[0] is None]

    def create_pair(self, e):
        self.removed.remove(e)
        return list(self.edge_tris[e])

    def hop(self, t, e):
        """monomer t hops across its edge e (to neighbour tp, whose partner becomes monomer)"""
        a, b = self.edge_tris[e]; tp = b if a == t else a
        tpp, ep = self.partner(tp)
        self.removed.add(e); self.removed.remove(ep)
        return tpp

    def bonds(self):
        return np.array(sorted(e for e in self.edge_tris if e not in self.removed))


# ---------------------------------------------------------------- sector basis
def binom_table(N):
    Bt = np.zeros((N + 1, N + 1), np.int64)
    for n in range(N + 1):
        for k in range(n + 1):
            Bt[n, k] = comb(n, k)
    return Bt


@njit(cache=True)
def gen_states(N, k, dim):
    out = np.empty(dim, np.int64)
    v = (1 << k) - 1
    for n in range(dim):
        out[n] = v
        t = v | (v - 1)
        v = (t + 1) | (((~t & -~t) - 1) >> (_ctz(v) + 1))
    return out


@njit(cache=True)
def _ctz(v):
    c = 0
    while (v & 1) == 0:
        v >>= 1; c += 1
    return c


@njit(cache=True)
def rank(v, Bt):
    r = 0; k = 0; pos = 0
    while v:
        if v & 1:
            k += 1
            r += Bt[pos, k]
        v >>= 1; pos += 1
    return r


def rank_tables(N, Bt):
    nl = N // 2
    nh = N - nl
    low = np.zeros(1 << nl, np.int64); lowc = np.zeros(1 << nl, np.int64)
    for v in range(1 << nl):
        r = 0; k = 0
        for pos in range(nl):
            if (v >> pos) & 1:
                k += 1; r += Bt[pos, k]
        low[v] = r; lowc[v] = k
    high = np.zeros((1 << nh, nl + 1), np.int64)
    for v in range(1 << nh):
        for c in range(nl + 1):
            r = 0; k = c
            for pos in range(nh):
                if (v >> pos) & 1:
                    k += 1; r += Bt[pos + nl, k]
            high[v, c] = r
    return nl, low, lowc, high


@njit(cache=True)
def frank(v, nl, low, lowc, high):
    l = v & ((1 << nl) - 1)
    return low[l] + high[v >> nl, lowc[l]]


@njit(parallel=True, cache=True)
def matvec2(x, y, states, bonds, nl, low, lowc, high):
    dim = states.shape[0]
    nb = bonds.shape[0]
    for n in prange(dim):
        v = states[n]
        acc = 0.0
        diag = 0.0
        for b in range(nb):
            i = bonds[b, 0]; j = bonds[b, 1]
            si = (v >> i) & 1; sj = (v >> j) & 1
            if si == sj:
                diag += 0.25
            else:
                diag -= 0.25
                w = v ^ ((1 << i) | (1 << j))
                acc += 0.5 * x[frank(w, nl, low, lowc, high)]
        y[n] = acc + diag * x[n]


@njit(parallel=True, cache=True)
def matvec(x, y, states, bonds, Bt):
    dim = states.shape[0]
    nb = bonds.shape[0]
    for n in prange(dim):
        v = states[n]
        acc = 0.0
        diag = 0.0
        for b in range(nb):
            i = bonds[b, 0]; j = bonds[b, 1]
            si = (v >> i) & 1; sj = (v >> j) & 1
            if si == sj:
                diag += 0.25
            else:
                diag -= 0.25
                w = v ^ ((1 << i) | (1 << j))
                acc += 0.5 * x[rank(w, Bt)]
        y[n] = acc + diag * x[n]


@njit(parallel=True, cache=True)
def szmap(x, states, N):
    out = np.zeros(N)
    for n in range(states.shape[0]):
        p = x[n] * x[n]
        v = states[n]
        for i in range(N):
            out[i] += p * (((v >> i) & 1) - 0.5)
    return out


def lowest(bonds, N, nup, k=3):
    dim = comb(N, nup)
    Bt = binom_table(N)
    states = gen_states(N, nup, dim)
    b = np.ascontiguousarray(bonds, np.int64)
    nl, low, lowc, high = rank_tables(N, Bt)
    def mv(x):
        y = np.empty_like(x); matvec2(np.ascontiguousarray(x, np.float64), y, states, b, nl, low, lowc, high); return y
    op = LinearOperator((dim, dim), matvec=mv, dtype=np.float64)
    w, v = eigsh(op, k=k, which="SA", tol=1e-10, ncv=max(2 * k + 1, 12))
    o = np.argsort(w)
    return w[o], v[:, o], states
