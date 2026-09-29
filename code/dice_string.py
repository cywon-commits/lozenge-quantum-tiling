"""
Classical Heisenberg ferrimagnet on lozenge tilings of the triangular lattice
(dice lattice = reference tiling) with a pair of frustrated-triangle defects
(monomers) connected by a tiling string.

Sublattices: color c = (i - j) mod 3 ; 0 = hub (r1), 1 = rim B (r2), 2 = rim C (r3)
Dice tiling: every B-C edge of the triangular lattice is removed.
"""
import numpy as np
from scipy.optimize import minimize

A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, np.sqrt(3) / 2])


class Tiling:
    def __init__(self, L):
        assert L % 3 == 0
        self.L = L
        N = L * L
        self.N = N
        ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
        self.ij = np.stack([ii.ravel(), jj.ravel()], 1)          # site n = i*L + j
        self.color = (self.ij[:, 0] - self.ij[:, 1]) % 3
        self.pos = self.ij[:, :1] * A1 + self.ij[:, 1:] * A2

        # triangles: (kind, i, j) kind 0 = up, 1 = down
        self.tri_sites = []
        self.tri_cent = []   # centroid in unwrapped (i,j)-coords
        for i in range(L):
            for j in range(L):
                up = [(i, j), (i + 1, j), (i, j + 1)]
                dn = [(i + 1, j), (i, j + 1), (i + 1, j + 1)]
                for kind, tri in enumerate((up, dn)):
                    self.tri_sites.append([self.s(a, b) for a, b in tri])
                    c = np.mean([a * A1 + b * A2 for a, b in tri], 0)
                    self.tri_cent.append(c)
        self.tri_cent = np.array(self.tri_cent)
        self.T = len(self.tri_sites)
        # edges
        self.edge_tris = {}
        self.tri_edges = []
        for t, sts in enumerate(self.tri_sites):
            es = []
            for a in range(3):
                e = tuple(sorted((sts[a], sts[(a + 1) % 3])))
                es.append(e)
                self.edge_tris.setdefault(e, []).append(t)
            self.tri_edges.append(es)
        # dice tiling: removed = B-C edges
        self.removed = set(e for e in self.edge_tris
                           if {self.color[e[0]], self.color[e[1]]} == {1, 2})
        self.check_pairing()

    def s(self, i, j):
        return (i % self.L) * self.L + (j % self.L)

    def partner(self, t):
        """triangle paired with t (via its removed edge) or None if monomer"""
        rem = [e for e in self.tri_edges[t] if e in self.removed]
        if not rem:
            return None, None
        assert len(rem) == 1
        e = rem[0]
        a, b = self.edge_tris[e]
        return (b if a == t else a), e

    def check_pairing(self):
        mono = [t for t in range(self.T) if self.partner(t)[0] is None]
        return mono

    def create_pair(self, e):
        self.removed.remove(e)
        return list(self.edge_tris[e])

    def move(self, t, other_monomer, direction, offset=0.0):
        """move monomer t one hop maximizing displacement along direction.
        returns new monomer triangle index and displacement vector"""
        best = None
        for e in self.tri_edges[t]:
            a, b = self.edge_tris[e]
            tp = b if a == t else a
            if tp == other_monomer:
                continue
            tpp, ep = self.partner(tp)
            d = self.wrap(self.tri_cent[tpp] - self.tri_cent[t])
            fwd = d @ direction
            score = (fwd > 0.1) * 100 + fwd - 1.5 * abs(offset + d @ np.array([-direction[1], direction[0]]))
            if best is None or score > best[0]:
                best = (score, e, tp, tpp, ep, d)
        _, e, tp, tpp, ep, d = best
        self.removed.add(e)      # t and tp form a lozenge
        self.removed.remove(ep)  # tpp becomes monomer
        return tpp, d

    def wrap(self, d):
        # minimal image in oblique coords
        M = np.array([A1, A2]).T * self.L
        f = np.linalg.solve(M, d)
        f -= np.round(f)
        return M @ f

    def bonds(self):
        return np.array([e for e in self.edge_tris if e not in self.removed])


def minimize_spins(bonds, mu, J=1.0, h=0.0, K=0.0, x0=None, seed=0, tol=1e-12, maxiter=200000):
    """E = J sum mu_i mu_j S_i.S_j - h sum mu_i S_i^z - K sum (S_i^z)^2 ; |S|=1"""
    N = len(mu)
    b0, b1 = bonds[:, 0], bonds[:, 1]
    w = J * mu[b0] * mu[b1]
    rng = np.random.default_rng(seed)
    if x0 is None:
        x0 = rng.normal(size=(N, 3))
    def f(x):
        v = x.reshape(N, 3)
        nv = np.linalg.norm(v, axis=1, keepdims=True)
        s = v / nv
        dots = np.einsum("ij,ij->i", s[b0], s[b1])
        E = np.sum(w * dots) - h * np.sum(mu * s[:, 2]) - K * np.sum(s[:, 2] ** 2)
        g = np.zeros_like(s)
        np.add.at(g, b0, w[:, None] * s[b1])
        np.add.at(g, b1, w[:, None] * s[b0])
        g[:, 2] += -h * mu - 2 * K * s[:, 2]
        g = (g - np.einsum("ij,ij->i", g, s)[:, None] * s) / nv
        return E, g.ravel()
    res = minimize(f, np.asarray(x0, float).ravel(), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, maxfun=maxiter, ftol=tol, gtol=1e-9, maxcor=30))
    v = res.x.reshape(N, 3)
    return res.fun, v / np.linalg.norm(v, axis=1, keepdims=True), res
