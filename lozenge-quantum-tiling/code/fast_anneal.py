"""Two-stage search over lozenge tilings at larger L.
Stage 1: simulated annealing with the local coordination model fitted to LSWT
         E_model = sum_i eps(z_i) + c_m * S_Lieb      (fast, O(1) per flip)
Stage 2: short zero/low-temperature LSWT refinement of the stage-1 result.
Cost function in both stages: F = E - h * S_Lieb.
"""
import numpy as np, sys, json, time, pickle
from tilt import strips
from tiling_moves import site_edges
from lswt import lswt_energy, signs

EPS = {3: -0.8544, 4: -0.6582, 5: -0.4423, 6: -0.2061}   # per-site LSWT fit (S=1/2)
C_M = 0.0304                                            # per unit Lieb spin


class State:
    def __init__(self, T):
        self.T = T; self.SE = site_edges(T)
        self.nbr = [[e[1] if e[0] == v else e[0] for e in self.SE[v]] for v in range(T.N)]
        self.rem = [[e in T.removed for e in self.SE[v]] for v in range(T.N)]
        self.z = np.array([6 - sum(r) for r in self.rem])
        c = signs(T.N, T.bonds()); assert c is not None
        self.c = c.copy(); self.sumc = int(c.sum())
        self.E = sum(EPS[int(k)] for k in self.z) + C_M * abs(self.sumc) / 2

    def flippable(self, v):
        r = self.rem[v]
        return sum(r) == 3 and all(r[k] != r[(k + 1) % 6] for k in range(6))

    def dE(self, v):
        # after flip: v keeps z=3; each neighbour gains/loses one bond
        d = 0.0
        for k, u in enumerate(self.nbr[v]):
            zu = self.z[u]; nz = zu + (1 if self.rem[v][k] else -1)
            d += EPS[int(nz)] - EPS[int(zu)] if 3 <= nz <= 6 else 1e9
        ns = self.sumc - 2 * self.c[v]
        d += C_M * (abs(ns) - abs(self.sumc)) / 2
        return d, ns

    def flip(self, v, ns):
        T = self.T
        for k, e in enumerate(self.SE[v]):
            u = self.nbr[v][k]
            if self.rem[v][k]:
                T.removed.remove(e); self.z[u] += 1
            else:
                T.removed.add(e); self.z[u] -= 1
            self.rem[v][k] = not self.rem[v][k]
            # update the neighbour's view of this edge
            ku = self.SE[u].index(e); self.rem[u][ku] = self.rem[v][k]
        self.c[v] *= -1; self.sumc = ns

    def S(self):
        return abs(self.sumc) / 2


def stage1(T, h, nsweeps, seed, T0=0.08, T1=0.001):
    rng = np.random.default_rng(seed); st = State(T); N = T.N
    F = st.E - h * st.S(); best = (F, set(T.removed))
    nsteps = nsweeps * N
    for step in range(nsteps):
        tau = T0 * (T1 / T0) ** (step / nsteps)
        v = int(rng.integers(N))
        if not st.flippable(v): continue
        d, ns = st.dE(v)
        dF = d - h * (abs(ns) - abs(st.sumc)) / 2
        if dF <= 0 or rng.random() < np.exp(-dF / tau):
            st.flip(v, ns); st.E += d; F += dF
            if F < best[0] - 1e-12: best = (F, set(T.removed))
    T.removed = best[1]
    return T


def stage2(T, h, nsteps, seed, tau=0.002):
    rng = np.random.default_rng(seed + 999)
    SE = site_edges(T)
    E, S = lswt_energy(T.N, T.bonds()); F = E - h * S
    from tiling_moves import flippable, flip
    for step in range(nsteps):
        for t in range(50):
            v = int(rng.integers(T.N))
            if flippable(T, SE, v): break
        flip(T, SE, v)
        E2, S2 = lswt_energy(T.N, T.bonds()); F2 = E2 - h * S2
        if F2 <= F or rng.random() < np.exp(-(F2 - F) / tau):
            F, E, S = F2, E2, S2
        else:
            flip(T, SE, v)
    return T, E, S


if __name__ == "__main__":
    L = int(sys.argv[1]); iv = json.loads(sys.argv[2]); h = float(sys.argv[3])
    nsw = int(sys.argv[4]); n2 = int(sys.argv[5]); seed = int(sys.argv[6])
    t = time.time()
    T, ok = strips(L, iv); assert ok
    T = stage1(T, h, nsw, seed)
    E1, S1 = lswt_energy(T.N, T.bonds())
    T, E, S = stage2(T, h, n2, seed)
    tag = f"fa_{L}_{'_'.join(str(w) for a, w in iv) or 'dice'}_{h}_{seed}"
    pickle.dump(set(T.removed), open(tag + ".pkl", "wb"))
    z = np.bincount(T.bonds().ravel(), minlength=T.N)
    print(json.dumps(dict(L=L, iv=iv, h=h, seed=seed, stage1=dict(e=E1 / T.N, m=S1 / T.N), e=E / T.N, m=S / T.N, F=(E - h * S) / T.N,
                          deg={int(k): int(v) for k, v in zip(*np.unique(z, return_counts=True))}, pkl=tag + ".pkl", t=time.time() - t)), flush=True)
