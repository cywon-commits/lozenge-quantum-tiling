"""Simulated annealing over lozenge tilings (fixed tilt sector, local hexagon flips)
with the LSWT energy  F = E_LSWT(S=1/2) - h * S_Lieb  as cost."""
import numpy as np, sys, json, time, pickle
from tilt import strips
from tiling_moves import site_edges, flippable, flip, degrees
from lswt import lswt_energy


def cost(T, h):
    E, S = lswt_energy(T.N, T.bonds())
    return E - h * S, E, S


def anneal(L, iv, h, nsteps=4000, T0=0.3, T1=0.003, seed=0, prescramble=0):
    rng = np.random.default_rng(seed)
    T, ok = strips(L, iv); assert ok
    SE = site_edges(T)
    for k in range(prescramble):
        v = rng.integers(T.N)
        if flippable(T, SE, v): flip(T, SE, v)
    F, E, S = cost(T, h)
    best = (F, E, S, set(T.removed))
    for step in range(nsteps):
        tau = T0 * (T1 / T0) ** (step / nsteps)
        for tries in range(50):
            v = rng.integers(T.N)
            if flippable(T, SE, v): break
        else:
            continue
        flip(T, SE, v)
        F2, E2, S2 = cost(T, h)
        if F2 <= F or rng.random() < np.exp(-(F2 - F) / tau):
            F, E, S = F2, E2, S2
            if F < best[0]: best = (F, E, S, set(T.removed))
        else:
            flip(T, SE, v)
    return best, T


if __name__ == "__main__":
    L = int(sys.argv[1]); iv = json.loads(sys.argv[2]); h = float(sys.argv[3]); ns = int(sys.argv[4]); seed = int(sys.argv[5]); pre = int(sys.argv[6])
    t = time.time()
    T0, ok = strips(L, iv); F0, E0, S0 = cost(T0, h)
    best, T = anneal(L, iv, h, ns, seed=seed, prescramble=pre)
    F, E, S, rem = best
    T.removed = rem
    z = degrees(T)
    r = dict(L=L, iv=iv, h=h, seed=seed, pre=pre, start=dict(F=F0 / T.N, e=E0 / T.N, m=S0 / T.N),
             best=dict(F=F / T.N, e=E / T.N, m=S / T.N), deg={int(k): int(v) for k, v in zip(*np.unique(z, return_counts=True))}, t=time.time() - t)
    print(json.dumps(r), flush=True)
    pickle.dump(rem, open(f"ann_{L}_{'_'.join(str(w) for a, w in iv) or 'dice'}_{h}_{seed}_{pre}.pkl", "wb"))
