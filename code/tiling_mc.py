"""Finite-temperature Monte Carlo of lozenge tilings (local hexagon flips, fixed tilt sector)
with the QMC-fitted coordination model  F = sum_i f(z_i) + (c_m - h) * S_Lieb   (S=1/2, J=1).
Tracks energy, Lieb moment and overlap with a reference (ground-state) tiling."""
import numpy as np, json, pickle, sys
from numba import njit
from dice_string import Tiling
from tiling_moves import site_edges
from lswt import signs


def build(T):
    SE = site_edges(T)
    eid = {e: k for k, e in enumerate(T.edge_tris)}
    N = T.N
    nbr = np.zeros((N, 6), np.int64); ed = np.zeros((N, 6), np.int64)
    for v in range(N):
        for k, e in enumerate(SE[v]):
            nbr[v, k] = e[1] if e[0] == v else e[0]; ed[v, k] = eid[e]
    rem = np.zeros(len(eid), np.bool_)
    for e in T.removed: rem[eid[e]] = True
    return nbr, ed, rem, eid


@njit(cache=True)
def mc(nbr, ed, rem, z, c, f, cmh, beta, nsweep, nmeas_every, ref, seed):
    np.random.seed(seed)
    N = z.shape[0]; sumc = 0
    for v in range(N): sumc += c[v]
    F = 0.0
    for v in range(N): F += f[z[v]]
    F += cmh * abs(sumc) / 2
    nm = nsweep // nmeas_every
    Es = np.zeros(nm); Ms = np.zeros(nm); Qs = np.zeros(nm); k = 0
    for sw in range(nsweep):
        for t in range(N):
            v = np.random.randint(N)
            # flippable: alternating removed pattern
            s = 0; ok = True
            for a in range(6):
                r = rem[ed[v, a]]
                s += r
                if r == rem[ed[v, (a + 1) % 6]]: ok = False
            if not ok or s != 3: continue
            dF = 0.0
            for a in range(6):
                u = nbr[v, a]; zu = z[u]
                nz = zu + 1 if rem[ed[v, a]] else zu - 1
                if nz < 3 or nz > 6: dF = 1e9; break
                dF += f[nz] - f[zu]
            ns = sumc - 2 * c[v]
            dF += cmh * (abs(ns) - abs(sumc)) / 2
            if dF <= 0 or np.random.random() < np.exp(-beta * dF):
                for a in range(6):
                    u = nbr[v, a]
                    if rem[ed[v, a]]:
                        rem[ed[v, a]] = False; z[u] += 1
                    else:
                        rem[ed[v, a]] = True; z[u] -= 1
                c[v] = -c[v]; sumc = ns; F += dF
        if (sw + 1) % nmeas_every == 0:
            q = 0.0
            for e in range(rem.shape[0]):
                if rem[e] and ref[e]: q += 1
            Es[k] = F / N; Ms[k] = abs(sumc) / 2 / N; Qs[k] = q / (N); k += 1
    return Es, Ms, Qs


def run(Tinit, Tref, h, temps, nsweep=4000, every=10, seed=0, coef=None):
    sm = json.load(open('scaled_model.json')); b = sm['b']
    EPS = {3: -0.8544, 4: -0.6582, 5: -0.4423, 6: -0.2061}
    f = np.zeros(8)
    for k, v in EPS.items(): f[k] = b * v
    cm = b * 0.0304; const = sm['a']
    nbr, ed, rem, eid = build(Tinit)
    ref = np.zeros(len(eid), np.bool_)
    for e in Tref.removed: ref[eid[e]] = True
    z = np.bincount(Tinit.bonds().ravel(), minlength=Tinit.N).astype(np.int64)
    c = signs(Tinit.N, Tinit.bonds()).astype(np.int64)
    out = []
    for i, Tm in enumerate(temps):
        Es, Ms, Qs = mc(nbr, ed, rem, z, c, f, cm - h, 1.0 / Tm, nsweep, every, ref, seed + i)
        h2 = len(Es) // 2
        out.append(dict(T=Tm, e=float(Es[h2:].mean() + const + (h * Ms[h2:].mean())), m=float(Ms[h2:].mean()), q=float(Qs[h2:].mean()),
                        cv=float(Es[h2:].var() * Tinit.N / Tm ** 2)))
    return out
