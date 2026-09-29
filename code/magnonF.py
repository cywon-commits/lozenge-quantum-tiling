"""Magnon free energy of the M/2 crystal vs thermally melted tilings (LSWT, S=1/2, L=24):
F_mag(T)/N = (T/N) sum_n ln(1 - exp(-w_n/T)).  Melted tilings: snapshots of the tiling MC above T_m."""
import numpy as np, json, pickle
from dice_string import Tiling
from periodic import Cell, replicate
from tiling_mc import build, mc
from lswt import lswt_energy, signs
from tiling_moves import degrees
R = json.load(open('periodic_all_lswt.json')); R = [r for r in R if abs(r['m_rep'] - 1/12) < 1e-4]; R.sort(key=lambda r: r['e_lswt'])
L = 24
def crystal():
    for r in R:
        try: return replicate(Cell(*r['cell']), [tuple(x) for x in r['tiling']], L)
        except Exception: pass
sm = json.load(open('scaled_model.json')); b = sm['b']
EPS = {3: -0.8544, 4: -0.6582, 5: -0.4423, 6: -0.2061}; f = np.zeros(8)
for k, v in EPS.items(): f[k] = b * v
cm = b * 0.0304; h = 0.19
def modes(T):
    E, S, mu = lswt_energy(T.N, T.bonds(), return_modes=True); w = 0.5 * np.abs(mu); return E / T.N, S / T.N, w[w > 1e-2]
def Fmag(w, N, Tt): return Tt * np.sum(np.log(1 - np.exp(-w / Tt))) / N
Tc = crystal(); Ec, mc_, wc = modes(Tc)
out = dict(crystal=dict(e=Ec, m=mc_), melted=[])
temps = [0.01, 0.02, 0.03, 0.04]
for seed, Tmelt in ((1, 0.035), (2, 0.035), (3, 0.045)):
    T0 = crystal(); nbr, ed, rem, eid = build(T0)
    z = np.bincount(T0.bonds().ravel(), minlength=T0.N).astype(np.int64); c = signs(T0.N, T0.bonds()).astype(np.int64)
    ref = rem.copy()
    mc(nbr, ed, rem, z, c, f, cm - h, 1 / Tmelt, 3000, 10, ref, seed)
    Tm = Tiling(L); Tm.removed = {e for e, k in eid.items() if rem[k]}; assert not Tm.check_pairing()
    Em, mm, wm = modes(Tm)
    row = dict(seed=seed, Tmc=Tmelt, e=Em, m=mm, hist={int(k): int(v) for k, v in zip(*np.unique(degrees(Tm), return_counts=True))})
    for Tt in temps: row[f'dF_{Tt}'] = Fmag(wm, Tm.N, Tt) - Fmag(wc, Tc.N, Tt)
    out['melted'].append(row); print(row, flush=True)
out['crystal'].update({f'F_{Tt}': Fmag(wc, Tc.N, Tt) for Tt in temps})
json.dump(out, open('paper/magnonF.json', 'w'), indent=1); print(out['crystal'])
