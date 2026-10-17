"""v4: replace the M/6 and M/3 crystals by the LSWT-best structures (complete class-resolved enumeration, QMC-confirmed)
and recompute their fluctuation measures. Other structures and S(q,w) data are unchanged."""
import json, pickle, numpy as np, time
from dice_string import Tiling
from lswt import lswt_energy
from lswt_fluct import fluct
from tiling_moves import degrees
L = 36
S = pickle.load(open('paper/structs_L36.pkl', 'rb'))
S['M6'] = pickle.load(open('qnew/lswtbest_M6_L36.pkl', 'rb')); S['M3'] = pickle.load(open('qnew/lswtbest_M3_L36.pkl', 'rb'))
pickle.dump(S, open('paper/structs_L36.pkl', 'wb'))
res = json.load(open('paper/fluct_structs.json'))
Mf = np.array([[1, 0.5], [0, np.sqrt(3) / 2]]) * L
for k in ('M6', 'M3'):
    T = Tiling(L); T.removed = set(S[k]); assert not T.check_pairing()
    B = T.bonds(); z = degrees(T)
    f = np.array([np.linalg.solve(Mf, x) for x in T.pos]); reg = np.where((f[:, 1] % 1) < 0.5)[0]
    o = fluct(T.N, B, region=reg); E, Sl = lswt_energy(T.N, B)
    res[k] = dict(m=Sl / T.N, e_lswt=E / T.N, n=float(o['n'].mean()), s_site=float(o['s_site'].mean()), S_half=o['S_region'] / (2 * L),
                  n_z={int(q): float(o['n'][z == q].mean()) for q in np.unique(z)}, frac_z={int(q): float(np.mean(z == q)) for q in np.unique(z)})
    print(k, res[k]['m'], res[k]['e_lswt'], res[k]['n'], res[k]['s_site'], res[k]['S_half'], flush=True)
json.dump(res, open('paper/fluct_structs.json', 'w'), indent=1)
for k in ('square', 'M6', 'M3', 'M2', 'M23', 'dice'): print(k, round(res[k]['n'], 4), round(res[k]['s_site'], 4), round(res[k]['S_half'], 4))
