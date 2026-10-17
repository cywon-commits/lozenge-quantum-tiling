"""CE-scored T=0 hull in the hub-free window (0 <= M <= M_dice/2) over ALL enumerated classes,
then exact Bloch LSWT for the CE-best structures of each m (check)."""
import json, glob, sys, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from features import features
from bloch_lswt import energy
from periodic import Cell
F = json.load(open('fit.json'))['+ pairs up to r2<=7']; cols = F['cols']; w = np.array(F['w'])
best = {}
for f in sorted(glob.glob('cand_*.json')):
    for r in json.load(open(f)):
        if '6' in r['hist']: continue
        e = w[0] + features(r['cell'], r['tiling'])[cols] @ w[1:]
        k = round(r['m'], 8)
        best.setdefault(k, []).append((e, r))
res = []
for k in sorted(best):
    L = sorted(best[k], key=lambda t: t[0])[:3]
    for e, r in L:
        E, _ = energy(Cell(*r['cell']), [tuple(x) for x in r['tiling']], nk=16)
        res.append(dict(m=k, e_ce=e, e_inf=E, cell=r['cell'], hist=r['hist'], N=len(r['tiling'])))
esq = min(x['e_inf'] for x in res if x['m'] == 0); ehf = min(x['e_inf'] for x in res if abs(x['m'] - 1 / 12) < 1e-8)
ED = -0.63315; lin = lambda m: esq + (ED - esq) * 6 * m
P = {}
for x in res:
    if x['m'] not in P or x['e_inf'] < P[x['m']]['e_inf']: P[x['m']] = x
P = sorted(P.values(), key=lambda x: x['m'])
H = []
for p in P:
    while len(H) >= 2 and (H[-1]['e_inf'] - H[-2]['e_inf']) * (p['m'] - H[-2]['m']) >= (p['e_inf'] - H[-2]['e_inf']) * (H[-1]['m'] - H[-2]['m']): H.pop()
    H.append(p)
print("M/Md    N   dev_LSWT(1e-3)  CE-LSWT(1e-6)  hull  cell")
for p in P:
    print(f"{6*p['m']:.4f} {p['N']:3d}  {1e3*(p['e_inf']-lin(p['m'])):+8.3f}      {1e6*(p['e_ce']-p['e_inf']):+6.1f}      {'*' if p in H else ' '}   {p['cell']} {p['hist']}")
print("hull fields:", [(round(6 * a['m'], 3), round(6 * b['m'], 3), round((b['e_inf'] - a['e_inf']) / (b['m'] - a['m']), 4)) for a, b in zip(H, H[1:])])
json.dump(dict(points=P, hull=H), open('ce_hull.json', 'w'), indent=1)
