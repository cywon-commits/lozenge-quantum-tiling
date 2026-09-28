import json, glob, numpy as np
from periodic import EPS, C_M
from hull_lswt import lower_hull
esq = EPS[4]; ed = (EPS[6] + 2 * EPS[3]) / 3 + C_M / 6
lin = lambda m: esq + (ed - esq) * 6 * m
R = []
for f in glob.glob('big_*.json'): R += json.load(open(f))
old = json.load(open('periodic_all_lswt.json'))
best = {}
for r in R + old:
    k = round(r['m'], 5)
    if k not in best or r['e_model'] < best[k]['e_model']: best[k] = r
ms = np.array(sorted(best)); e = np.array([best[k]['e_model'] for k in ms])
H = lower_hull(ms, e); onh = set(ms[H])
print("m      M/Md   dev_model   cell   hist   onhull  new?")
oldm = set(round(r['m'], 5) for r in old)
for k, ee in zip(ms, e):
    r = best[k]; print(f"{k:.5f} {6*k:.3f} {ee-lin(k):+.5f} {r['cell']} {r['hist']} {'H' if k in onh else ' '} {'' if k in oldm else 'NEW'}")
