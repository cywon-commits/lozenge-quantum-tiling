"""Stage B: re-rank enumeration candidates by thermodynamic-limit LSWT (Bloch) and build the T=0 hull.

    python rank_bloch.py [--tol 0.002] [--per 30] [--nk 24] cand_*.json

1. For each imbalance m keep classes with e_model <= min_m(e_model) + tol (at most --per of them).
2. Bloch LSWT energy E_inf (S=1/2) for each kept class (cached in bloch_cache.json).
3. (Optionally merge with an old reference list, --ref) and build the lower convex hull
   of e(m). Hull vertices = crystals of the field staircase; slopes = transition fields (h = de/dm).
Outputs: rank_result.json (all ranked candidates, hull, steps, new vertices) and a printed table.
"""
import json, sys, os, glob, numpy as np
from periodic import Cell
from bloch_lswt import energy

args = sys.argv[1:]
def opt(name, default, typ):
    if name in args:
        i = args.index(name); v = typ(args[i + 1]); del args[i:i + 2]; return v
    return default
TOL = opt('--tol', 0.002, float); PER = opt('--per', 30, int); NK = opt('--nk', 24, int)
REF = opt('--ref', '', str)       # optional: include an old reference list (e.g. ../ref/bloch_lswt.json)
files = args or sorted(glob.glob('cand_*.json'))

cache = json.load(open('bloch_cache.json')) if os.path.exists('bloch_cache.json') else {}
cands = []
for f in files:
    R = json.load(open(f)); byD = {}
    for r in R: byD.setdefault(round(r['m'], 8), []).append(r)
    for k, L in byD.items():
        L.sort(key=lambda r: r['e_model']); e0 = L[0]['e_model']
        cands += [r for r in L if r['e_model'] <= e0 + TOL][:PER]
print(f"{len(cands)} candidates from {len(files)} files", flush=True)
for i, r in enumerate(cands):
    key = f"{r['cell']}|{r['key']}|{round(r['m'],8)}"
    if key not in cache:
        E, m = energy(Cell(*r['cell']), [tuple(x) for x in r['tiling']], nk=NK)
        cache[key] = E
        if i % 20 == 0: json.dump(cache, open('bloch_cache.json', 'w'))
    r['e_inf'] = cache[key]
json.dump(cache, open('bloch_cache.json', 'w'))

ref = json.load(open(REF)) if REF else []
pts = [dict(m=r['m'], e=r['e_inf'], cell=r['cell'], hist=r['hist'], src='ref36') for r in ref]
pts += [dict(m=r['m'], e=r['e_inf'], cell=r['cell'], hist=r['hist'], src=os.path.basename(f), key=r['key'])
        for r in cands]
best = {}
for p in pts:                      # a new candidate replaces the reference only if lower by > 2e-6 J (k-grid noise)
    k = round(p['m'], 8)
    thr = 2e-6 if (k in best and best[k]['src'] == 'ref36' and p['src'] != 'ref36') else 1e-9
    if k not in best or p['e'] < best[k]['e'] - thr: best[k] = p
P = sorted(best.values(), key=lambda p: p['m'])
esq = next(p['e'] for p in P if p['m'] == 0); edi = next(p['e'] for p in P if abs(p['m'] - 1 / 6) < 1e-9)
lin = lambda m: esq + (edi - esq) * 6 * m
# lower convex hull (monotone chain) on m in [0, 1/6]
H = []
for p in P:
    if p['m'] > 1 / 6 + 1e-9: continue
    while len(H) >= 2:
        (a, b) = H[-2], H[-1]
        if (b['e'] - a['e']) * (p['m'] - a['m']) >= (p['e'] - a['e']) * (b['m'] - a['m']) - 1e-12: H.pop()
        else: break
    H.append(p)
steps = [dict(h=(b['e'] - a['e']) / (b['m'] - a['m']), frm=6 * a['m'], to=6 * b['m']) for a, b in zip(H, H[1:])]
print("\nM/Md    dev_LSWT(1e-3 J)  cell        hist                       source")
for p in P:
    on = '*' if any(p is h for h in H) else ' '
    print(f"{6*p['m']:.4f} {on} {1e3*(p['e']-lin(p['m'])):+8.3f}      {str(p['cell']):11s} {str(p['hist']):26s} {p['src']}")
print("\nhull steps (h [J], M/Md from -> to):")
for s in steps: print(f"  h={s['h']:.4f}  {s['frm']:.4f} -> {s['to']:.4f}")
new = [p for p in H if p['src'] != 'ref36' and 1e-9 < p['m'] < 1 / 6 - 1e-9]
print("\nhull vertices (crystals) passed to stage C:", [(round(6 * p['m'], 4), p['cell']) for p in new] or "none")
print("(make_qmc.py skips those identical to the already published L=36 QMC crystals)")
json.dump(dict(tol=TOL, nk=NK, points=P, hull=H, steps=steps, new_vertices=new,
               candidates=[{k: v for k, v in r.items() if k != 'tiling'} | dict(tiling=r['tiling']) for r in cands]),
          open('rank_result.json', 'w'))
