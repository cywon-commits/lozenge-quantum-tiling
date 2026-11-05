"""v8 staircase: published QMC points + crystals from the unrestricted (D6-class) enumeration, evaluated by SSE on
k x k copies of their cells (qdec_cell.py, beta J = 36). Writes paper/staircase_v8.json (run from data/)."""
import json, glob, numpy as np
ESQ, ED = -0.669437, -0.63786; lin = lambda m: ESQ + (ED - ESQ) * 6 * m
B = json.load(open('big_staircase.json')); Q = json.load(open('paper/qdec.json'))
lab = {0: 'square', 1/6: 'M6', 1/3: 'M3', 1/2: 'M2', 2/3: 'M23', 1: 'dice', 1/4: 'M14old'}
pts = {}
for m, e, er in B['points']:
    k = [v for M, v in lab.items() if abs(6 * m - M) < 1e-3][0]
    d = Q.get(k, {}); pts[k] = dict(m=m, e=e, er=er, src='paper', Exy=d.get('Exy'), Ezz=d.get('Ezz'))
runs = {}
for f in glob.glob('qdec_cells/c_*.log'):
    for l in open(f):
        if l.startswith('{'):
            r = json.loads(l)
            if 'sdl' in r['key']: continue
            runs.setdefault(r['key'], {})[(r['seed'], r['h'])] = r
name = {'new_3o47_k5': 'M347', 'x0p1111_cell_k6': 'M19', 'x0p0909_cell_k5': 'M111', 'm35_cell_k8': 'M35', 'x0p1429_cell_k5': 'M17', 'x0p2_cell_k6': 'M15', 'x0p25_cell_k6': 'M14'}
for key, rs in runs.items():
    v = list(rs.values()); m = v[0]['m_lieb']; e0 = np.array([r['E'][0] - r['Ezee'][0] for r in v])
    er = max(np.std(e0) / np.sqrt(len(v)), max(r['E'][1] for r in v) / np.sqrt(len(v)))
    pts[name[key]] = dict(m=m, e=float(e0.mean()), er=float(er), src=f'{key}, {len(v)} seeds', nseed=len(v),
                          Exy=float(np.mean([r['Exy'][0] for r in v])), Ezz=float(np.mean([r['Ezz'][0] for r in v])))
if pts['M14']['e'] > pts['M14old']['e']: pts['M14'] = pts['M14old']
del pts['M14old']
keys = sorted(pts, key=lambda k: pts[k]['m'])
m = np.array([pts[k]['m'] for k in keys]); e = np.array([pts[k]['e'] for k in keys]); er = np.array([pts[k]['er'] for k in keys])
hs = np.linspace(0, 0.35, 3501); chi = np.where(m == 0, 0.065, 0)
def stair(es): return (es[None, :] - hs[:, None] * m[None, :] - 0.5 * chi[None, :] * hs[:, None] ** 2).argmin(1)
sel = stair(e)
steps = [(float(hs[i]), keys[sel[i - 1]], keys[sel[i]]) for i in range(1, len(hs)) if sel[i] != sel[i - 1]]
rng = np.random.default_rng(1); pres = np.zeros(len(m)); edges = {k: [] for k in keys}
for _ in range(4000):
    s = stair(e + rng.normal(0, er)); u = set(s)
    for i in u:
        pres[i] += 1; edges[keys[i]].append(hs[np.argmax(s == i)])
for i, k in enumerate(keys):
    pts[k].update(dev=1e3 * (e[i] - lin(m[i])), dev_er=1e3 * er[i], presence=pres[i] / 4000,
                  edge=(float(np.mean(edges[k])), float(np.std(edges[k]))) if edges[k] else None)
out = dict(keys=keys, points=pts, steps=steps, path=[keys[i] for i in sorted(set(sel), key=list(sel).index)])
json.dump(out, open('paper/staircase_v8.json', 'w'), indent=1)
for k in keys: p = pts[k]; print(f"{k:6s} M/Md={6*p['m']:.4f} dev={p['dev']:+.2f}({p['dev_er']:.2f}) pres={p['presence']:.3f} edge={p['edge']} Exy={p['Exy']} Ezz={p['Ezz']}")
print(steps)
