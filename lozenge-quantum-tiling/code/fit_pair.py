"""Test of the local coordination model: (i) residuals by tiling class, (ii) does adding nearest-neighbour
(z_i, z_j) bond-type terms reduce the residual?  (iii) out-of-sample test on periodic crystals (Bloch LSWT)."""
import numpy as np, json, glob
from tilt import strips
from tiling_moves import site_edges, flippable, flip, degrees
from lswt import lswt_energy
from periodic import Cell
from bloch_lswt import cell_bonds, energy, signs_cell
EPS = np.array([-0.8544, -0.6582, -0.4423, -0.2061]); CM = 0.0304
PAIRS = [(a, b) for a in range(3, 7) for b in range(a, 7)]
def feats(z, bonds, m, N):
    site = [np.mean(z == k) for k in range(3, 7)]
    pc = {p: 0 for p in PAIRS}
    for i, j in bonds:
        a, b = sorted((int(z[i]), int(z[j]))); pc[(a, b)] += 1
    return site, [pc[p] / N for p in PAIRS], m
rng = np.random.default_rng(0); rows = []
for L, iv, lab in ((18, [], 'dice'), (18, [[0, 6]], 'strip6'), (18, [[0, 12]], 'strip12'), (18, [[0, 18]], 'square'), (18, [[0, 6], [9, 6]], 's6d3'),
                   (24, [], 'dice'), (24, [[0, 12]], 'strip12'), (24, [[0, 24]], 'square')):
    for nflip in (0, 5, 20, 60, 200, 600):
        for rep in range(3 if nflip else 1):
            T, ok = strips(L, iv); SE = site_edges(T)
            for k in range(nflip):
                cand = [v for v in rng.choice(T.N, 40, replace=False) if flippable(T, SE, v)]
                if cand: flip(T, SE, cand[0])
            E, S = lswt_energy(T.N, T.bonds()); z = degrees(T)
            s, p, m = feats(z, T.bonds(), S / T.N, T.N)
            rows.append(dict(L=L, cls=lab, nflip=nflip, e=E / T.N, site=s, pair=p, m=m))
print('training tilings', len(rows), flush=True)
X1 = np.array([r['site'] + [r['m']] for r in rows]); Y = np.array([r['e'] for r in rows])
X2 = np.array([r['site'] + [r['m']] + r['pair'] for r in rows])
def fit(X, Y):
    c, *_ = np.linalg.lstsq(X, Y, rcond=None); return c, X @ c - Y
c1, r1 = fit(X1, Y); c2, r2 = fit(X2, Y)
def loo(X, Y):
    e = []
    for i in range(len(Y)):
        k = np.arange(len(Y)) != i; c, *_ = np.linalg.lstsq(X[k], Y[k], rcond=None); e.append(X[i] @ c - Y[i])
    return np.sqrt(np.mean(np.square(e)))
out = dict(n=len(Y), spread=float(Y.std()), site_coef=c1.tolist(), rms_site=float(np.sqrt(np.mean(r1**2))), loo_site=float(loo(X1, Y)),
           rms_pair=float(np.sqrt(np.mean(r2**2))), loo_pair=float(loo(X2, Y)), max_site=float(np.abs(r1).max()))
cls = {}
for r, res in zip(rows, r1): cls.setdefault(f"{r['cls']}_L{r['L']}", []).append(res); cls.setdefault(f"nflip{r['nflip']}", []).append(res)
out['by_class'] = {k: dict(n=len(v), mean=float(np.mean(v)), rms=float(np.sqrt(np.mean(np.square(v))))) for k, v in cls.items()}
# out-of-sample: periodic crystals, Bloch LSWT (thermodynamic limit)
B = json.load(open('paper/bloch_lswt.json')); test = []
O = json.load(open('periodic_all_lswt.json'))
for f in glob.glob('big_*_s*.json'): O += json.load(open(f))
cand = {}
for r in O: cand.setdefault(round(r['m'], 5), []).append(r)
for k, L_ in cand.items():
    for r in sorted(L_, key=lambda r: r['e_model'])[:8]:
        C = Cell(*r['cell']); Bd = cell_bonds(C, r['tiling']); z = np.zeros(C.N, int)
        for a, b, _, _ in Bd: z[a] += 1; z[b] += 1
        E, m = energy(C, r['tiling'], nk=12)
        s, p, _ = feats(z, [(a, b) for a, b, _, _ in Bd], m, C.N)
        test.append(dict(cell=r['cell'], m=m, hist=r['hist'], e=E, pred_site=float(np.r_[s, m] @ c1), pred_pair=float(np.r_[s, m, p] @ c2)))
res_t1 = np.array([t['pred_site'] - t['e'] for t in test]); res_t2 = np.array([t['pred_pair'] - t['e'] for t in test])
out['test_n'] = len(test); out['test_rms_site'] = float(np.sqrt(np.mean(res_t1**2))); out['test_rms_pair'] = float(np.sqrt(np.mean(res_t2**2)))
# same-histogram spread among crystals
g = {}
for t in test: g.setdefault((round(t['m'], 5), json.dumps(t['hist'], sort_keys=True)), set()).add(round(t['e'], 7))
out['same_hist_spread'] = {f"{k[0]}|{k[1]}": float(max(v) - min(v)) for k, v in g.items() if len(v) > 1}
# rank stability: per m, is the lowest-model candidate also lowest in LSWT?  Spearman rho
from scipy.stats import spearmanr
rk = {}
for k in sorted(set(round(t['m'], 5) for t in test)):
    T_ = [t for t in test if round(t['m'], 5) == k]
    if len(T_) < 3: continue
    rho = spearmanr([t['pred_pair'] for t in T_], [t['e'] for t in T_]).correlation
    gapbest = sorted(t['e'] for t in T_)
    rk[str(k)] = dict(n=len(T_), rho_pair=None if np.isnan(rho) else float(rho), e_min=gapbest[0], e_spread=gapbest[-1] - gapbest[0])
out['rank'] = rk; out['test'] = test
json.dump(out, open('paper/model_test.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ('test',)}, indent=1))
