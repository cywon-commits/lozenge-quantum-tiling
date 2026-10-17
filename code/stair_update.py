"""Stage D: rebuild the QMC magnetisation staircase with the new crystals and bootstrap it.

    cd data; python ../code/stair_update.py   # reads qbig_*.log, qc.log and qnew/stair/qnew_*_s*.log

Same conventions as big_fig.py / stair_boot.py of the manuscript:
  e0 = e + h (m + m_L)/2, deviations from the square-dice line, T=0 hull with chi(square)=0.065,
  bootstrap of the staircase with the QMC error bars (4000 samples).
Checks that every run stayed in its Lieb sector (|m - m_L| < 0.002); runs that did not are dropped.
Outputs stair_new.json, stair_new.png and a printed table.
"""
import json, glob, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

REF = '.'                     # run from data/
NEW = 'qnew/stair/'
ESQ, ED = -0.669437, -0.63786; lin = lambda m: ESQ + (ED - ESQ) * 6 * m   # square: TD limit (Sandvik 1997); dice: QMC L=24

def load(f):
    out = []
    for l in open(f):
        if l.startswith('{'):
            r = json.loads(l); N = r['L'] ** 2; mL = r['S_lieb'] / N
            ok = abs(r['m'] - mL) < 0.002
            if not ok: print(f"  [drop] {f}: m={r['m']:.4f} left the Lieb sector m_L={mL:.4f}")
            else: out.append((mL, r['e'] + r['h'] * (r['m'] + mL) / 2, r['e_err']))
    return out

pts = {0.0: (0.0, ESQ, 5e-6, 'square'), round(1 / 6, 5): (1 / 6, ED, 3e-4, 'dice')}
for m, e, er in load(os.path.join(REF, 'qc.log')):
    k = round(m, 5)
    if k not in pts or e < pts[k][1]: pts[k] = (m, e, er, 'qc L<=24')
for f in sorted(glob.glob(os.path.join(REF, 'qbig_*.log'))):
    v = load(f)
    if v: pts[round(v[0][0], 5)] = (v[0][0], np.mean([x[1] for x in v]), max(x[2] for x in v), os.path.basename(f))
new = {}
for tag in sorted({f.rsplit('_s', 1)[0] for f in glob.glob(NEW + 'qnew_*_s*.log')}):
    v = sum((load(f) for f in glob.glob(tag + '_s*.log')), [])
    if not v: continue
    m = v[0][0]; e = np.mean([x[1] for x in v]); er = max(np.std([x[1] for x in v]) / np.sqrt(len(v)), max(x[2] for x in v) / np.sqrt(len(v)))
    tag = os.path.basename(tag); new[tag] = (m, e, er)
    k = round(m, 5)
    if k not in pts or e < pts[k][1]: pts[k] = (m, e, er, tag)
P = sorted(pts.values()); m = np.array([p[0] for p in P]); e = np.array([p[1] for p in P]); er = np.array([p[2] for p in P])
print("M/Md     dev (1e-3 J)   source")
for p in P: print(f"{6*p[0]:.4f}  {1e3*(p[1]-lin(p[0])):+7.2f}({1e3*p[2]:.2f})   {p[3]}")
print("\nnew QMC runs:"); [print(f"  {t}: M/Md={6*v[0]:.4f} dev={1e3*(v[1]-lin(v[0])):+.2f}({1e3*v[2]:.2f})") for t, v in new.items()]
hs = np.linspace(0, 0.35, 3501); chi = np.where(m == 0, 0.065, 0.0)
def stair(es):
    G = es[None, :] - hs[:, None] * m[None, :] - 0.5 * chi[None, :] * hs[:, None] ** 2; return G.argmin(1)
sel = stair(e)
steps = [(round(hs[i], 3), round(6 * m[sel[i - 1]], 4), round(6 * m[sel[i]], 4)) for i in range(1, len(hs)) if sel[i] != sel[i - 1]]
print("\nsteps (h, M/Md from, to):", steps)
rng = np.random.default_rng(1); NS = 4000; lab = [round(6 * x, 4) for x in m]; pres = {l: 0 for l in lab}
for s in range(NS):
    for k in set(stair(e + rng.normal(0, er))): pres[lab[k]] += 1
pres = {str(k): v / NS for k, v in pres.items()}
print("bootstrap presence:", {k: v for k, v in pres.items() if v > 0.01})
json.dump(dict(points=[list(p) for p in P], steps=steps, presence=pres, new={k: list(v) for k, v in new.items()}),
          open(NEW + 'stair_esqTD.json', 'w'), indent=1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].errorbar(6 * m, 1e3 * (e - lin(m)), 1e3 * er, fmt='o-', color='C3', ms=6)
for t, v in new.items(): ax[0].plot(6 * v[0], 1e3 * (v[1] - lin(v[0])), 's', mfc='none', mec='k', ms=12)
ax[0].axhline(0, color='k', lw=.5); ax[0].set_xlabel('M / M_dice', fontsize=13); ax[0].set_ylabel('δe (10⁻³ J)', fontsize=13)
ax[0].set_title('QMC L=36; □ = new cells', fontsize=12)
ax[1].plot(hs, 6 * m[sel], 'k-', lw=2); ax[1].set_xlabel('h / J', fontsize=13); ax[1].set_ylabel('M / M_dice', fontsize=13)
ax[1].set_title('T=0 staircase', fontsize=12)
plt.tight_layout(); plt.savefig(NEW + 'stair_esqTD.png', dpi=130)
