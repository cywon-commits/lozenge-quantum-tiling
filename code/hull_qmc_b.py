"""(b) QMC verification of the LSWT hull structures at L=24, 30."""
import json, glob, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from hull_lswt import lower_hull

def load(f):
    out = []
    try:
        for l in open(f):
            if l.startswith('{'): out.append(json.loads(l))
    except FileNotFoundError:
        pass
    return out

ref = {  # e0 of pure phases from earlier QMC (h-corrected)
    18: {'square': -0.66920, 'dice': -0.63794},
    24: {'dice': -0.63786},
    30: {'square': -0.66917},   # L=30 square not run: L=18 and L=24 agree to 3e-5, use L=24 value
}
for r in load('qb_sq.log'):
    ref[24]['square'] = r['e'] + r['h'] * r['m'] / 2           # m_L = 0, linear response
for r in load('qb30.log'):
    if r.get('kind') == 'tilt' and r['iv'] == []:
        ref[30]['dice'] = r['e'] + r['h'] * (r['m'] + r['S_lieb'] / 900) / 2
lsw = {L: {p['pkl']: p for p in json.load(open(f'hull_fa_{L}.json')) if p['pkl']} for L in (18, 24, 30)}
res = {}
for L, f in ((18, 'qhull.log'), (24, 'qb24.log'), (30, 'qb30.log')):
    pts = []
    for r in load(f):
        if r.get('kind') != 'pkl': continue
        N = L * L; mL = r['S_lieb'] / N; e0 = r['e'] + r['h'] * (r['m'] + mL) / 2
        pts.append(dict(m=mL, e0=e0, err=max(r['e_err'], 3e-4), pkl=r['pkl'], m_meas=r['m']))
    for k, v in ref[L].items():
        pts.append(dict(m=0.0 if k == 'square' else 1 / 6, e0=v, err=4e-4, pkl=k, m_meas=None))
    res[L] = sorted(pts, key=lambda p: p['m'])

for L, pts in res.items():
    if not pts: continue
    m = np.array([p['m'] for p in pts]); e = np.array([p['e0'] for p in pts])
    print(f"L={L}: {len(pts)} points  (square ref {'yes' if 'square' in ref[L] else 'NO'}, dice ref {'yes' if 'dice' in ref[L] else 'NO'})")
    if 'square' in ref[L] and 'dice' in ref[L]:
        lin = ref[L]['square'] + (ref[L]['dice'] - ref[L]['square']) * 6 * m
        H = lower_hull(m, e)
        for p, d in zip(pts, e - lin):
            print(f"   m={p['m']:.4f}  e0={p['e0']:.5f}±{p['err']:.5f}  dev={d:+.5f}   {p['pkl']}")
        print("   hull steps:", [f"{(e[b]-e[a])/(m[b]-m[a]):.3f}" for a, b in zip(H[:-1], H[1:])])
    else:
        for p in pts: print(f"   m={p['m']:.4f}  e0={p['e0']:.5f}  {p['pkl']}")
json.dump({str(k): v for k, v in res.items()}, open('hull_qmc_b.json', 'w'), indent=1, default=float)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for L, c in ((18, 'C0'), (24, 'C1'), (30, 'C2')):
    pts = res[L]
    if not ('square' in ref[L] and 'dice' in ref[L]): continue
    m = np.array([p['m'] for p in pts]); e = np.array([p['e0'] for p in pts]); er = np.array([p['err'] for p in pts])
    lin = ref[L]['square'] + (ref[L]['dice'] - ref[L]['square']) * 6 * m
    ax[0].errorbar(m, e - lin, er, fmt='o-', color=c, label=f'QMC L={L}', ms=5)
    # LSWT dev scaled by QMC/LSWT (square-dice gap ratio) for comparison
    H = lower_hull(m, e); hs = np.linspace(0, 0.35, 1401)
    chi = np.where(m == 0, 0.065, 0.0)
    G = e[None, :] - hs[:, None] * m[None, :] - 0.5 * chi[None, :] * hs[:, None] ** 2
    ax[1].plot(hs, m[G.argmin(1)], color=c, label=f'QMC L={L}')
ax[0].axhline(0, color='k', lw=.5); ax[0].set_xlabel('m'); ax[0].set_ylabel('e0 − linear(square→dice)  (QMC, J/site)')
ax[0].set_title('QMC-evaluated hull structures'); ax[0].legend(fontsize=8)
ax[1].set_xlabel('h / J'); ax[1].set_ylabel('selected m'); ax[1].set_title('T=0 selection from QMC energies'); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig('figures/qmc_hull_b.png', dpi=140)
