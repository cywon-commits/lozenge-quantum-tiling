"""Wall (dice|square interface) energy from all strip configurations.
delta = e0 - [(1-f) e_sq + f e_dice],  f = dice fraction = 6 m  (m = Lieb moment/site, only for configs
without domain-reversing width-3 square strips);  sigma = delta * N / (n_walls * L)   [J per row length]
"""
import json, glob, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
E_SQ, E_SQ_ERR = -0.6693, 0.0005
E_D = -0.6379
rows = []
for f in ['q3.log', 'q4.log'] + glob.glob('r_*.log') + glob.glob('field*.log') + glob.glob('multi*.log') + ['sc24.log', 'sc18.log', 'lam.log']:
    try:
        for l in open(f):
            if l.startswith('{'):
                r = json.loads(l)
                if r.get('kind') == 'tilt': rows.append(r)
    except FileNotFoundError:
        pass
cfg = {}
for r in rows:
    L = r['L']; N = L * L; iv = [tuple(x) for x in r['iv']]
    widths = [w for a, w in iv]
    if any(w % 6 for w in widths) or sum(widths) in (0, L):   # skip domain-reversing (odd-unit) strips and pure phases
        continue
    mL = r['S_lieb'] / N; e0 = r['e'] + r['h'] * (r['m'] + mL) / 2
    key = (L, tuple(iv))
    cfg.setdefault(key, dict(L=L, iv=iv, m=mL, nw=2 * len(iv), e0=[]))['e0'].append(e0)
out = []
for k, c in cfg.items():
    L = c['L']; N = L * L; f = 6 * c['m']
    e0 = np.mean(c['e0'])
    d = e0 - ((1 - f) * E_SQ + f * E_D)
    sig = d * N / (c['nw'] * L)
    sig_sys = (1 - f) * E_SQ_ERR * N / (c['nw'] * L)
    out.append(dict(L=L, iv=c['iv'], f=f, nw=c['nw'], e0=e0, delta=d, sigma=sig, sig_sys=sig_sys))
out.sort(key=lambda x: (x['nw'], x['L'], x['f']))
print(" L   config                         f     walls  e0        sigma (J/a)")
for o in out:
    print(f"{o['L']:3d}  {str(o['iv']):30s} {o['f']:.3f}  {o['nw']}   {o['e0']:.5f}  {o['sigma']:+.4f} (sys ±{o['sig_sys']:.4f})")
s2 = [o['sigma'] for o in out if o['nw'] == 2]; s4 = [o['sigma'] for o in out if o['nw'] > 2]
print("mean sigma: 2 walls %.4f ± %.4f (n=%d);  >2 walls %.4f ± %.4f (n=%d)" % (
    np.mean(s2), np.std(s2) / np.sqrt(len(s2)), len(s2), np.mean(s4), np.std(s4) / np.sqrt(len(s4)), len(s4)))
json.dump(out, open('walls.json', 'w'), default=list)

# ---- figure
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
for o in out:
    c = 'C0' if o['nw'] == 2 else 'C3'
    ax[0].errorbar(o['f'] + 0.004 * (o['L'] - 24) / 6, o['sigma'], o['sig_sys'], marker='o' if o['nw'] == 2 else '*',
                   ms=6 if o['nw'] == 2 else 11, color=c, ls='none', alpha=.8)
ax[0].plot([], [], 'o', color='C0', label='single square strip (2 walls)')
ax[0].plot([], [], '*', color='C3', ms=11, label='several strips / dice lamellae (4–6 walls)')
ax[0].axhline(0, color='k', lw=.6); ax[0].set_xlabel('dice fraction f = 6m'); ax[0].set_ylabel('wall energy σ (J per unit length)')
ax[0].set_title('dice | square wall energy is negative (L = 18–30)'); ax[0].legend(fontsize=8)
# hull model: non-interacting walls with sigma, packing limits dice>=3 rows, square>=6 rows
sig = np.mean(s2 + s4)
f = np.linspace(0, 1, 401)
n = np.where(f <= 1 / 3, 2 * f / 3, 2 * (1 - f) / 6)            # walls per row
e = (1 - f) * E_SQ + f * E_D + sig * n
ax[1].plot(f / 6, e - ((1 - f) * E_SQ + f * E_D), 'k-', label=f'model: max wall density, σ={sig:.4f}')
for o in out:
    ax[1].plot(o['f'] / 6, o['delta'], 'o' if o['nw'] == 2 else '*', color='C0' if o['nw'] == 2 else 'C3', ms=6 if o['nw'] == 2 else 11)
ax[1].axvline(1 / 18, color='gray', ls=':'); ax[1].text(1 / 18 + .002, ax[1].get_ylim()[0] * 0.9 if ax[1].get_ylim()[0] < 0 else -0.003, 's6d3\n(M = M_dice/3)', fontsize=8)
ax[1].set_xlabel('m'); ax[1].set_ylabel('e0 − linear interpolation (J/site)'); ax[1].legend(fontsize=8)
ax[1].set_title('best structure at each m: dice lamellae (3 rows) in square (6 rows)')
plt.tight_layout(); plt.savefig('figures/wall_energy.png', dpi=140)
