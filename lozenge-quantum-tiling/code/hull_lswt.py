import json, glob, numpy as np, pickle, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from tilt import strips
from lswt import lswt_energy
from tiling_moves import degrees

pts = []   # (m, e, label, source)
for f in ['ann18.log', 'ann18b.log']:
    for l in open(f):
        if l.startswith('{'):
            r = json.loads(l)
            ivs = '_'.join(str(w) for a, w in r['iv']) or 'dice'
            pk = f"ann_18_{ivs}_{r['h']}_{r['seed']}_{r['pre']}.pkl"
            pts.append((r['best']['m'], r['best']['e'], f"anneal sector={ivs} h={r['h']}", pk))
for iv, lab in (([], 'dice'), ([[0, 18]], 'square'), ([[0, 6]], 'strip w6'), ([[0, 12]], 'strip w12'), ([[0, 6], [9, 6]], 's6d3')):
    T, ok = strips(18, iv); E, S = lswt_energy(T.N, T.bonds()); pts.append((S / T.N, E / T.N, lab, None))
m = np.array([p[0] for p in pts]); e = np.array([p[1] for p in pts])


def lower_hull(m, e):
    idx = np.lexsort((e, m)); H = []
    for i in idx:
        while len(H) >= 2:
            a, b = H[-2], H[-1]
            if (m[b] - m[a]) * (e[i] - e[a]) - (e[b] - e[a]) * (m[i] - m[a]) <= 0: H.pop()
            else: break
        H.append(i)
    # keep only the part with increasing slope from min-m to max-m (lower hull)
    return H


H = lower_hull(m, e)
# remove duplicates in m keeping lowest
print("lower hull (LSWT, L=18):")
for a, b in zip(H[:-1], H[1:]):
    print(f"  {pts[a][2]:35s} m={m[a]:.4f} e={e[a]:.5f}  -> step h = {(e[b]-e[a])/(m[b]-m[a]):.3f}")
print(f"  {pts[H[-1]][2]:35s} m={m[H[-1]]:.4f} e={e[H[-1]]:.5f}")
json.dump([dict(m=float(m[i]), e=float(e[i]), label=pts[i][2], pkl=pts[i][3]) for i in H], open('hull_lswt.json', 'w'), indent=1)

fig, ax = plt.subplots(figsize=(6.5, 4.6))
for p in pts:
    c = 'C0' if p[3] else 'C3'
    ax.plot(p[0], p[1], 'o' if p[3] else 's', color=c, ms=5 if p[3] else 8, alpha=.7)
ax.plot(m[H], e[H], 'k-', lw=1)
for i in H: ax.annotate(pts[i][2].replace('anneal ', ''), (m[i], e[i]), fontsize=6, xytext=(3, -8), textcoords='offset points')
ax.set_xlabel('m (Lieb moment per site)'); ax.set_ylabel('e_LSWT per site (J)')
ax.set_title('LSWT, L=18: annealed tilings (blue) and strip structures (red)')
plt.tight_layout(); plt.savefig('figures/lswt_hull_L18.png', dpi=140)
