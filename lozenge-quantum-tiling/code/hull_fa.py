import json, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from tilt import strips
from lswt import lswt_energy
from hull_lswt import lower_hull


def collect(L):
    pts = []
    f = {18: None, 24: 'fa24.log', 30: 'fa30.log'}[L]
    if f:
        for l in open(f):
            if l.startswith('{'):
                r = json.loads(l); pts.append((r['m'], r['e'], f"anneal h={r['h']} sec={r['iv']}", r['pkl']))
    else:
        for fn in ('ann18.log', 'ann18b.log'):
            for l in open(fn):
                if l.startswith('{'):
                    r = json.loads(l); ivs = '_'.join(str(w) for a, w in r['iv']) or 'dice'
                    pts.append((r['best']['m'], r['best']['e'], f"anneal h={r['h']} sec={ivs}", f"ann_18_{ivs}_{r['h']}_{r['seed']}_{r['pre']}.pkl"))
    for iv, lab in (([], 'dice'), ([[0, L]], 'square')):
        T, ok = strips(L, iv); E, S = lswt_energy(T.N, T.bonds()); pts.append((S / T.N, E / T.N, lab, None))
    return pts


res = {}
for L in (18, 24, 30):
    try:
        pts = collect(L)
    except FileNotFoundError:
        continue
    m = np.array([p[0] for p in pts]); e = np.array([p[1] for p in pts])
    H = lower_hull(m, e)
    H = [i for k, i in enumerate(H) if k == 0 or m[i] > m[H[k - 1]] + 1e-9]
    res[L] = dict(m=m, e=e, H=H, pts=pts)
    esq, ed = e[[i for i, p in enumerate(pts) if p[2] == 'square'][0]], e[[i for i, p in enumerate(pts) if p[2] == 'dice'][0]]
    print(f"L={L}: {len(pts)} structures; square e={esq:.5f}, dice e={ed:.5f}")
    for a, b in zip(H[:-1], H[1:]):
        print(f"   m={m[a]:.4f} e={e[a]:.5f} dev={e[a]-(esq+(ed-esq)*m[a]*6):+.5f}  {pts[a][2]:35s} -> h={(e[b]-e[a])/(m[b]-m[a]):.3f}")
    print(f"   m={m[H[-1]]:.4f} e={e[H[-1]]:.5f}  {pts[H[-1]][2]}")
    json.dump([dict(m=float(m[i]), e=float(e[i]), label=pts[i][2], pkl=pts[i][3]) for i in H], open(f'hull_fa_{L}.json', 'w'), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for L, c in ((18, 'C0'), (24, 'C1'), (30, 'C2')):
    if L not in res: continue
    r = res[L]; m, e, H = r['m'], r['e'], r['H']
    esq = e[[i for i, p in enumerate(r['pts']) if p[2] == 'square'][0]]; ed = e[[i for i, p in enumerate(r['pts']) if p[2] == 'dice'][0]]
    dev = e - (esq + (ed - esq) * 6 * m)
    ax[0].plot(m, dev, '.', color=c, alpha=.35); ax[0].plot(m[H], dev[H], 'o-', color=c, label=f'L={L} hull')
    hs = np.linspace(0, 0.3, 1201); G = e[H][None, :] - hs[:, None] * m[H][None, :]
    ax[1].plot(hs, m[H][G.argmin(1)], color=c, label=f'L={L}')
ax[0].axhline(0, color='k', lw=.5); ax[0].set_xlabel('m'); ax[0].set_ylabel('e_LSWT − linear(square→dice)')
ax[0].set_title('LSWT: all annealed structures and lower hull'); ax[0].legend(fontsize=8)
ax[1].set_xlabel('h / J (LSWT units)'); ax[1].set_ylabel('selected m'); ax[1].legend(fontsize=8); ax[1].set_title('T=0 staircase from LSWT hull')
plt.tight_layout(); plt.savefig('figures/lswt_hull_L18_24_30.png', dpi=140)
