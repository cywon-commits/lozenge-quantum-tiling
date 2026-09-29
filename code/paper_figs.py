"""All figures for the manuscript, drawn at print size (single column 3.4 in, double 7.0 in)."""
import json, glob, pickle, numpy as np, matplotlib, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from dice_string import Tiling
from tilt import strips
from tiling_moves import degrees

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'legend.fontsize': 8, 'legend.frameon': False, 'lines.linewidth': 1.4, 'lines.markersize': 5,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'xtick.direction': 'in',
    'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 'font.family': 'serif',
    'mathtext.fontset': 'cm', 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'figure.dpi': 150})
C = dict(blue='#0072B2', red='#D55E00', green='#009E73', pink='#CC79A7', orange='#E69F00', sky='#56B4E9', gray='#7f7f7f')
OUT = 'paper/figs/'
W1, W2 = 3.4, 7.0
ESQ, ED = -0.66917, -0.63786                      # QMC S=1/2 e0 (square, dice)
lin = lambda m: ESQ + (ED - ESQ) * 6 * m
A1 = np.array([1.0, 0]); A2 = np.array([0.5, np.sqrt(3) / 2])
ZC = {3: '#c6dbef', 4: '#6baed6', 5: '#2171b5', 6: '#08306b'}   # one hue, light -> dark with z
ZS = {3: 2.6, 4: 3.1, 5: 3.6, 6: 4.2}
TILE = {0: '#f3e3b8', 1: '#d7e6ef', 2: '#e9dcef'}


def tag(ax, s, x=-0.02, y=1.0, **kw):
    ax.text(x, y, s, transform=ax.transAxes, ha='right', va='bottom', fontsize=10, fontweight='bold', **kw)


def draw_tiling(ax, T, win, rot=-30, zcol=True):
    L = T.L; z = degrees(T)
    R = np.array([[np.cos(np.radians(rot)), -np.sin(np.radians(rot))], [np.sin(np.radians(rot)), np.cos(np.radians(rot))]])
    Tv = [L * A1, L * A2]; shifts = [a * Tv[0] + b * Tv[1] for a in range(-2, 3) for b in range(-2, 3)]
    x0, x1, y0, y1 = win
    for e in T.removed:
        t1, t2 = T.edge_tris[e]; vs = list(dict.fromkeys(T.tri_sites[t1] + T.tri_sites[t2]))
        base = T.pos[vs[0]]; P = np.array([base + T.wrap(T.pos[v] - base) for v in vs]); cen = P.mean(0)
        P = P[np.argsort(np.arctan2(*(P - cen).T[::-1]))]
        d = ((T.ij[e[1]] - T.ij[e[0]] + L // 2) % L - L // 2)
        o = 0 if abs(d[1]) == 0 else (1 if abs(d[0]) == 0 else 2)
        for s in shifts:
            Q = (P + s) @ R.T; c = Q.mean(0)
            if x0 - 1 <= c[0] <= x1 + 1 and y0 - 1 <= c[1] <= y1 + 1:
                ax.add_patch(Polygon(Q, fc=TILE[o], ec='#666666', lw=0.45))
    for i in range(T.N):
        for s in shifts:
            p = (T.pos[i] + s) @ R.T
            if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                ax.plot(*p, 'o', ms=ZS[int(z[i])], mfc=ZC[int(z[i])], mec='k', mew=0.3, zorder=3)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect('equal'); ax.axis('off')


def zlegend(fig, y=0.0):
    h = [Line2D([], [], marker='o', ls='', ms=ZS[k] + 1, mfc=ZC[k], mec='k', mew=0.3, label=f'$z={k}$') for k in (3, 4, 5, 6)]
    fig.legend(handles=h, loc='lower center', ncol=4, bbox_to_anchor=(0.5, y), handletextpad=0.2, columnspacing=1.2)


def load_log(fs):
    out = []
    for f in fs:
        for l in open(f):
            if l.startswith('{'): out.append(json.loads(l))
    return out


# ------------------------------------------------------------------ Fig 1: tilings and coordination
def fig1():
    S = pickle.load(open('paper/structs_L36.pkl', 'rb'))
    def T36(k): T = Tiling(36); T.removed = S[k]; return T
    Tr = Tiling(18); Tr.removed = pickle.load(open('ann_18_12_0.09_22_300.pkl', 'rb'))
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2))
    win = (-0.5, 9.5, -0.5, 7.6)
    for ax, T, t, s in zip(axs, (T36('square'), Tr, T36('dice')),
                           (r'square-type: all $z=4$, $M=0$', r'generic tiling: $\langle z\rangle=4$', r'dice: $z=3,6$, $M=M_{\rm dice}$'), 'abc'):
        draw_tiling(ax, T, win); ax.set_title(f'({s}) '+t, pad=3)
    plt.subplots_adjust(wspace=0.04, bottom=0.1); zlegend(fig, y=-0.02)
    fig.savefig(OUT + 'fig1_tilings.pdf')


# ------------------------------------------------------------------ Fig 2: coordination mechanism
def fig2():
    EPS = {3: -0.8544, 4: -0.6582, 5: -0.4423, 6: -0.2061}
    fig, axs = plt.subplots(2, 1, figsize=(W1, 4.9))
    ax = axs[0]
    zs = np.array([3, 4, 5, 6]); ev = np.array([EPS[k] for k in zs])
    ax.plot(zs, ev, 'o', color=C['blue'], ms=7, zorder=3, label=r'LSWT fit $\varepsilon(z)$')
    zz = np.linspace(3, 6, 50); p = np.polyfit(zs, ev, 2); ax.plot(zz, np.polyval(p, zz), '-', color=C['blue'], lw=1)
    ax.plot([3, 6], [EPS[3], EPS[6]], '--', color=C['red'], lw=1.2, label='chord $z=3\\leftrightarrow6$ (dice)')
    ax.plot([4], [(EPS[3] * 2 + EPS[6]) / 3], 's', mfc='white', mec=C['red'], ms=7, zorder=4)
    ax.annotate('', xy=(4.0, EPS[4] + 0.004), xytext=(4.0, (2 * EPS[3] + EPS[6]) / 3 - 0.004),
                arrowprops=dict(arrowstyle='->', color='k', lw=0.8))
    ax.text(4.12, -0.655, r'$\Delta=0.020J$', fontsize=8.5, va='center')
    ax.set_xlabel('site coordination $z$'); ax.set_ylabel(r'$\varepsilon(z)$  ($J$ per site)')
    ax.set_xticks([3, 4, 5, 6]); ax.legend(loc='upper left'); tag(ax, '(a)', x=-0.2)
    ax = axs[1]
    d = np.load('degfit_m.npz'); X, Y = d['X'], d['Y']
    pred = X @ np.array([EPS[k] for k in (3, 4, 5, 6)]) + 0.0304 * d['m']
    ax.plot(pred - pred.mean(), Y - pred.mean(), 'o', ms=3.5, mfc='none', mec=C['gray'], label='LSWT, 75 tilings')
    sm = json.load(open('scaled_model.json'))
    # QMC points: pure phases, strips, periodic crystals, annealed networks
    from fit_scaled import X as XQ, Y as YQ          # recomputes the QMC training set
    ax.plot(sm['a'] + sm['b'] * XQ - pred.mean(), YQ - pred.mean(), 's', ms=4, color=C['red'], label='QMC (scaled model)')
    lo, hi = -0.012, 0.03
    ax.plot([lo, hi], [lo, hi], 'k-', lw=0.6)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r'model $\sum_i\varepsilon(z_i)/N + c_m m$ (shifted)'); ax.set_ylabel('computed $e$ (shifted, $J$)')
    ax.legend(loc='upper left'); tag(ax, '(b)', x=-0.2)
    plt.tight_layout(h_pad=1.2); fig.savefig(OUT + 'fig2_mechanism.pdf')


# ------------------------------------------------------------------ Fig 3: tilt family and walls
def fig3():
    rows = [r for r in load_log(['q1.log', 'q2.log', 'q3.log', 'q4.log'] + glob.glob('r_*.log')) if 'kind' in r]
    m_of = lambda r: r['S_lieb'] / r['L'] ** 2
    fig, axs = plt.subplots(2, 1, figsize=(W1, 4.9))
    ax = axs[0]
    for L, mk, col in ((12, 's', C['blue']), (18, 'o', C['red'])):
        tl = [r for r in rows if r['L'] == L and r['h'] == 0 and r['kind'] in ('tilt', 'square', 'dice') and r['beta'] in ((12, 24) if L == 12 else (18,))]
        dd = {}
        for r in tl: dd.setdefault(round(m_of(r), 6), []).append(r['e'])
        ms = np.array(sorted(dd)); es = np.array([np.mean(dd[m]) for m in ms])
        ax.plot(6 * ms, es, marker=mk, ls='-', color=col, label=f'strip (tilt) family, $L={L}$')
    fl = [r for r in rows if r['L'] == 12 and r['h'] == 0 and r['kind'] in ('random', 'stripe') and r['beta'] == 12]
    ax.plot([6 * m_of(r) for r in fl], [r['e'] for r in fl], 'x', color=C['gray'], ms=5, label='random hexagon flips, $L=12$')
    ax.set_ylim(-0.674, -0.632)
    ax.set_xlabel(r'$M/M_{\rm dice}$'); ax.set_ylabel(r'$e_0$ ($J$ per site), QMC $S=\frac{1}{2}$'); ax.legend(loc='upper left'); tag(ax, '(a)', x=-0.2)
    ax = axs[1]
    Wd = json.load(open('walls.json'))
    Ls = np.array([w['L'] for w in Wd]); sg = np.array([w['sigma'] for w in Wd]); ss = np.array([w['sig_sys'] for w in Wd])
    fr = np.array([w['f'] for w in Wd])
    for f, mk, col in ((1 / 3, 'o', C['blue']), (0.5, 's', C['red']), (2 / 3, '^', C['green'])):
        k = np.isclose(fr, f, atol=0.02)
        if k.any(): ax.errorbar(Ls[k] + (f - 0.5) * 1.2, sg[k], ss[k], fmt=mk, color=col, capsize=2, label=f'dice fraction {f:.2f}')
    k = ~(np.isclose(fr, 1 / 3, atol=.02) | np.isclose(fr, .5, atol=.02) | np.isclose(fr, 2 / 3, atol=.02))
    if k.any(): ax.errorbar(Ls[k], sg[k], ss[k], fmt='D', color=C['gray'], capsize=2, label='other fractions')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('torus size $L$'); ax.set_ylabel(r'wall energy $\sigma$ ($J/a$)'); ax.set_ylim(-0.022, 0.011)
    ax.legend(loc='upper center', ncol=2, columnspacing=0.8); tag(ax, '(b)', x=-0.2)
    plt.tight_layout(h_pad=1.2); fig.savefig(OUT + 'fig3_tilt_walls.pdf')


# ------------------------------------------------------------------ Fig 4: crystals
def fig4():
    S = pickle.load(open('paper/structs_L36.pkl', 'rb'))
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2))
    win = (-0.5, 11.5, -0.5, 9.1)
    ttl = {'M6': r'$M_{\rm dice}/6$  (cell $9\times4$)', 'M2': r'$M_{\rm dice}/2$ (cell $6\times4$, no hubs)',
           'M23': r'$2M_{\rm dice}/3$  (cell $9\times4$)'}
    for ax, k, s in zip(axs, ('M6', 'M2', 'M23'), 'abc'):
        T = Tiling(36); T.removed = S[k]; draw_tiling(ax, T, win); ax.set_title(f'({s}) '+ttl[k], pad=3)
    plt.subplots_adjust(wspace=0.04, bottom=0.1); zlegend(fig, y=-0.02)
    fig.savefig(OUT + 'fig4_crystals.pdf')


# ------------------------------------------------------------------ Fig 5: hull and staircase
def fig5():
    B = json.load(open('big_staircase.json')); P = np.array(B['points'])
    m, e, er = P[:, 0], P[:, 1], P[:, 2]
    A = json.load(open('hull_qmc_b.json'))
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.75))
    ax = axs[0]
    first = True
    for L, pts in A.items():
        q = [p for p in pts if p['pkl'] not in ('square', 'dice')]
        ax.errorbar([6 * p['m'] for p in q], [1e3 * (p['e0'] - lin(p['m'])) for p in q], [1e3 * p['err'] for p in q],
                    fmt='o', ms=3.5, color=C['gray'], mfc='white', capsize=0, lw=0.8, label='irregular (annealed) networks' if first else None)
        first = False
    ax.errorbar(6 * m, 1e3 * (e - lin(m)), 1e3 * er, fmt='s', color=C['red'], ms=5, capsize=2, label='periodic crystals')
    # lower hull
    from hull_lswt import lower_hull
    H = lower_hull(m, e); ax.plot(6 * m[H], 1e3 * (e[H] - lin(m[H])), '-', color=C['red'], lw=1.0, zorder=0, label='lower convex hull')
    lab = {1 / 6: (r'$\frac{1}{6}$', -0.05, 0.0, 'right'), 1 / 4: (r'$\frac{1}{4}$', -0.005, -0.62, 'center'), 1 / 3: (r'$\frac{1}{3}$', -0.05, -0.55, 'right'),
           1 / 2: (r'$\frac{1}{2}$', 0.0, -0.75, 'center'), 2 / 3: (r'$\frac{2}{3}$', 0.05, -0.2, 'left')}
    ax2 = ax.secondary_xaxis('top'); ax2.set_xticks([1/6, 1/4, 1/3, 1/2, 2/3])
    ax2.set_xticklabels([r'$\frac{1}{6}$', r'$\frac{1}{4}$', r'$\frac{1}{3}$', r'$\frac{1}{2}$', r'$\frac{2}{3}$']); ax2.tick_params(direction='in', pad=2)
    for x in (1/6, 1/4, 1/3, 1/2, 2/3): ax.axvline(x, color='#dddddd', lw=0.6, zorder=-1)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_xlabel(r'$M/M_{\rm dice}$'); ax.set_ylabel(r'$e_0-e_{\rm lin}$  ($10^{-3}J$ per site)'); ax.set_ylim(-6.4, 3.3)
    ax.legend(loc='upper center', handlelength=1.5); tag(ax, '(a)', x=-0.15)
    ax = axs[1]
    hs = np.linspace(0, 0.35, 3501); chi = np.where(m == 0, 0.065, 0.0)
    G = e[None, :] - hs[:, None] * m[None, :] - 0.5 * chi[None, :] * hs[:, None] ** 2; sel = G.argmin(1)
    ax.plot(hs, 6 * m[sel], '-', color=C['blue'], lw=1.8, label=r'QMC, $S=\frac{1}{2}$')
    ax.plot([0, 0, 0.35], [0, 1, 1], '--', color=C['red'], lw=1.2, label='classical')
    for y, s in ((1 / 6, r'$\frac{1}{6}$'), (1 / 3, r'$\frac{1}{3}$'), (1 / 2, r'$\frac{1}{2}$'), (2 / 3, r'$\frac{2}{3}$')):
        i = np.where(np.isclose(6 * m[sel], y, atol=1e-3))[0]
        if len(i): ax.text(hs[i].mean(), y + 0.03, s, ha='center', va='bottom', fontsize=9)
    for hh in B['steps']: ax.axvline(hh[0], color=C['gray'], lw=0.4, ls=':')
    ax.set_xlabel('$h/J$'); ax.set_ylabel(r'$M/M_{\rm dice}$'); ax.set_xlim(0, 0.35); ax.set_ylim(-0.03, 1.1)
    ax.legend(loc='lower right'); tag(ax, '(b)', x=-0.15)
    plt.tight_layout(w_pad=1.5); fig.savefig(OUT + 'fig5_staircase.pdf')


# ------------------------------------------------------------------ Fig 6: fluctuations
def fig6():
    F = json.load(open('paper/fluct_structs.json')); B = json.load(open('big_staircase.json'))
    order = ['square', 'M6', 'M3', 'M2', 'M23', 'dice']
    steps = [0] + [s[0] for s in B['steps']] + [0.35]
    # the marginal M/3 window (width 0.004 J) is merged into its neighbours for display
    fig, axs = plt.subplots(2, 1, figsize=(W1, 4.6), sharex=True)
    def stair(ax, key, col, lab, scale=1):
        for k, (a, b) in zip(order, zip(steps[:-1], steps[1:])):
            ax.plot([a, b], [scale * F[k][key]] * 2, '-', color=col, lw=2)
        for k1, k2, x in zip(order[:-1], order[1:], steps[1:-1]):
            ax.plot([x, x], [scale * F[k1][key], scale * F[k2][key]], '-', color=col, lw=0.8)
        ax.plot([], [], '-', color=col, lw=2, label=lab)
    stair(axs[0], 'n', C['blue'], r'spin deviation $\langle\Delta S\rangle$')
    stair(axs[0], 's_site', C['red'], r'site entropy $s_1$ (nats)')
    axs[0].set_ylabel('per site'); axs[0].legend(loc='center left', bbox_to_anchor=(0, 0.47)); axs[0].set_ylim(0, 0.58); tag(axs[0], '(a)', x=-0.2)
    stair(axs[1], 'S_half', C['green'], r'$S_{A}/\ell$, half-torus cut')
    axs[1].set_ylabel(r'entanglement per boundary length'); axs[1].legend(loc='lower left'); tag(axs[1], '(b)', x=-0.2)
    lab = {'square': '0', 'M6': r'$\frac{1}{6}$', 'M3': r'$\frac{1}{3}$', 'M2': r'$\frac{1}{2}$', 'M23': r'$\frac{2}{3}$', 'dice': '1'}
    for k, (a, b) in zip(order, zip(steps[:-1], steps[1:])):
        axs[1].text((a + b) / 2, F[k]['S_half'] - 0.004, lab[k], ha='center', va='top', fontsize=8.5)
    axs[1].set_xlabel('$h/J$  (QMC phase boundaries)'); axs[1].set_xlim(0, 0.35)
    lo = min(F[k]['S_half'] for k in order); hi = max(F[k]['S_half'] for k in order)
    axs[1].set_ylim(lo - 0.05, hi + 0.012)
    plt.tight_layout(h_pad=0.6); fig.savefig(OUT + 'fig6_fluct.pdf')


# ------------------------------------------------------------------ Fig 7: S(q,w)
def fig7():
    d = np.load('paper/sqw_structs.npz'); wg, sg = d['wg'], d['sg']; t = sg / 36
    fig, axs = plt.subplots(2, 3, figsize=(W2, 4.1), sharex=True, sharey=True)
    ttl = {'square': 'square-type', 'M2': r'$M_{\rm dice}/2$ crystal', 'dice': 'dice'}
    for j, k in enumerate(('square', 'M2', 'dice')):
        for i, (cut, clab) in enumerate((('b1', r'$\mathbf{q}=s\,\mathbf{b}_1$'), ('b12', r'$\mathbf{q}=s(\mathbf{b}_1+\mathbf{b}_2)$'))):
            ax = axs[i, j]; I = d[f'{k}_{cut}']
            im = ax.imshow(np.log10(I.T + 1e-3), origin='lower', aspect='auto', extent=[t[0], t[-1], wg[0], wg[-1]],
                           cmap='magma', vmin=-2.3, vmax=0.8, interpolation='nearest')
            if i == 0: ax.set_title(ttl[k])
            if j == 0: ax.set_ylabel(r'$\omega/J$')
            if i == 1: ax.set_xlabel('$s$')
            ax.text(0.03, 0.95, clab, transform=ax.transAxes, color='w', fontsize=8, va='top')
            tag(ax, f'({"abcdef"[3 * i + j]})', x=0.99, y=0.86, color='w') if False else None
    cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.015); cb.set_label(r'$\log_{10}S^{xx}(\mathbf{q},\omega)$')
    fig.savefig(OUT + 'fig7_sqw.pdf')


# ------------------------------------------------------------------ Fig 8: spin S
def fig8():
    R = json.load(open('spinS.json'))
    S = np.array([r['S'] for r in R]); gap = np.array([r['gap'] for r in R]); dv = np.array([r['dev_M2'] for r in R]); ht = np.array([r['h_top'] for r in R])
    fig, axs = plt.subplots(2, 1, figsize=(W1, 4.6))
    ax = axs[0]
    ax.plot(1 / S, gap / S, 'o-', color=C['blue'], label=r'$(e_{\rm dice}-e_{\rm sq})/S$')
    ax.plot(1 / S, -10 * dv / S, 's-', color=C['red'], label=r'$-10\times$dev$(M_{\rm dice}/2)/S$')
    ax.plot([0], [0.050], 'o', mfc='white', mec=C['blue'], ms=7); ax.plot([0], [10 * 0.0077], 's', mfc='white', mec=C['red'], ms=7)
    ax.text(0.12, 0.05, r'LSWT ($S\to\infty$)', fontsize=8, va='center')
    ax.set_xlabel('$1/S$'); ax.set_ylabel(r'energy$/S$  ($J$)'); ax.set_xlim(-0.1, 2.2); ax.set_ylim(0, 0.12)
    ax.legend(loc='upper left'); tag(ax, '(a)', x=-0.2)
    ax = axs[1]
    ax.plot(S, ht, 'o-', color=C['green'], label=r'$h_c(M_{\rm dice}/2\to$dice$)$')
    ax.plot(S, ht / (12 * S) * 10, 's--', color=C['gray'], label=r'$10\,h_c/h_{\rm sat}$, $h_{\rm sat}=12JS$')
    ax.set_xlabel('$S$'); ax.set_ylabel('field ($J$)'); ax.set_ylim(0, 0.45); ax.set_xticks([0.5, 1, 1.5])
    ax.legend(loc='upper right'); tag(ax, '(b)', x=-0.2)
    plt.tight_layout(h_pad=1.0); fig.savefig(OUT + 'fig8_spinS.pdf')


# ------------------------------------------------------------------ Fig 9: melting
def fig9():
    M = json.load(open('melt.json'))
    fig, axs = plt.subplots(2, 1, figsize=(W1, 4.6))
    ax = axs[0]; cols = {0.15: C['blue'], 0.19: C['red'], 0.23: C['green']}
    Tm = {}
    for h in (0.15, 0.19, 0.23):
        for L, ls, mk in ((24, '-', 'o'), (48, '--', 's')):
            o = M[f'heat_L{L}_h{h}']; T = np.array([r['T'] for r in o]); q = np.array([r['q'] for r in o])
            ax.plot(T, q, ls, marker=mk, ms=3, color=cols[h], label=f'$h={h}$' if L == 24 else None)
            k = np.where(q < 0.5)[0]
            if len(k) and k[0] > 0:
                i = k[0]; Tm.setdefault(h, []).append(T[i - 1] + (0.5 - q[i - 1]) * (T[i] - T[i - 1]) / (q[i] - q[i - 1]))
    o = M['cool_L48_h0.19']; ax.plot([r['T'] for r in o], [r['q'] for r in o], ':', marker='^', ms=3, color='k', label='cooling, $h=0.19$')
    ax.plot([], [], 'k-', marker='o', ms=3, label='$L=24$'); ax.plot([], [], 'k--', marker='s', ms=3, label='$L=48$')
    ax.set_xlabel('$T/J$'); ax.set_ylabel('overlap with crystal $q$'); ax.set_xlim(0, 0.09); ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.98), ncol=1, fontsize=7.5); tag(ax, '(a)', x=-0.2)
    ax = axs[1]
    hs = sorted(Tm); ax.errorbar(hs, [np.mean(Tm[h]) for h in hs], [np.ptp(Tm[h]) / 2 + 5e-4 for h in hs], fmt='o-', color=C['blue'], capsize=2)
    ax.set_xlabel('$h/J$'); ax.set_ylabel(r'$T_m/J$  ($q=\frac{1}{2}$)'); ax.set_xlim(0.13, 0.25); ax.set_ylim(0, 0.036); tag(ax, '(b)', x=-0.2)
    json.dump({str(h): float(np.mean(v)) for h, v in Tm.items()}, open('paper/Tm.json', 'w'))
    plt.tight_layout(h_pad=1.0); fig.savefig(OUT + 'fig9_melting.pdf')


# ------------------------------------------------------------------ Fig A1: LSWT hull L dependence
def figA1():
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    for L, col, mk in ((18, C['blue'], 'o'), (24, C['red'], 's'), (30, C['green'], '^')):
        H = json.load(open(f'hull_fa_{L}.json')); m = np.array([p['m'] for p in H]); e = np.array([p['e'] for p in H])
        e_sq = e[np.argmin(m)]; e_d = e[np.argmax(m)]; ll = e_sq + (e_d - e_sq) * m / m.max()
        o = np.argsort(m); ax.plot(6 * m[o], 1e3 * (e - ll)[o], marker=mk, color=col, label=f'$L={L}$')
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_xlabel(r'$M/M_{\rm dice}$'); ax.set_ylabel(r'$e-e_{\rm lin}$ ($10^{-3}J$), LSWT'); ax.legend(loc='lower center', ncol=3)
    ax.set_ylim(-3.6, 0.6); fig.savefig(OUT + 'figA1_lswt_L.pdf')


# ------------------------------------------------------------------ Fig A2: tests of the coordination model and of the wall picture
def figA2():
    MT = json.load(open('paper/model_test.json'))
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.5))
    ax = axs[0]
    te = MT['test']; mm = np.array([6 * t['m'] for t in te]); r1 = np.array([t['pred_site'] - t['e'] for t in te]); r2 = np.array([t['pred_pair'] - t['e'] for t in te])
    ax.plot(mm - 0.012, 1e4 * r1, 'o', ms=3.5, mfc='none', mec=C['blue'], label='site model')
    ax.plot(mm + 0.012, 1e4 * r2, 's', ms=3.2, color=C['red'], label='+ bond-pair terms')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(r'$M/M_{\rm dice}$'); ax.set_ylabel(r'model $-$ LSWT ($10^{-4}J$)'); ax.set_ylim(-9, 6)
    ax.legend(loc='lower left', handletextpad=0.2); tag(ax, '(a)', x=-0.22)
    ax = axs[1]
    for L, col, mk in ((36, C['blue'], 'o'), (48, C['red'], 's')):
        W = json.load(open(f'paper/wall_d_L{L}.json' if L == 48 else 'paper/wall_d.json'))
        w = np.array([r['w'] for r in W['rows']]); s = np.array([r['sigma'] for r in W['rows']])
        ax.plot(w / L, 1e3 * s, marker=mk, color=col, label=f'$L={L}$')
    ax.set_xlabel('dice fraction $w/L$'); ax.set_ylabel(r'$\sigma$ per wall ($10^{-3}J/a$)'); ax.legend(loc='lower center'); tag(ax, '(b)', x=-0.22)
    ax = axs[2]
    CT = json.load(open('paper/cool_test.json')); cr = CT['crystal'][-1]; Gc = cr['e'] - 0.19 * cr['m']
    ns = [3000, 30000, 150000]; G = [CT[str(n)][-1]['e'] - 0.19 * CT[str(n)][-1]['m'] for n in ns]; q = [CT[str(n)][-1]['q'] for n in ns]
    ax.semilogx(ns, 1e3 * (np.array(G) - Gc), 'o-', color=C['blue'])
    ax.axhline(0, color='k', lw=0.5)
    for n, g, qq in zip(ns, G, q): ax.text(n, 1e3 * (g - Gc) + 0.08, f'$q$={qq:.2f}', ha='center', fontsize=8)
    ax.set_xlabel('sweeps per temperature'); ax.set_ylabel(r'$G_{\rm cooled}-G_{\rm crystal}$ ($10^{-3}J$)'); ax.set_ylim(-0.2, 1.6)
    ax.set_xlim(1500, 3e5); tag(ax, '(c)', x=-0.22)
    plt.tight_layout(w_pad=1.2); fig.savefig(OUT + 'figA2_tests.pdf')


if __name__ == '__main__':
    import os; os.makedirs(OUT, exist_ok=True)
    for f in (sys.argv[1:] or ['fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6', 'fig7', 'fig8', 'fig9', 'figA1']):
        globals()[f](); print('done', f, flush=True)
