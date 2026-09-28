"""Slide-sized figures: figsize in inches = pixels/100 on the 1920x1080 canvas, fonts in pt (1 pt = 1.39 px)."""
import json, pickle, numpy as np, matplotlib, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import paper_figs as P
from dice_string import Tiling

plt.rcParams.update({'font.size': 19, 'axes.labelsize': 20, 'axes.titlesize': 20, 'xtick.labelsize': 17, 'ytick.labelsize': 17,
                     'legend.fontsize': 17, 'lines.linewidth': 2.6, 'lines.markersize': 10, 'axes.linewidth': 1.2,
                     'xtick.major.width': 1.2, 'ytick.major.width': 1.2, 'xtick.major.size': 6, 'ytick.major.size': 6,
                     'savefig.pad_inches': 0.08, 'font.family': 'sans-serif'})
C = P.C; OUT = 'slides/img/'
BG = '#FBFAF6'


def save(fig, name):
    fig.savefig(OUT + name + '.png', dpi=200, facecolor=BG); plt.close(fig)


def big_z_legend(fig, y):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', ls='', ms=14, mfc=P.ZC[k], mec='k', mew=0.6, label=f'z = {k}') for k in (3, 4, 5, 6)]
    fig.legend(handles=h, loc='lower center', ncol=4, bbox_to_anchor=(0.5, y), fontsize=19, columnspacing=2.0, handletextpad=0.3)


def tilings():
    S = pickle.load(open('paper/structs_L36.pkl', 'rb'))
    def T36(k): T = Tiling(36); T.removed = S[k]; return T
    Tr = Tiling(18); Tr.removed = pickle.load(open('ann_18_12_0.09_22_300.pkl', 'rb'))
    old = dict(P.ZS); P.ZS.update({3: 7, 4: 8, 5: 9, 6: 11})
    fig, axs = plt.subplots(1, 3, figsize=(16.6, 5.6), facecolor=BG)
    for ax, T, t in zip(axs, (T36('square'), Tr, T36('dice')), ('square-type  (M = 0)', 'generic tiling', 'dice  (M = M_dice)')):
        P.draw_tiling(ax, T, (-0.5, 7.5, -0.5, 5.6)); ax.set_title(t, fontsize=22, pad=8)
    plt.subplots_adjust(wspace=0.05, bottom=0.12); big_z_legend(fig, -0.01); save(fig, 'tilings')
    fig, axs = plt.subplots(1, 3, figsize=(16.6, 5.6), facecolor=BG)
    for ax, k, t in zip(axs, ('M6', 'M2', 'M23'), ('M_dice / 6', 'M_dice / 2  (no hubs)', '2 M_dice / 3')):
        P.draw_tiling(ax, T36(k), (-0.5, 9.5, -0.5, 7.3)); ax.set_title(t, fontsize=22, pad=8)
    plt.subplots_adjust(wspace=0.05, bottom=0.12); big_z_legend(fig, -0.01); save(fig, 'crystals')
    fig, ax = plt.subplots(figsize=(8.0, 6.4), facecolor=BG)
    P.draw_tiling(ax, T36('M2'), (-0.5, 11.5, -0.5, 9.0)); save(fig, 'cover_m2')
    P.ZS.update(old)


def eps():
    EPS = {3: -0.8544, 4: -0.6582, 5: -0.4423, 6: -0.2061}
    fig, ax = plt.subplots(figsize=(7.8, 6.2), facecolor=BG); ax.set_facecolor(BG)
    zs = np.array([3, 4, 5, 6]); ev = np.array([EPS[k] for k in zs]); p = np.polyfit(zs, ev, 2); zz = np.linspace(3, 6, 60)
    ax.plot(zz, np.polyval(p, zz), '-', color=C['blue'], lw=2); ax.plot(zs, ev, 'o', color=C['blue'], ms=14, zorder=3, label='zero-point energy  ε(z)')
    ax.plot([3, 6], [EPS[3], EPS[6]], '--', color=C['red'], lw=2.4, label='dice average (z = 3,3,6)')
    y4 = (2 * EPS[3] + EPS[6]) / 3
    ax.plot([4], [y4], 's', mfc='white', mec=C['red'], mew=2.4, ms=14, zorder=4)
    ax.annotate('', xy=(4.0, EPS[4] - 0.003), xytext=(4.0, y4 + 0.003), arrowprops=dict(arrowstyle='<->', color='k', lw=1.8))
    ax.text(4.12, -0.70, 'square wins\nby 0.020 J/site', fontsize=19, va='center')
    ax.set_xlabel('site coordination  z'); ax.set_ylabel('ε(z)   (J per site)'); ax.set_xticks([3, 4, 5, 6])
    ax.legend(loc='upper left', frameon=False); save(fig, 'eps')


def walls():
    Wd = json.load(open('walls.json'))
    fig, ax = plt.subplots(figsize=(7.8, 5.8), facecolor=BG); ax.set_facecolor(BG)
    Ls = np.array([w['L'] for w in Wd]); sg = np.array([w['sigma'] for w in Wd]); ss = np.array([w['sig_sys'] for w in Wd])
    ax.errorbar(Ls, sg, ss, fmt='o', color=C['blue'], ms=11, capsize=4, lw=2)
    ax.axhline(0, color='k', lw=1.2); ax.axhspan(0, 0.012, color='#eeeeee', zorder=-1)
    ax.text(21, 0.0065, 'costly walls would be here', ha='center', fontsize=18, color='#555555')
    ax.set_xlabel('torus size  L'); ax.set_ylabel('wall energy  σ  (J / a)'); ax.set_ylim(-0.022, 0.012)
    save(fig, 'walls')


def stair():
    B = json.load(open('big_staircase.json')); Pp = np.array(B['points']); m, e, er = Pp[:, 0], Pp[:, 1], Pp[:, 2]
    A = json.load(open('hull_qmc_b.json')); lin = P.lin
    fig, axs = plt.subplots(1, 2, figsize=(16.6, 6.2), facecolor=BG)
    ax = axs[0]; ax.set_facecolor(BG); first = True
    for L, pts in A.items():
        q = [p for p in pts if p['pkl'] not in ('square', 'dice')]
        ax.errorbar([6 * p['m'] for p in q], [1e3 * (p['e0'] - lin(p['m'])) for p in q], [1e3 * p['err'] for p in q], fmt='o', ms=8,
                    color=C['gray'], mfc='white', lw=1.4, label='irregular networks' if first else None); first = False
    from hull_lswt import lower_hull
    H = lower_hull(m, e); ax.plot(6 * m[H], 1e3 * (e[H] - lin(m[H])), '-', color=C['red'], lw=2.4, zorder=1)
    ax.errorbar(6 * m, 1e3 * (e - lin(m)), 1e3 * er, fmt='s', color=C['red'], ms=13, capsize=4, lw=2, label='periodic crystals', zorder=3)
    ax.axhline(0, color='k', lw=1, ls=':')
    ax.set_xlabel('M / M_dice'); ax.set_ylabel('energy below the straight line\n(10⁻³ J per site)'); ax.set_ylim(-6.3, 2.2)
    ax.set_xticks([0, 1/6, 1/3, 1/2, 2/3, 1]); ax.set_xticklabels(['0', '1/6', '1/3', '1/2', '2/3', '1'])
    ax.legend(loc='upper center', frameon=False, ncol=2, columnspacing=1.0)
    ax = axs[1]; ax.set_facecolor(BG)
    hs = np.linspace(0, 0.35, 3501); chi = np.where(m == 0, 0.065, 0.0)
    G = e[None, :] - hs[:, None] * m[None, :] - 0.5 * chi[None, :] * hs[:, None] ** 2; sel = G.argmin(1)
    ax.plot([0, 0, 0.35], [0, 1, 1], '--', color=C['red'], lw=2.4, label='classical: jump at h = 0')
    ax.plot(hs, 6 * m[sel], '-', color=C['blue'], lw=3.4, label='quantum S = 1/2 (QMC)')
    for y, s in ((1/6, '1/6'), (1/3, '1/3'), (1/2, '1/2'), (2/3, '2/3')):
        i = np.where(np.isclose(6 * m[sel], y, atol=1e-3))[0]; ax.text(hs[i].mean(), y + 0.035, s, ha='center', va='bottom', fontsize=19, color=C['blue'])
    ax.set_xlabel('magnetic field  h / J'); ax.set_ylabel('M / M_dice'); ax.set_xlim(0, 0.35); ax.set_ylim(-0.03, 1.12)
    ax.legend(loc='lower right', frameon=False)
    plt.tight_layout(w_pad=3); save(fig, 'staircase')


def fluct():
    F = json.load(open('paper/fluct_structs.json')); B = json.load(open('big_staircase.json'))
    order = ['square', 'M6', 'M3', 'M2', 'M23', 'dice']
    steps = [0] + [s[0] for s in B['steps']] + [0.35]
    fig, ax = plt.subplots(figsize=(7.8, 5.8), facecolor=BG); ax.set_facecolor(BG)
    for key, col, lab in (('s_site', C['red'], 'single-site entanglement'), ('n', C['blue'], 'spin deviation  ⟨ΔS⟩')):
        for k, (a, b) in zip(order, zip(steps[:-1], steps[1:])): ax.plot([a, b], [F[k][key]] * 2, '-', color=col, lw=3.4)
        for k1, k2, x in zip(order[:-1], order[1:], steps[1:-1]): ax.plot([x, x], [F[k1][key], F[k2][key]], '-', color=col, lw=1.4)
        ax.plot([], [], '-', color=col, lw=3.4, label=lab)
    ax.set_xlabel('magnetic field  h / J'); ax.set_ylabel('per site'); ax.set_ylim(0, 0.62); ax.set_xlim(0, 0.35)
    ax.legend(loc='upper right', frameon=False); save(fig, 'fluct')
    d = np.load('paper/sqw_structs.npz'); wg, sg = d['wg'], d['sg']; t = sg / 36
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 5.8), sharey=True, facecolor=BG)
    for ax, k, tt in zip(axs, ('square', 'M2'), ('square-type', 'M_dice/2 crystal')):
        ax.imshow(np.log10(d[f'{k}_b1'].T + 1e-3), origin='lower', aspect='auto', extent=[t[0], t[-1], wg[0], wg[-1]], cmap='magma', vmin=-2.3, vmax=0.8)
        ax.set_title(tt, fontsize=20); ax.set_xlabel('q / b₁')
    axs[0].set_ylabel('ω / J'); plt.tight_layout(w_pad=0.8); save(fig, 'sqw')


def spin_melt():
    R = json.load(open('spinS.json'))
    S = np.array([r['S'] for r in R]); ht = np.array([r['h_top'] for r in R])
    fig, ax = plt.subplots(figsize=(7.8, 5.6), facecolor=BG); ax.set_facecolor(BG)
    ax.plot(S, ht, 'o-', color=C['green'], ms=14, lw=3, label='M/2 → dice step field')
    ax.set_xlabel('spin length  S'); ax.set_ylabel('h_c / J'); ax.set_ylim(0, 0.4); ax.set_xticks([0.5, 1, 1.5]); ax.set_xticklabels(['1/2', '1', '3/2'])
    ax.legend(loc='upper right', frameon=False); save(fig, 'spinS')
    M = json.load(open('melt.json')); cols = {0.15: C['blue'], 0.19: C['red'], 0.23: C['green']}
    fig, ax = plt.subplots(figsize=(7.8, 5.6), facecolor=BG); ax.set_facecolor(BG)
    for h in (0.15, 0.19, 0.23):
        o = M[f'heat_L48_h{h}']; ax.plot([r['T'] for r in o], [r['q'] for r in o], '-o', ms=6, color=cols[h], label=f'heating, h = {h} J')
    o = M['cool_L48_h0.19']; ax.plot([r['T'] for r in o], [r['q'] for r in o], ':^', ms=7, color='k', lw=2.4, label='slow cooling  (glass)')
    ax.set_xlabel('temperature  T / J'); ax.set_ylabel('overlap with crystal'); ax.set_xlim(0, 0.08); ax.set_ylim(0, 1.08)
    ax.legend(loc='upper right', frameon=False); save(fig, 'melt')


if __name__ == '__main__':
    import os; os.makedirs(OUT, exist_ok=True)
    for f in (sys.argv[1:] or ['tilings', 'eps', 'walls', 'stair', 'fluct', 'spin_melt']):
        globals()[f](); print('done', f, flush=True)
