"""Orthographic painter's-algorithm rendering of the lifted stepped surface, vertices colored by curvature."""
import numpy as np, pickle, sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from dice_string import Tiling
from tiling_moves import degrees
from lift import lift, faces
FACE = {0: '#f4f1ea', 1: '#c7c1b4', 2: '#948c7e'}
KC = {90: '#D55E00', -90: '#56B4E9', -180: '#08306b'}
def basis(tilt_deg, rot_deg):
    n = np.ones(3) / np.sqrt(3)
    a = np.array([1, -1, 0]) / np.sqrt(2); b = np.cross(n, a)
    t, r = np.radians(tilt_deg), np.radians(rot_deg)
    ax_ = np.cos(r) * a + np.sin(r) * b
    v = np.cos(t) * n + np.sin(t) * np.cross(ax_, n)          # view direction tilted about ax_
    u1 = ax_; u2 = np.cross(v, u1)
    return v, u1, u2
def panel(ax, T, title, n=20, tilt=30, rot=0, rad=0.16, fs=9, half=3.6):
    z = degrees(T); H = lift(T, 0, n); F = faces(T, H)
    v, u1, u2 = basis(tilt, rot)
    items = []
    for f in F:
        P = np.array(f, float); nrm = np.cross(P[1] - P[0], P[3] - P[0]); o = int(np.argmax(np.abs(nrm)))
        items.append((P.mean(0) @ v, 'f', P, FACE[o]))
    for (i, j), h in H.items():
        K = int(90 * (4 - z[T.s(i % T.L, j % T.L)]))
        if K in KC: items.append((np.array(h, float) @ v + 0.3, 'p', np.array(h, float), K))
    items.sort(key=lambda x: x[0])
    xy = []
    for d, kind, P, c in items:
        if kind == 'f':
            Q = np.c_[P @ u1, P @ u2]; ax.add_patch(Polygon(Q, fc=c, ec='#5a5a5a', lw=0.4)); xy.append(Q)
        else:
            q = (P @ u1, P @ u2); ax.add_patch(Circle(q, rad * (1.35 if c == -180 else 1), fc=KC[c], ec='k', lw=0.3, zorder=3))
    Q = np.vstack(xy); c = 0.5 * (Q.min(0) + Q.max(0)); ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - 0.8 * half, c[1] + 0.8 * half)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title, fontsize=fs, pad=2)
def load(key):
    S = pickle.load(open('paper/structs_L36.pkl', 'rb')); S['pleated'] = pickle.load(open('paper/pleated_L36.pkl', 'rb'))
    T = Tiling(36); T.removed = S[key]; return T
if __name__ == '__main__':
    fig, axs = plt.subplots(2, 3, figsize=(9, 6))
    for ax, key in zip(axs.ravel(), ('pleated', 'M6', 'M3', 'M2', 'M23', 'dice')):
        panel(ax, load(key), key, tilt=float(sys.argv[1]), rot=float(sys.argv[2]))
    plt.savefig('figures/curv_test.png', dpi=90)


def paper_fig(out='paper/figs/fig10_curvature.pdf'):
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm'})
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.3))
    T_ = [('pleated', r'(a) $h=0$ family: folded, $K=0$'), ('M6', r'(b) $M_{\rm dice}/6$'), ('M3', r'(c) $M_{\rm dice}/3$'),
          ('M2', r'(d) $M_{\rm dice}/2$: no hubs'), ('M23', r'(e) $2M_{\rm dice}/3$'), ('dice', r'(f) dice')]
    for ax, (k, t) in zip(axs.ravel(), T_): panel(ax, load(k), t, fs=9.5, rad=0.2, half=4.6)
    h = [Line2D([], [], marker='o', ls='', ms=7, mfc=KC[90], mec='k', mew=0.4, label=r'$K=+\pi/2$ ($z=3$)'),
         Line2D([], [], marker='o', ls='', ms=7, mfc=KC[-90], mec='k', mew=0.4, label=r'$K=-\pi/2$ ($z=5$)'),
         Line2D([], [], marker='o', ls='', ms=9, mfc=KC[-180], mec='k', mew=0.4, label=r'$K=-\pi$ ($z=6$, hub)')]
    fig.legend(handles=h, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=9)
    plt.subplots_adjust(wspace=0.04, hspace=0.22, bottom=0.08, top=0.95, left=0.01, right=0.99)
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)


def slide_fig(out='slides/img/curvature.png'):
    plt.rcParams.update({'font.family': 'sans-serif'})
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(1, 4, figsize=(16.6, 4.5), facecolor='#FBFAF6')
    for ax, (k, t) in zip(axs, (('pleated', 'h = 0: folded, K = 0'), ('M3', 'M_dice/3'), ('M2', 'M_dice/2 (no hubs)'), ('dice', 'dice'))):
        panel(ax, load(k), t, fs=21, rad=0.2, half=3.3)
    h = [Line2D([], [], marker='o', ls='', ms=16, mfc=KC[90], mec='k', mew=0.6, label='K = +π/2  (z = 3, cone)'),
         Line2D([], [], marker='o', ls='', ms=16, mfc=KC[-90], mec='k', mew=0.6, label='K = −π/2  (z = 5, saddle)'),
         Line2D([], [], marker='o', ls='', ms=19, mfc=KC[-180], mec='k', mew=0.6, label='K = −π  (z = 6, hub)')]
    fig.legend(handles=h, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=19)
    plt.subplots_adjust(wspace=0.05, bottom=0.12)
    fig.savefig(out, dpi=200, facecolor='#FBFAF6', bbox_inches='tight', pad_inches=0.08)
