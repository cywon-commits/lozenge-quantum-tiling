"""Fig. 11: defect lattice gas (cluster expansion) fitted to Bloch-LSWT energies of all hub-free periodic tilings (cells <= 36 sites)."""
import json, os, numpy as np, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ce'))
from paper_figs import plt, C, W1, tag, OUT
from features import NAMES, SHELLS
R = json.load(open('ce/hubfree_lswt.json')); X = np.load('ce/X.npy'); y = np.array([r['e_inf'] for r in R])
rng = np.random.default_rng(0); fold = rng.integers(0, 5, len(y))
def cvpred(cols):
    A = np.c_[np.ones(len(y)), X[:, cols]]; p = np.zeros(len(y))
    for k in range(5):
        t = fold != k; w = np.linalg.lstsq(A[t], y[t], rcond=None)[0]; p[~t] = A[~t] @ w
    return p
F = json.load(open('ce/fit.json'))
p0 = cvpred([0, 1]); p1 = cvpred(F['+ pairs up to r2<=7']['cols'])
rms = lambda d: np.sqrt(np.mean(d ** 2))
out = dict(n=len(y), rms_site=rms(y - p0), rms_pair=rms(y - p1))
fig, axs = plt.subplots(2, 1, figsize=(W1, 5.0))
ax = axs[0]; c = y.mean()
ax.plot(1e3 * (p0 - c), 1e3 * (y - c), '.', ms=2.0, color=C['gray'], alpha=0.5, label=r'site model, rms $%.1f\times10^{-4}J$' % (1e4 * out['rms_site']))
ax.plot(1e3 * (p1 - c), 1e3 * (y - c), '.', ms=2.0, color=C['blue'], alpha=0.6, label=r'defect lattice gas, rms $%.1f\times10^{-5}J$' % (1e5 * out['rms_pair']))
lo, hi = 1e3 * (y.min() - c) - 0.3, 1e3 * (y.max() - c) + 0.3
ax.plot([lo, hi], [lo, hi], 'k-', lw=0.6); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel(r'model (cross-validated, shifted, $10^{-3}J$)'); ax.set_ylabel(r'LSWT $e_\infty$ (shifted, $10^{-3}J$)')
ax.legend(loc='upper left', markerscale=4, handletextpad=0.3); tag(ax, '(a)', x=-0.2)
ax.text(0.97, 0.04, f'{len(y)} hub-free tilings\ncells $\\leq36$ sites', transform=ax.transAxes, ha='right', va='bottom', fontsize=8)
# (b) pair energies vs distance
G = json.load(open('ce/fit.json'))['+ pairs up to r2<=16']; w = dict(zip([NAMES[i] for i in G['cols']], G['w'][1:]))
ax = axs[1]
for kk, col, mk, lab in ((1, C['blue'], 'o', r"same curvature ($kk'=+1$): $W<0$"), (-1, C['red'], 's', r"opposite curvature ($kk'=-1$): $W>0$")):
    r, v = [], []
    for s in SHELLS:
        vals = [w[f'r2={s}|kk={kk:+d}|ee={ee:+d}'] for ee in (1, -1)]
        if s == 1: vals[1] += w[f'nn_bonded|kk={kk:+d}']          # bonded nearest neighbours (opposite sublattice)
        r.append(np.sqrt(s)); v.append(np.mean(np.abs(vals)))
    ax.semilogy(r, 1e3 * np.array(v), mk + '-', color=col, ms=5, label=lab)
rr = np.linspace(1, 4, 50); ax.semilogy(rr, 2.3 * rr ** -3.5, 'k--', lw=0.8, label=r'$\propto r^{-3.5}$')
ax.axhspan(1e-3, 0.02, color='0.92', zorder=-1); ax.text(1.05, 0.013, 'fit resolution', ha='left', va='top', fontsize=8, color='0.35')
ax.set_xlabel(r'defect separation $r$ (lattice units)'); ax.set_ylabel(r'$|W(r)|$ ($10^{-3}J$)'); ax.set_ylim(4e-3, 60)
ax.legend(loc='upper right', handlelength=1.6); tag(ax, '(b)', x=-0.2)
plt.tight_layout(h_pad=1.0); fig.savefig(OUT + 'fig11_defectgas.pdf')
json.dump(out, open('paper/defectgas.json', 'w')); print(out)
