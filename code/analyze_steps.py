"""Combine h=0 and in-field QMC runs of the winding (tilt) sectors.
Model per sector k:  m_k(h) = mL_k + chi_k h   ->  e_k(h) = e0_k - mL_k h - chi_k h^2/2
Each run gives an estimate  e0 = e + h (m + mL)/2  and (if h>0) chi = (m - mL)/h.
"""
import json, glob, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

rows = []
for f in ['q3.log', 'q4.log'] + glob.glob('r_*.log') + glob.glob('field*.log') + glob.glob('multi*.log'):
    for l in open(f):
        if l.startswith('{'):
            r = json.loads(l)
            if r.get('kind') in ('tilt', 'dice') and r['L'] in (18, 24):
                r['multi'] = r.get('kind') == 'tilt' and len(r['iv']) > 1
                rows.append(r)
rng = np.random.default_rng(0)
out = {}
for L in (18, 24):
    N = L * L
    sec = {}
    for r in rows:
        if r['L'] != L or r['multi']: continue
        mL = r['S_lieb'] / N
        h = r['h']; m = r['m'] if h > 0 else mL
        e0 = r['e'] + h * (m + mL) / 2
        err = max(r.get('e_err', 5e-4), 3e-4)
        s = sec.setdefault(mL, dict(e0=[], w=[], chi=[]))
        s['e0'].append(e0); s['w'].append(1 / err ** 2)
        if h > 0: s['chi'].append((m - mL) / h)
    ms = np.array(sorted(sec))
    e0 = np.array([np.average(sec[m]['e0'], weights=sec[m]['w']) for m in ms])
    # error: max(statistical, scatter between runs)
    eerr = []
    for m in ms:
        w = np.array(sec[m]['w']); x = np.array(sec[m]['e0'])
        stat = 1 / np.sqrt(w.sum())
        scat = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0
        eerr.append(max(stat, scat, 3e-4))
    eerr = np.array(eerr)
    chi = np.array([np.mean(sec[m]['chi']) if sec[m]['chi'] else 0.0 for m in ms])
    out[L] = dict(ms=ms, e0=e0, eerr=eerr, chi=chi, n=[len(sec[m]['e0']) for m in ms])
    # step fields with bootstrap
    hs = np.linspace(0, 0.35, 1401)
    def steps(e0s):
        G = e0s[None, :] - hs[:, None] * ms[None, :] - 0.5 * chi[None, :] * hs[:, None] ** 2
        sel = G.argmin(1)
        return [hs[i] for i in range(1, len(hs)) if sel[i] != sel[i - 1]], ms[sel]
    st0, msel = steps(e0)
    boot = []
    for b in range(2000):
        st, _ = steps(e0 + rng.normal(0, eerr))
        boot.append(st)
    nsteps = [len(s) for s in boot]
    out[L].update(steps=st0, msel=msel, hs=hs, frac_full=np.mean(np.array(nsteps) == len(ms) - 1),
                  boot=[s for s in boot if len(s) == len(ms) - 1])
    print(f"L={L}")
    for m, e, er, c, n in zip(ms, e0, eerr, chi, out[L]['n']):
        print(f"   m={m:.4f}  e0={e:.5f} ± {er:.5f}  chi={c:.3f}  (runs {n})")
    B = np.array(out[L]['boot'])
    print("   step fields:", [f"{s:.3f}" for s in st0], " bootstrap mean±sd:",
          [f"{a:.3f}±{b:.3f}" for a, b in zip(B.mean(0), B.std(0))] if len(B) else None,
          f"  P(all {len(ms)-1} steps resolved) = {out[L]['frac_full']:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for L, mk in ((18, 'o'), (24, 's')):
    o = out[L]
    ax[0].errorbar(o['ms'], o['e0'], o['eerr'], marker=mk, label=f'L={L}')
    ax[1].plot(o['hs'], o['msel'], label=f'L={L}: steps at ' + ', '.join(f'{s:.3f}' for s in o['steps']))
for L, mk in ((18, 'o'), (24, 's')):
    for r in rows:
        if r['L'] == L and r['multi']:
            N = L * L; mL = r['S_lieb'] / N; e0 = r['e'] + r['h'] * (r['m'] + mL) / 2
            ax[0].plot(mL + (0.002 if L == 24 else -0.002), e0, marker='*', ms=11, color='C3' if L == 24 else 'C2', ls='none')
ax[0].plot([], [], '*', color='C2', ms=11, label='two strips (4 interfaces), L=18')
ax[0].plot([], [], '*', color='C3', ms=11, label='two strips (4 interfaces), L=24')
ax[0].set_xlabel('Lieb moment per site m'); ax[0].set_ylabel('e(h=0) per site (J)'); ax[0].legend(fontsize=8)
ax[0].set_title('winding (tilt) sectors: e(m) convex?')
ax[1].plot([0, 0, 0.35], [0, 1 / 6, 1 / 6], 'r--', label='classical')
ax[1].set_xlabel('h / J'); ax[1].set_ylabel('selected sector m'); ax[1].legend(fontsize=8)
ax[1].set_title('T=0 sector selection incl. measured canting χ')
plt.tight_layout(); plt.savefig('figures/tilt_staircase_L18_L24.png', dpi=140)
