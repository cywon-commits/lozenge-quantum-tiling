import json, numpy as np
B = json.load(open('big_staircase.json')); P = np.array(B['points']); m, e, er = P[:, 0], P[:, 1], P[:, 2]
rng = np.random.default_rng(1); hs = np.linspace(0, 0.35, 3501); chi = np.where(m == 0, 0.065, 0.0)
lab = [round(6 * x, 3) for x in m]; pres = {l: 0 for l in lab}; bounds = {}
NS = 4000
for s in range(NS):
    es = e + rng.normal(0, er)
    G = es[None, :] - hs[:, None] * m[None, :] - 0.5 * chi[None, :] * hs[:, None] ** 2; sel = G.argmin(1)
    for k in set(sel): pres[lab[k]] += 1
    for i in np.where(np.diff(sel) != 0)[0]:
        bounds.setdefault((lab[sel[i]], lab[sel[i + 1]]), []).append(hs[i + 1])
out = dict(presence={str(k): v / NS for k, v in pres.items()},
           steps={f"{a}->{b}": dict(freq=len(v) / NS, mean=float(np.mean(v)), std=float(np.std(v))) for (a, b), v in bounds.items() if len(v) > NS * 0.02})
# window widths of the plateaus that are present
json.dump(out, open('paper/stair_boot.json', 'w'), indent=1); print(json.dumps(out, indent=1))
