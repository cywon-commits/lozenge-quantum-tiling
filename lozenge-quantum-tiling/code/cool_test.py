import numpy as np, json, time
from periodic import Cell, replicate
from tiling_mc import run
R = json.load(open('periodic_all_lswt.json')); R = [r for r in R if abs(r['m_rep'] - 1/12) < 1e-4]; R.sort(key=lambda r: r['e_lswt'])
def crystal(L):
    for r in R:
        try: return replicate(Cell(*r['cell']), [tuple(x) for x in r['tiling']], L)
        except Exception: pass
temps = list(np.round(np.linspace(0.05, 0.005, 19), 4)); out = {}
for ns in (3000, 30000, 150000):
    t = time.time(); o = run(crystal(24), crystal(24), 0.19, temps, nsweep=ns, every=10, seed=11)
    out[ns] = o; print(ns, 'final T', o[-1]['T'], 'q', round(o[-1]['q'], 3), 'M/Md', round(6 * o[-1]['m'], 3), 'e0', round(o[-1]['e'], 5), f'{time.time()-t:.0f}s', flush=True)
# reference: crystal kept at the lowest T
o = run(crystal(24), crystal(24), 0.19, [0.005], nsweep=3000, every=10, seed=3); out['crystal'] = o
print('crystal', o[-1])
json.dump({str(k): v for k, v in out.items()}, open('paper/cool_test.json', 'w'), indent=1)
