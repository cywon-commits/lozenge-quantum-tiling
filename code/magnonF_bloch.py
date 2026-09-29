import numpy as np, json, glob
from periodic import Cell
from bloch_lswt import energy
O = json.load(open('periodic_all_lswt.json'))
for f in glob.glob('big_*_s*.json'): O += json.load(open(f))
def best(m, key='e_lswt'):
    c = [o for o in O if abs(o['m'] - m) < 1e-4]
    return min(c, key=lambda o: o.get('e_lswt', o['e_model']))
out = {}
for lab, m in (('square', 0), ('M6', 1/36), ('M3', 1/18), ('M2', 1/12), ('M23', 1/9), ('dice', 1/6)):
    r = best(m); C = Cell(*r['cell'])
    E, mm, om = energy(C, r['tiling'], nk=64, return_omega=True)
    w = om.ravel(); w = w[w > 1e-9]; nk2 = om.shape[0]
    row = dict(cell=r['cell'], m=mm, e=E)
    for T in (0.01, 0.02, 0.03, 0.05):
        row[f'F_{T}'] = float(T * np.sum(np.log1p(-np.exp(-w / T))) / (nk2 * C.N))
    out[lab] = row; print(lab, row, flush=True)
json.dump(out, open('paper/magnonF_bloch.json', 'w'), indent=1)
