"""Stage C: turn the new LSWT hull vertices (and optional extra targets) into L=36 tilings + QMC jobs.

    python make_qmc.py [--extra 0.0417,0.0625] [--seeds 1,2] [--top 2]

For every target imbalance m (new hull vertices from rank_result.json, plus --extra values) the
--top lowest-e_inf candidates are replicated on the 36x36 torus (qnew_<tag>.pkl) and one QMC job
per seed is written to jobs_qmc.txt:
    python ../code/qmc_ckpt.py qnew_<tag>.pkl 36 36 <h> 1200 6000 <seed> >> qnew_<tag>_s<seed>.log
Candidates whose L=36 LSWT energy equals a published crystal (ref/big_m*_L36.pkl, |dE|<1e-7) are skipped.
The field h is the middle of the crystal's LSWT window, clipped to [0.10, 0.20] J (same protocol
as the published L=36 runs: beta=36, 1200 thermalisation + 6000 measurement sweeps).
"""
import json, sys, pickle, os, glob
from periodic import Cell, replicate
from dice_string import Tiling
from lswt import lswt_energy

args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return default
extra = [float(x) for x in opt('--extra', '').split(',') if x]
seeds = [int(x) for x in opt('--seeds', '1,2').split(',')]
top = int(opt('--top', '2'))
code = os.path.dirname(os.path.abspath(__file__))

R = json.load(open('rank_result.json'))
targets = sorted({round(p['m'], 8) for p in R['new_vertices']} | {round(x, 8) for x in extra})
H = R['hull']
def window(m):
    for a, b in zip(H, H[1:]):
        if abs(b['m'] - m) < 1e-8:
            lo = (b['e'] - a['e']) / (b['m'] - a['m'])
            i = H.index(b); hi = (H[i + 1]['e'] - b['e']) / (H[i + 1]['m'] - b['m']) if i + 1 < len(H) else lo + 0.04
            return lo, hi
    return 0.10, 0.20
# published L=36 QMC crystals (ref/big_m*_L36.pkl): real-space LSWT energy as structure fingerprint
REFD = '.'   # run from data/ (published crystals big_m*_L36.pkl); published = {}
for f in glob.glob(os.path.join(REFD, 'big_m*_L36.pkl')):
    T = Tiling(36); T.removed = pickle.load(open(f, 'rb')); E, S = lswt_energy(T.N, T.bonds())
    published[os.path.basename(f)] = E / T.N
jobs = []
for m in targets:
    C = sorted([r for r in R['candidates'] if abs(r['m'] - m) < 1e-8], key=lambda r: r['e_inf'])[:top]
    lo, hi = window(m); h = round(min(0.20, max(0.10, 0.5 * (lo + hi))), 3)
    for j, r in enumerate(C):
        tag = f"M{6*m:.4f}_{r['cell'][0]}x{r['cell'][1]}s{r['cell'][2]}_{j}"
        T = replicate(Cell(*r['cell']), [tuple(x) for x in r['tiling']], 36)
        E36 = lswt_energy(T.N, T.bonds())[0] / T.N
        same = [k for k, v in published.items() if abs(v - E36) < 1e-7]
        if same:
            print(tag, 'identical to published crystal', same[0], '-> skipped'); continue
        pickle.dump(T.removed, open(f'qnew_{tag}.pkl', 'wb'))
        json.dump(dict(tag=tag, m=m, cell=r['cell'], hist=r['hist'], e_inf=r['e_inf'], h=h, window=[lo, hi]),
                  open(f'qnew_{tag}.json', 'w'))
        for sd in seeds:
            jobs.append(f"python {code}/qmc_ckpt.py qnew_{tag}.pkl 36 36 {h} 1200 6000 {sd} >> qnew_{tag}_s{sd}.log 2>&1")
        print(tag, 'h =', h, 'LSWT window', (round(lo, 4), round(hi, 4)))
open('jobs_qmc.txt', 'w').write('\n'.join(jobs) + '\n')
print(len(jobs), 'jobs -> jobs_qmc.txt')
