"""Wall-wall interaction in LSWT: a dice strip of width w in a square-type background (two walls at distance w),
and a square strip of width w in dice. L x L torus, walls along the strips."""
import json, numpy as np, time
from tilt import strips
from lswt import lswt_energy
import sys
L = int(sys.argv[1]); out = []
Tsq, _ = strips(L, [[0, L]]); Esq = lswt_energy(Tsq.N, Tsq.bonds())[0] / Tsq.N
Td, _ = strips(L, []); Ed = lswt_energy(Td.N, Td.bonds())[0] / Td.N
print('bulk', Esq, Ed, flush=True)
for w in range(6, L, 6):
    T, ok = strips(L, [[w, L - w]])          # square occupies L-w rows, dice w rows
    if not ok: continue
    try: E, S = lswt_energy(T.N, T.bonds())
    except ValueError: print(w,'non-bipartite'); continue
    f = w / L
    sig = (E - T.N * ((1 - f) * Esq + f * Ed)) / (2 * L)      # per wall per unit length (a = lattice constant)
    out.append(dict(w=w, f=f, e=E / T.N, m=S / T.N, sigma=sig)); print(w, round(E / T.N, 6), round(sig, 5), flush=True)
json.dump(dict(L=L, Esq=Esq, Ed=Ed, rows=out), open(f'paper/wall_d_L{L}.json', 'w'), indent=1)
