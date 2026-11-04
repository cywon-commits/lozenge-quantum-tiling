"""Second-order expansion about the Ising limit, Eq. (ising) of the manuscript (run from data/):
e = -S^2 N_b/N - (S/N) sum_<ij> 1/(z_i + z_j - 1/S)   (J = 1).
Prints the deviation of each crystal from the formula's own square-dice line."""
import sys, pickle, numpy as np
sys.path.insert(0, '../code')
from dice_string import Tiling
from tiling_moves import degrees
from lswt import signs
S = pickle.load(open('paper/structs_L36.pkl', 'rb')); S8 = pickle.load(open('paper/structs_v8.pkl', 'rb'))
def get(k):
    if k in S8: T = Tiling(S8[k]['L']); T.removed = set(S8[k]['removed'])
    else: T = Tiling(36); T.removed = set(S[k])
    return T
def ising(T, Sp=0.5):
    B = np.array(T.bonds()); z = degrees(T)
    return -Sp * Sp * len(B) / T.N - Sp / T.N * np.sum(1 / (z[B[:, 0]] + z[B[:, 1]] - 1 / Sp)), B
res = {}
for k in ('square', 'M111', 'M19', 'M17', 'M6', 'M15', 'M14', 'M3', 'M2', 'M35', 'M23', 'dice'):
    T = get(k); e, B = ising(T); m = abs(np.sum(signs(T.N, B))) / 2 / T.N; res[k] = (m, e)
esq, ed = res['square'][1], res['dice'][1]
for k, (m, e) in res.items(): print(f'{k:6s} M/Md={6*m:.4f}  e={e:.5f}  dev={1e3*(e-(esq+6*m*(ed-esq))):+.2f}e-3')
print('e_dice - e_sq =', ed - esq)
