"""SSE (S=1/2) energy decomposition on a fixed tiling: E_xy = -<n_offdiag>/beta, E_zeeman = -h<m>, E_zz = E - E_xy - E_zeeman.
Also the equal-time estimator <S^z_i S^z_j> from the tau=0 spin state (check of E_zz).
usage: python qdec.py <structs.pkl> <key> L beta h ntherm nmeas seed   (checkpointed, resumable)"""
import numpy as np, sys, json, time, pickle, os
sys.path.insert(0, '/home/claude/w/lozenge-quantum-tiling/code')
from numba import njit
from dice_string import Tiling
from qmc_tiling import lieb
from sse_h import sweep, seed, EPS
@njit(cache=True)
def meas(spin, ops, b):
    n = 0; nod = 0
    for p in range(ops.shape[0]):
        if ops[p] != -1:
            n += 1
            if ops[p] % 2 == 1: nod += 1
    zz = 0.0
    for k in range(b.shape[0]): zz += 0.25 * spin[b[k, 0]] * spin[b[k, 1]]
    return n, nod, 0.5 * spin.sum(), zz
cj, kk, beta, h, ntherm, nmeas, sd = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]); key = os.path.basename(cj).split('.')[0] + f'_k{kk}'
ck = f'ck_{key}_{h}_{sd}.pkl'
sys.path.insert(0, '/home/claude/k2'); from cellgraph import cell_graph; from lswt import signs; _r = json.load(open(cj)); N, _b, _ = cell_graph(_r['cell'], _r['tiling'], kk); Sl = abs(int(np.sum(signs(N, _b)))) / 2; L = kk
b = np.ascontiguousarray(_b, np.int64); nb = len(b)
z = np.bincount(b.ravel(), minlength=N).astype(float)
hb = np.ascontiguousarray(np.stack([h / z[b[:, 0]], h / z[b[:, 1]]], 1)); Cb = 0.25 + 0.5 * (hb[:, 0] + hb[:, 1]) + EPS
if os.path.exists(ck):
    st = pickle.load(open(ck, 'rb')); spin, ops, n, nl, t, R = st['spin'], st['ops'], st['n'], st['nl'], st['t'], st['R']
    seed(sd * 1000 + t); np.random.seed(sd * 1000 + t)
else:
    np.random.seed(sd); seed(sd); spin = np.where(np.random.random(N) < 0.5, 1, -1).astype(np.int64)
    ops = np.full(max(40, int(beta * nb * 0.3)), -1, np.int64); n = 0; nl = 10; t = 0; R = []
last = time.time(); t0 = last
while t < ntherm + nmeas:
    n = sweep(spin, ops, b, beta, nb, n, Cb, hb, nl)
    if t < ntherm:
        M = len(ops); newM = int(1.3 * n) + 40
        if newM > M:
            new = np.full(newM, -1, np.int64); pos = np.sort(np.random.choice(newM, M, replace=False)); new[pos] = ops; ops = new
        if t > 20: nl = max(10, int(2 * n / int(os.environ.get('NLDIV', '10'))))
    else:
        R.append(meas(spin, ops, b))
    t += 1
    if time.time() - last > 90:
        pickle.dump(dict(spin=spin, ops=ops, n=n, nl=nl, t=t, R=R), open(ck + '.tmp', 'wb')); os.replace(ck + '.tmp', ck); last = time.time()
        print(f"t={t}/{ntherm+nmeas} {time.time()-t0:.0f}s", flush=True)
R = np.array(R, float); nbk = 20; R = R[:len(R) // nbk * nbk].reshape(nbk, -1, 4).mean(1)
E = (-R[:, 0] / beta + Cb.sum()) / N; Exy = -R[:, 1] / beta / N; mz = R[:, 2] / N; Ezee = -h * mz; Ezz = E - Exy - Ezee; Ezz_eq = R[:, 3] / N
f = lambda x: (float(x.mean()), float(x.std() / np.sqrt(nbk)))
print(json.dumps(dict(key=key, L=L, beta=beta, h=h, seed=sd, m_lieb=Sl / N, E=f(E), Exy=f(Exy), Ezz=f(Ezz), Ezz_eq=f(Ezz_eq), Ezee=f(Ezee), m=f(np.abs(mz)))), flush=True)
