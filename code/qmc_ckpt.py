"""Checkpointed SSE (S=1/2, field) on a pickled tiling. Resumable: state saved every ~90 s."""
import numpy as np, sys, json, time, pickle, os
from dice_string import Tiling
from qmc_tiling import lieb
from sse_h import sweep, measure, seed, EPS
pk, L, beta, h, ntherm, nmeas, sd = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
ck = f'ckpt_{os.path.basename(pk)}_{h}_{sd}.pkl'
T = Tiling(L); T.removed = pickle.load(open(pk, 'rb')); S = lieb(T); N = T.N
b = np.ascontiguousarray(T.bonds(), np.int64); nb = len(b)
z = np.bincount(b.ravel(), minlength=N).astype(float)
hb = np.ascontiguousarray(np.stack([h / z[b[:, 0]], h / z[b[:, 1]]], 1)); Cb = 0.25 + 0.5 * (hb[:, 0] + hb[:, 1]) + EPS
if os.path.exists(ck):
    st = pickle.load(open(ck, 'rb')); spin, ops, n, nl, t, Es, Ms = st['spin'], st['ops'], st['n'], st['nl'], st['t'], st['E'], st['M']
    seed(sd * 1000 + t); np.random.seed(sd * 1000 + t)
else:
    np.random.seed(sd); seed(sd); spin = np.where(np.random.random(N) < 0.5, 1, -1).astype(np.int64)
    ops = np.full(max(40, int(beta * nb * 0.3)), -1, np.int64); n = 0; nl = 10; t = 0; Es = []; Ms = []
t0 = time.time(); last = t0
while t < ntherm + nmeas:
    n = sweep(spin, ops, b, beta, nb, n, Cb, hb, nl)
    if t < ntherm:
        M = len(ops); newM = int(1.3 * n) + 40
        if newM > M:
            new = np.full(newM, -1, np.int64); pos = np.sort(np.random.choice(newM, M, replace=False)); new[pos] = ops; ops = new
        if t > 20: nl = max(10, int(2 * n / 10))
    else:
        sg, mz, nn = measure(spin, ops); Es.append(-nn / beta + Cb.sum()); Ms.append(mz)
    t += 1
    if time.time() - last > 90:
        pickle.dump(dict(spin=spin, ops=ops, n=n, nl=nl, t=t, E=Es, M=Ms), open(ck + '.tmp', 'wb')); os.replace(ck + '.tmp', ck); last = time.time()
        print(f"t={t}/{ntherm+nmeas}", flush=True)
E = np.array(Es); Mz = np.array(Ms); nbk = 20
Eb = E[:len(E)//nbk*nbk].reshape(nbk, -1).mean(1) / N; Mb = Mz[:len(Mz)//nbk*nbk].reshape(nbk, -1).mean(1) / N
print(json.dumps(dict(L=L, kind='pkl', pkl=pk, beta=beta, h=h, S_lieb=S, e=Eb.mean(), e_err=Eb.std() / np.sqrt(nbk), m=float(np.abs(Mb).mean()))), flush=True)
