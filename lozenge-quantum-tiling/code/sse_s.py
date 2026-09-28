"""SSE for the spin-S Heisenberg AF with a uniform field,  H = J sum S_i.S_j - h sum S^z_i,
on an arbitrary bipartite graph (sign-free).  Directed loops with heat-bath exit probabilities.
Site state n = 0..2S  (m = n - S).  Field distributed over bonds as h/z_i per site.
Bond operator H'_b = C_b - [J m_i m_j - h_i m_i - h_j m_j] - (J/2)(S+S- + S-S+),  C_b = J S^2 + (h_i+h_j) S + eps.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def wvert(n0, n1, n2, n3, S2, C, hi, hj):
    # legs: 0 lower i, 1 lower j, 2 upper i, 3 upper j ; values 0..S2 (=2S)
    if n0 < 0 or n1 < 0 or n2 < 0 or n3 < 0 or n0 > S2 or n1 > S2 or n2 > S2 or n3 > S2:
        return 0.0
    S = 0.5 * S2
    if n0 == n2 and n1 == n3:
        mi = n0 - S; mj = n1 - S
        return C - mi * mj + hi * mi + hj * mj
    if n2 + n3 != n0 + n1:
        return 0.0
    if n2 == n0 + 1 and n3 == n1 - 1:
        mi = n0 - S; mj = n1 - S
        return 0.5 * np.sqrt(S * (S + 1) - mi * (mi + 1)) * np.sqrt(S * (S + 1) - mj * (mj - 1))
    if n2 == n0 - 1 and n3 == n1 + 1:
        mi = n0 - S; mj = n1 - S
        return 0.5 * np.sqrt(S * (S + 1) - mi * (mi - 1)) * np.sqrt(S * (S + 1) - mj * (mj + 1))
    return 0.0


@njit(cache=True)
def seed(s):
    np.random.seed(s)


@njit(cache=True)
def sweep(spin, ops, bsites, beta, nb, n_ops, Cb, hb, S2, nloops):
    N = spin.shape[0]; M = ops.shape[0]
    for p in range(M):
        op = ops[p]
        if op == -1:
            b = np.random.randint(nb); i = bsites[b, 0]; j = bsites[b, 1]
            w = wvert(spin[i], spin[j], spin[i], spin[j], S2, Cb[b], hb[b, 0], hb[b, 1])
            if np.random.random() * (M - n_ops) < beta * nb * w:
                ops[p] = 2 * b; n_ops += 1
        elif op % 2 == 0:
            b = op // 2; i = bsites[b, 0]; j = bsites[b, 1]
            w = wvert(spin[i], spin[j], spin[i], spin[j], S2, Cb[b], hb[b, 0], hb[b, 1])
            if np.random.random() * beta * nb * w < (M - n_ops + 1):
                ops[p] = -1; n_ops -= 1
        else:
            # off-diagonal: stored as 2b+1 together with leg values in 'legs' array is not available here;
            # we keep off-diagonal operators' effect in the leg list, so propagate via stored delta
            pass
    return n_ops


@njit(cache=True)
def full_sweep(spin, ops, dlt, bsites, beta, nb, n_ops, Cb, hb, S2, nloops, nodiag=False):
    """ops[p] = -1 (identity) or 2b (diag) or 2b+1 (offdiag); dlt[p] = change of spin i (+1/-1) for offdiag."""
    N = spin.shape[0]; M = ops.shape[0]
    # diagonal update with propagation
    for p in range(M):
        op = ops[p]
        if nodiag:
            break
        if op == -1:
            b = np.random.randint(nb); i = bsites[b, 0]; j = bsites[b, 1]
            w = wvert(spin[i], spin[j], spin[i], spin[j], S2, Cb[b], hb[b, 0], hb[b, 1])
            if np.random.random() * (M - n_ops) < beta * nb * w:
                ops[p] = 2 * b; n_ops += 1
        elif op % 2 == 0:
            b = op // 2; i = bsites[b, 0]; j = bsites[b, 1]
            w = wvert(spin[i], spin[j], spin[i], spin[j], S2, Cb[b], hb[b, 0], hb[b, 1])
            if np.random.random() * beta * nb * w < (M - n_ops + 1):
                ops[p] = -1; n_ops -= 1
        else:
            b = op // 2; spin[bsites[b, 0]] += dlt[p]; spin[bsites[b, 1]] -= dlt[p]
    # vertex list
    link = np.full(4 * M, -1, np.int64); leg = np.zeros(4 * M, np.int64)
    first = np.full(N, -1, np.int64); last = np.full(N, -1, np.int64)
    cur = spin.copy()
    for p in range(M):
        op = ops[p]
        if op == -1: continue
        b = op // 2; i = bsites[b, 0]; j = bsites[b, 1]
        leg[4 * p] = cur[i]; leg[4 * p + 1] = cur[j]
        if op % 2 == 1:
            cur[i] += dlt[p]; cur[j] -= dlt[p]
        leg[4 * p + 2] = cur[i]; leg[4 * p + 3] = cur[j]
        for k in range(2):
            s = bsites[b, k]; vl = 4 * p + k; vu = 4 * p + 2 + k
            if last[s] != -1:
                link[last[s]] = vl; link[vl] = last[s]
            else:
                first[s] = vl
            last[s] = vu
    for s in range(N):
        if first[s] != -1:
            link[first[s]] = last[s]; link[last[s]] = first[s]
    if n_ops > 0:
        pr = np.zeros(4); dx = np.zeros(4, np.int64)
        for _ in range(nloops):
            j0 = np.random.randint(4 * M)
            while ops[j0 // 4] == -1:
                j0 = np.random.randint(4 * M)
            d0 = 1 if np.random.random() < 0.5 else -1
            if leg[j0] + d0 < 0 or leg[j0] + d0 > S2:
                continue
            j = j0; d = d0; steps = 0
            while True:
                p = j // 4; l = j % 4; base = 4 * p; bb = ops[p] // 2
                leg[base + l] += d
                tot = 0.0
                for x in range(4):
                    if x == l:
                        dx[x] = -d
                    elif (x // 2) == (l // 2):
                        dx[x] = -d
                    else:
                        dx[x] = d
                    leg[base + x] += dx[x]
                    w = wvert(leg[base], leg[base + 1], leg[base + 2], leg[base + 3], S2, Cb[bb], hb[bb, 0], hb[bb, 1])
                    leg[base + x] -= dx[x]
                    pr[x] = w; tot += w
                r = np.random.random() * tot; x = 0; acc = pr[0]
                while acc < r:
                    x += 1; acc += pr[x]
                leg[base + x] += dx[x]
                jn = link[base + x]; dn = dx[x]
                steps += 1
                if (jn == j0 and dn == d0) or (base + x == j0 and dn == -d0) or steps > 200 * M:
                    break
                j = jn; d = dn
        for p in range(M):
            op = ops[p]
            if op == -1: continue
            b = op // 2; base = 4 * p
            if leg[base] == leg[base + 2] and leg[base + 1] == leg[base + 3]:
                ops[p] = 2 * b; dlt[p] = 0
            else:
                ops[p] = 2 * b + 1; dlt[p] = leg[base + 2] - leg[base]
    for s in range(N):
        if first[s] != -1:
            spin[s] = leg[first[s]]
        else:
            spin[s] = np.random.randint(S2 + 1)
    return n_ops


def run(bonds, N, beta, S=1.0, h=0.0, ntherm=2000, nmeas=10000, seed_=0, eps=0.25, init=None):
    b = np.ascontiguousarray(bonds, np.int64); nb = len(b); S2 = int(round(2 * S))
    z = np.bincount(b.ravel(), minlength=N).astype(float)
    hb = np.ascontiguousarray(np.stack([h / z[b[:, 0]], h / z[b[:, 1]]], 1))
    Cb = S * S + (hb[:, 0] + hb[:, 1]) * S + eps
    np.random.seed(seed_); seed(seed_)
    spin = (np.random.randint(0, S2 + 1, N) if init is None else init).astype(np.int64)
    M = max(40, int(beta * nb * 0.5)); ops = np.full(M, -1, np.int64); dlt = np.zeros(M, np.int64); n = 0
    nl = 10
    for t in range(ntherm):
        n = full_sweep(spin, ops, dlt, b, beta, nb, n, Cb, hb, S2, nl)
        newM = int(1.3 * n) + 40
        if newM > M:
            pos = np.sort(np.random.choice(newM, M, replace=False))
            o2 = np.full(newM, -1, np.int64); d2 = np.zeros(newM, np.int64); o2[pos] = ops; d2[pos] = dlt
            ops, dlt, M = o2, d2, newM
        if t > 20: nl = max(10, int(2 * n / (10 * max(1, S2))))
    E = np.empty(nmeas); Mz = np.empty(nmeas)
    for t in range(nmeas):
        n = full_sweep(spin, ops, dlt, b, beta, nb, n, Cb, hb, S2, nl)
        E[t] = -n / beta + Cb.sum(); Mz[t] = (spin - 0.5 * S2).sum()
    return E, Mz
