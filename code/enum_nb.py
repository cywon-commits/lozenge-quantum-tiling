"""Numba enumeration of ALL periodic lozenge tilings of a supercell Cell(p,q,s), keeping
structurally distinct representatives per (imbalance, z-histogram, local-environment hash).

Compared with per_big.py (pure Python, keeps top-K per m by the coordination model, which
depends only on the z-histogram), this version
  * is ~100x faster (full DFS + analysis inside numba),
  * keeps one representative per translation-invariant structural class, so tilings with the
    same z-histogram but different arrangement survive and can be re-ranked by Bloch LSWT.

Class key: (D = |N_A - N_B|, n3, n4, n5, n6, H) with H = sum_i w[z_i, n3_i, n4_i, n5_i, n6_i],
where n_k,i = number of neighbours of site i with coordination k and w are fixed random
64-bit weights (translation invariant; collisions are harmless, they only merge classes).

Usage (run from the data directory):
    python enum_nb.py P Q S [--maxkeys 400000] [--keep 0.01]   (--keep: J/site window above the best e_model per m)
    python enum_nb.py P Q all            # all shifts s compatible with the L=36 torus
Output: cand_{P}x{Q}_s{S}.json  (list of dicts: cell, tiling, m, e_model, hist, key, count)
"""
import numpy as np, sys, json, time
from numba import njit
from numba.typed import Dict
from numba import types
from periodic import Cell, EPS, C_M, DIRS

LTORUS = 36
KEYT = types.UniTuple(types.int64, 6)


def build(C):
    N = C.N
    nbr = np.zeros((N, 3), np.int64)               # forward neighbour of site n in direction d, edge id n*3+d
    for n in range(N):
        i, j = C.site(n)
        for d, (di, dj) in enumerate(DIRS):
            nbr[n, d] = C.idx(i + di, j + dj)
    up = np.zeros((N, 3), np.int64); dn = np.zeros((N, 3), np.int64)
    for n in range(N):
        i, j = C.site(n)
        e = [(C.idx(i, j), 0), (C.idx(i, j), 1), (C.idx(i + 1, j), 2)]
        dd = [C.idx(i, j - 1), C.idx(i - 1, j), C.idx(i, j)]
        for k in range(3):
            up[n, k] = e[k][0] * 3 + e[k][1]; dn[n, k] = dd[k]
    # 6 incident edges per site: (neighbour, edge id)
    inc_n = np.zeros((N, 6), np.int64); inc_e = np.zeros((N, 6), np.int64); cnt = np.zeros(N, np.int64)
    for n in range(N):
        for d in range(3):
            m = nbr[n, d]; eid = n * 3 + d
            inc_n[n, cnt[n]] = m; inc_e[n, cnt[n]] = eid; cnt[n] += 1
            inc_n[m, cnt[m]] = n; inc_e[m, cnt[m]] = eid; cnt[m] += 1
    assert (cnt == 6).all(), "cell too small (repeated neighbours)"
    return up, dn, inc_n, inc_e


@njit(cache=True)
def _enum(N, up, dn, inc_n, inc_e, W, eps, cm, maxkeys, store, keys_arr, cnt_arr, nkeys_out):
    used = np.zeros(N, np.bool_); rem = np.zeros(3 * N, np.bool_)
    ch = np.zeros(N, np.int64); cd = np.zeros(N, np.int64); ptr = np.zeros(N + 1, np.int64)
    z = np.zeros(N, np.int64); col = np.zeros(N, np.int64); queue = np.zeros(N, np.int64)
    d_ = Dict.empty(key_type=KEYT, value_type=types.int64)
    ntil = 0; nbip = 0; nk = 0; overflow = False
    u = 0; ptr[0] = 0
    while u >= 0:
        if u == N:
            ntil += 1
            # ---- analyse complete tiling
            ok = True
            for n in range(N):
                c = 0
                for t in range(6):
                    if not rem[inc_e[n, t]]:
                        c += 1
                z[n] = c
                if c < 3:
                    ok = False
            if ok:
                for n in range(N):
                    col[n] = 0
                col[0] = 1; queue[0] = 0; qh = 0; qt = 1
                while qh < qt and ok:
                    a = queue[qh]; qh += 1
                    for t in range(6):
                        if rem[inc_e[a, t]]:
                            continue
                        b = inc_n[a, t]
                        if col[b] == 0:
                            col[b] = -col[a]; queue[qt] = b; qt += 1
                        elif col[b] == col[a]:
                            ok = False; break
                if qt < N:
                    ok = False
            if ok:
                nbip += 1
                D = 0; h3 = 0; h4 = 0; h5 = 0; h6 = 0; H = np.int64(0)
                for n in range(N):
                    D += col[n]
                    zz = z[n]
                    if zz == 3: h3 += 1
                    elif zz == 4: h4 += 1
                    elif zz == 5: h5 += 1
                    else: h6 += 1
                    c3 = 0; c4 = 0; c5 = 0; c6 = 0
                    for t in range(6):
                        if rem[inc_e[n, t]]:
                            continue
                        zb = z[inc_n[n, t]]
                        if zb == 3: c3 += 1
                        elif zb == 4: c4 += 1
                        elif zb == 5: c5 += 1
                        else: c6 += 1
                    H += W[zz, c3, c4, c5, c6]
                if D < 0: D = -D
                key = (np.int64(D), np.int64(h3), np.int64(h4), np.int64(h5), np.int64(h6), H)
                if key in d_:
                    cnt_arr[d_[key]] += 1
                elif nk < maxkeys:
                    d_[key] = nk
                    for n in range(N):
                        store[nk, n] = ch[n]
                    keys_arr[nk, 0] = D; keys_arr[nk, 1] = h3; keys_arr[nk, 2] = h4
                    keys_arr[nk, 3] = h5; keys_arr[nk, 4] = h6; keys_arr[nk, 5] = H
                    cnt_arr[nk] = 1; nk += 1
                else:
                    overflow = True
            u -= 1
            if u >= 0:
                used[cd[u]] = False; rem[ch[u]] = False
            continue
        k = ptr[u]; found = False
        while k < 3:
            dd = dn[u, k]; e = up[u, k]; k += 1
            if not used[dd]:
                used[dd] = True; rem[e] = True; ch[u] = e; cd[u] = dd; found = True; break
        ptr[u] = k
        if found:
            u += 1
            if u < N:
                ptr[u] = 0
        else:
            ptr[u] = 0; u -= 1
            if u >= 0:
                used[cd[u]] = False; rem[ch[u]] = False
    nkeys_out[0] = nk; nkeys_out[1] = ntil; nkeys_out[2] = nbip; nkeys_out[3] = 1 if overflow else 0


def run(p, q, s, maxkeys=400_000, keep=0.01):
    C = Cell(p, q, s); N = C.N
    up, dn, inc_n, inc_e = build(C)
    if q > p:   # DFS along the short cell direction first (keeps the search frontier narrow; 3x12, 4x12 were ~10x slower)
        perm = np.array(sorted(range(N), key=lambda n: (n % q, n // q)))
        up, dn = np.ascontiguousarray(up[perm]), np.ascontiguousarray(dn[perm])
    W = np.random.default_rng(12345).integers(-2**62, 2**62, size=(7, 7, 7, 7, 7), dtype=np.int64)
    eps = np.array([0, 0, 0, EPS[3], EPS[4], EPS[5], EPS[6]])
    store = np.zeros((maxkeys, N), np.int64); keys = np.zeros((maxkeys, 6), np.int64)
    cnt = np.zeros(maxkeys, np.int64); info = np.zeros(4, np.int64)
    t0 = time.time()
    _enum(N, up, dn, inc_n, inc_e, W, eps, C_M, maxkeys, store, keys, cnt, info)
    nk, ntil, nbip, ovf = info
    out = []
    for r in range(nk):
        D, h3, h4, h5, h6, H = keys[r]
        m = D / 2 / N
        e = (h3 * EPS[3] + h4 * EPS[4] + h5 * EPS[5] + h6 * EPS[6]) / N + C_M * m
        hist = {str(k): int(v) for k, v in zip((3, 4, 5, 6), (h3, h4, h5, h6)) if v}
        til = [[int(x // 3), int(x % 3)] for x in store[r]]
        out.append(dict(cell=[p, q, s], tiling=til, m=m, e_model=e, hist=hist, key=int(H), count=int(cnt[r])))
    emin = {}
    for r in out: emin[r['m']] = min(emin.get(r['m'], 1e9), r['e_model'])
    out = [r for r in out if r['e_model'] <= emin[r['m']] + keep]      # drop classes far above the best of their m
    return out, dict(cell=[p, q, s], N=N, tilings=int(ntil), bipartite=int(nbip), classes=int(nk),
                     overflow=bool(ovf), seconds=round(time.time() - t0, 1))


def shifts(p, q, L=LTORUS):
    return [s for s in range(p) if ((L // q) * s) % p == 0]


if __name__ == "__main__":
    p, q = int(sys.argv[1]), int(sys.argv[2])
    maxkeys = int(sys.argv[sys.argv.index('--maxkeys') + 1]) if '--maxkeys' in sys.argv else 400_000
    keep = float(sys.argv[sys.argv.index('--keep') + 1]) if '--keep' in sys.argv else 0.01
    assert LTORUS % p == 0 and LTORUS % q == 0, "cell must fit the L=36 torus"
    ss = shifts(p, q) if sys.argv[3] == 'all' else [int(sys.argv[3])]
    for s in ss:
        out, info = run(p, q, s, maxkeys, keep)
        json.dump(out, open(f'cand_{p}x{q}_s{s}.json', 'w'))
        with open('enum_log.jsonl', 'a') as f: f.write(json.dumps(info) + '\n')
        print(json.dumps(info), flush=True)
