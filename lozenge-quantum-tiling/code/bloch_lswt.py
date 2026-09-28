"""LSWT ground-state energy of a PERIODIC lozenge tiling in the thermodynamic limit (Bloch theorem).
mu(k) = eig( L(k)^{1/2} P L(k)^{1/2} ), L(k) the Bloch graph Laplacian of the supercell.
E/N = -J S^2 (N_b/N) + (JS/2N) ( <sum_nu |mu_nu(k)|>_k - sum_i z_i )."""
import numpy as np
from collections import deque
from periodic import Cell, DIRS


def cell_bonds(C, tiling):
    rem = set(tuple(x) for x in tiling); B = []
    for n in range(C.N):
        i, j = C.site(n)
        for d, (di, dj) in enumerate(DIRS):
            if (n, d) in rem: continue
            I, Jj = i + di, j + dj
            k2 = Jj // C.q; ii = I - k2 * C.s; k1 = ii // C.p
            B.append((n, C.idx(I, Jj), k1, k2))
    return B


def signs_cell(N, B):
    adj = [[] for _ in range(N)]
    for a, b, _, _ in B: adj[a].append(b); adj[b].append(a)
    c = np.zeros(N, int); c[0] = 1; q = deque([0]); seen = {0}
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen: seen.add(v); c[v] = -c[u]; q.append(v)
            elif c[v] == c[u]: return None
    return c


def energy(C, tiling, nk=24, S=0.5, J=1.0, return_omega=False):
    B = cell_bonds(C, tiling); N = C.N
    P = signs_cell(N, B); z = np.zeros(N)
    for a, b, _, _ in B: z[a] += 1; z[b] += 1
    tot = 0.0; om = []
    for t1 in range(nk):
        for t2 in range(nk):
            th1, th2 = 2 * np.pi * (t1 + 0.5) / nk, 2 * np.pi * (t2 + 0.5) / nk     # shifted grid avoids k=0 zero mode
            Lk = np.diag(z).astype(complex)
            for a, b, k1, k2 in B:
                ph = np.exp(1j * (th1 * k1 + th2 * k2))
                Lk[a, b] -= ph; Lk[b, a] -= np.conj(ph)
            w, V = np.linalg.eigh(Lk); w = np.clip(w, 0, None)
            Lh = (V * np.sqrt(w)) @ V.conj().T
            mu = np.linalg.eigvalsh(Lh @ (P[:, None] * Lh))
            tot += np.abs(mu).sum()
            if return_omega: om.append(J * S * np.abs(mu))
    tot /= nk * nk
    E = (-J * S * S * len(B) + 0.5 * J * S * (tot - z.sum())) / N
    m = abs(P.sum()) / 2 / N
    return (E, m, np.array(om)) if return_omega else (E, m)


if __name__ == '__main__':
    import json, sys
    from periodic import EPS, C_M
    targets = {}
    O = json.load(open('periodic_all_lswt.json'))
    import glob
    for f in glob.glob('big_*_s*.json'): O += json.load(open(f))
    # square & dice cells
    sq = Cell(3, 3, 0)
    res = []
    lin = None
    for name, filt in (('square', lambda r: r['hist'] == {'4': r['m'] * 0 + sum(r['hist'].values())} if False else None),):
        pass
    rows = [r for r in O]
    best = {}
    for r in rows:
        k = round(r['m'], 5)
        best.setdefault(k, []).append(r)
    out = []
    for k in sorted(best):
        L = sorted(best[k], key=lambda r: r['e_model'])[:6]
        seen = set()
        for r in L:
            key = (tuple(r['cell']), r['e_model'])
            if key in seen: continue
            seen.add(key)
            C = Cell(*r['cell'])
            E24, m = energy(C, r['tiling'], nk=24)
            out.append(dict(m=m, cell=r['cell'], hist=r['hist'], e_model=r['e_model'], e_inf=E24))
            print(f"m={m:.4f} cell={r['cell']} hist={r['hist']} model={r['e_model']:.5f} LSWT_inf={E24:.5f}", flush=True)
    json.dump(out, open('paper/bloch_lswt.json', 'w'), indent=1)
