"""Illustrative properties of the s6d3 superlattice (vs square and dice):
 (1) LSWT transverse dynamical structure factor S^xx(q, w)
 (2) emergent U(1) gauge field of the lozenge (dimer) tiling: height function / electric field
 (3) momentum-resolved entanglement spectrum of the Gaussian (LSWT) ground state for a cut parallel to the layers
"""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from tilt import strips
from lswt_fluct import bogoliubov
from lswt import signs

A1 = np.array([1.0, 0]); A2 = np.array([0.5, np.sqrt(3) / 2])


def structs(L):
    return {'square': strips(L, [[0, L]])[0],
            's6d3': strips(L, [[a, 6] for a in range(0, L, 9)])[0],
            'dice': strips(L, [])[0]}


# ---------------------------------------------------------------- (1) S^xx(q,w)
def sqw(T, qpath, wgrid, eta=0.03, S=0.5):
    B = T.bonds(); N = T.N
    om, U, V, z, keep = bogoliubov(N, B, S=S)
    om = om[keep]; C = U + V                         # amplitude of (a_i + a_i^dag) on b_n^dag
    ph = np.exp(-1j * qpath @ T.pos.T)                # (nq, N)
    amp = np.abs(ph @ C) ** 2 * S / 2 / N             # (nq, nmodes)
    lor = eta / np.pi / ((wgrid[:, None] - om[None, :]) ** 2 + eta ** 2)
    return amp @ lor.T, om                             # (nq, nw)


# ---------------------------------------------------------------- (2) gauge field
def efield(T):
    """dimer (lozenge) electric field on the honeycomb of triangles:
    E = sum over the 3 bonds of each up-triangle of (n_dimer - 1/3) * unit vector (up -> down triangle)."""
    L = T.L; E = []; P = []
    for t, sts in enumerate(T.tri_sites):
        if t % 2: continue                            # up triangles only (kind 0)
        c = T.tri_cent[t]; Et = np.zeros(2)
        for e in T.tri_edges[t]:
            a, b = T.edge_tris[e]; tp = b if a == t else a
            d = T.wrap(T.tri_cent[tp] - c); d /= np.linalg.norm(d)
            Et += ((e in T.removed) - 1 / 3) * d
        E.append(Et); P.append(c)
    return np.array(P), np.array(E)


# ---------------------------------------------------------------- (3) entanglement spectrum
def ent_spectrum(T, S=0.5):
    """cut parallel to the layers: region = g in [0, 1/2), g = frac. coord along (1,-1);
    translation t:(i,j)->(i+1,j+1) keeps the region and the s6d3 layers -> label by momentum k."""
    L = T.L; N = T.N
    om, U, V, z, keep = bogoliubov(N, T.bonds(), S=S)
    pass
    g = ((T.ij[:, 0] - T.ij[:, 1]) % L) / L
    reg = np.where(g < 0.5)[0]
    nij = V @ V.T; mij = U @ V.T; mij = 0.5 * (mij + mij.T)
    X = 0.5 * np.eye(N) + nij + mij; Pm = 0.5 * np.eye(N) + nij - mij
    # orbits under translation (1,1)
    idx = {tuple(x): n for n, x in enumerate(T.ij)}
    reps = [n for n in reg if T.ij[n, 0] == T.ij[n, 1] + ((T.ij[n, 0] - T.ij[n, 1]) % L) or True]
    # choose representatives: sites with i = 0 (each orbit (i+s, j+s) hits i=0 exactly once)
    reps = [n for n in reg if T.ij[n, 0] == 0]
    orb = np.array([[idx[((T.ij[r, 0] + s) % L, (T.ij[r, 1] + s) % L)] for s in range(L)] for r in reps])  # (nr, L)
    ks, eps = [], []
    for kk in range(L):
        k = 2 * np.pi * kk / L
        ph = np.exp(1j * k * np.arange(L))
        # X_k[a,b] = sum_s X[orb[a,0], orb[b,s]] e^{i k s}
        Xk = (X[orb[:, 0][:, None, None], orb[None, :, :]] * ph[None, None, :]).sum(-1)
        Pk = (Pm[orb[:, 0][:, None, None], orb[None, :, :]] * ph[None, None, :]).sum(-1)
        nu = np.sqrt(np.clip(np.linalg.eigvals(Xk @ Pk).real, 0.25 + 1e-12, None))
        e = np.log((nu + 0.5) / (nu - 0.5))
        ks += [kk if kk <= L // 2 else kk - L] * len(e); eps += list(e)
    return np.array(ks) * 2 * np.pi / L, np.array(eps)


if __name__ == "__main__":
    L = 36
    St = structs(L)
    for k, T in St.items():
        c = signs(T.N, T.bonds()); print(k, "N", T.N, "S_Lieb", abs(c.sum()) / 2)
    # ---- (1) S(q,w): path perpendicular to layers (along reciprocal of (1,-1) stacking) and parallel
    Bm = 2 * np.pi * np.linalg.inv(np.array([A1, A2])).T   # reciprocal of a1, a2 (rows)
    b_perp = (Bm[0] - Bm[1]) / 2                            # conjugate to g (stacking)
    b_par = (Bm[0] + Bm[1]) / 2
    sgrid = np.arange(-L, L + 1)
    t = sgrid / L
    qpaths = {'perp': np.outer(sgrid, (Bm[0] - Bm[1]) / L), 'par': np.outer(sgrid, (Bm[0] + Bm[1]) / L)}
    wg = np.linspace(0, 2.2, 300)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for col, (name, T) in enumerate(St.items()):
        for row, (pn, qp) in enumerate(qpaths.items()):
            I, om = sqw(T, qp, wg)
            ax = axs[row, col]
            ax.imshow(np.log10(I.T + 1e-3), origin='lower', aspect='auto', extent=[t[0], t[-1], wg[0], wg[-1]], cmap='magma', vmin=-2.5, vmax=1)
            ax.set_title(f'{name}: q {"⊥" if pn == "perp" else "∥"} layers', fontsize=10)
            ax.set_xlabel('q · r / 2π per row  (s/L)'); [ax.axvline(v, color='w', lw=.4, ls=':') for v in np.arange(-1, 1.01, 1/9 if pn=='perp' else 1/3)] if name=='s6d3' else None; ax.set_ylabel('ω / J')
    fig.suptitle(f'LSWT transverse dynamical structure factor log10 S^xx(q,ω), S=½, L={L}', fontsize=11)
    plt.tight_layout(); plt.savefig('figures/s6d3_Sqw.png', dpi=130)
    print("Sqw done")
