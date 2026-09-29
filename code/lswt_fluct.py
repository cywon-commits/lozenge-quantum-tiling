"""Quantum-fluctuation measures of the LSWT ground state on a bipartite graph (Colpa diagonalisation).
Returns per-site spin deviation n_i = <a_i^dag a_i> (moment reduction dS_i), single-site entanglement
entropy (Gaussian), zero-point energy, and the entanglement entropy of a region (Gaussian state).
Zero modes (omega < wc) are excluded, as usual for finite-size LSWT."""
import numpy as np
from lswt import signs


def bogoliubov(N, bonds, S=0.5, J=1.0, eps=1e-7, wc=1e-2):
    bonds = np.asarray(bonds)
    Adj = np.zeros((N, N)); Adj[bonds[:, 0], bonds[:, 1]] = 1; Adj[bonds[:, 1], bonds[:, 0]] = 1
    z = Adj.sum(1)
    A = J * S * np.diag(z) + eps * np.eye(N); B = J * S * Adj
    M = np.block([[A, B], [B, A]])
    K = np.linalg.cholesky(M).T          # M = K^T K
    s3 = np.diag(np.r_[np.ones(N), -np.ones(N)])
    W = K @ s3 @ K.T
    lam, U = np.linalg.eigh(W)
    order = np.r_[np.argsort(-lam)[:N], np.argsort(lam)[:N]]   # N positive (desc), N negative (asc)
    lam = lam[order]; U = U[:, order]
    Ed = s3 @ np.diag(lam)                 # = diag(omega, omega)
    om = np.diag(Ed)[:N]
    T = np.linalg.solve(K, U @ np.diag(np.sqrt(np.diag(Ed))))
    Uu = T[:N, :N]; V = T[:N, N:]
    keep = om > wc
    return om, Uu[:, keep], V[:, keep], z, keep


def fluct(N, bonds, region=None, S=0.5):
    om, Uu, V, z, keep = bogoliubov(N, bonds, S=S)
    n = (V ** 2).sum(1)                      # spin deviation per site
    sent = (n + 1) * np.log(n + 1) - np.where(n > 0, n * np.log(np.where(n > 0, n, 1)), 0)
    out = dict(n=n, s_site=sent, z=z, nzero=int((~keep).sum()))
    if region is not None:
        r = np.asarray(region)
        nij = V[r] @ V[r].T; mij = Uu[r] @ V[r].T
        mij = 0.5 * (mij + mij.T)
        X = 0.5 * np.eye(len(r)) + nij + mij; P = 0.5 * np.eye(len(r)) + nij - mij
        nu = np.sqrt(np.clip(np.linalg.eigvals(X @ P).real, 0.25, None))
        out['S_region'] = float(np.sum((nu + .5) * np.log(nu + .5) - np.where(nu > .5 + 1e-12, (nu - .5) * np.log(np.maximum(nu - .5, 1e-300)), 0)))
    return out
