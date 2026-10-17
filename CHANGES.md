# Changes in this update (manuscript v7, 2026-10-17)

Unzip over the repository root. Files are only added or replaced; nothing is deleted.

## manuscript/
- `main.tex`, `main.pdf`: v7. Five authors (Suhwan Yoo, Jeehoon Choi, Chaeyeon Kim, Hee Young Kim, Changyeon Won), NRF acknowledgment (NRF-2023R1A2C1006050), APS-length abstract, Appendix C removed (appendices are now A = tests, B = LSWT hull), new Sec. V C (defect lattice gas).
- `figs/`: all figures regenerated; new `fig11_defectgas.pdf`. Fig. 2(a) now shows eps(z) minus the dice chord.

## Physics changes behind v7
- Square-type reference: e_sq = -0.669437(5) J (thermodynamic limit, Sandvik 1997) instead of the L = 24 value -0.66917(30) J. Bloch LSWT shows that crystals with an extensive moment have no global finite-size term, so only the compensated reference needed the correction.
- M/6 and M/3 crystals replaced by the LSWT-best structures from the class-resolved enumeration (`data/qnew/lswtbest_M{6,3}_L36.pkl`), QMC at L = 36 with 3 (M/6) and 2 (M/3) seeds.
- delta e (1e-3 J): M/6 -3.13(11), M/4 -2.77(31), M/3 -4.53(9), M/2 -4.98(9), 2M/3 -4.37(12).
- Step fields: 0.085, 0.139, 0.173, 0.211, 0.268 J. `stair_update.py` on its 0.0001 J grid prints 0.174 and 0.212. The manuscript quotes the exact chord slopes, which are 0.173 and 0.211.
- Bootstrap (4000 samples): every plateau is present in 100 % of samples.
- Enumeration up to 48-site cells found no new hull vertices.

## code/ (new or replaced)
- `paper_figs.py` (new reference, new Fig. 2a), `paper_data_v4.py` (inserts the new crystals into `paper/structs_L36.pkl`, recomputes fluctuations), `ce_fig.py` (Fig. 11).
- `enum_nb.py` (numba class-resolved enumeration), `rank_bloch.py`, `make_qmc.py`, `stair_update.py` (all run from `data/`), `qnew_run.sh`.
- `ce/`: cluster-expansion (defect lattice gas) scripts; run them from `data/ce/`.
- Not changed, but still using the old reference -0.66917: `big_fig.py`, `final_c.py`, `fit_qmc_model.py`, `fit_scaled.py`, `hull_qmc_b.py`. They are not used for the v7 staircase numbers. `fit_scaled.py` and `hull_qmc_b.py` feed only the model-rescaling panel and the irregular-network circles, whose shift is far below the plotted scale.

## data/ (new or replaced)
- `big_staircase.json`, `paper/stair_boot.json`, `paper/structs_L36.pkl`, `paper/fluct_structs.json`, `paper/curvature.json`, `paper/defectgas.json`.
- `qnew/`: new crystals, QMC logs, and `stair/stair_esqTD.json` + png. QMC checkpoints are not included.
- `ce/`: enumeration candidates for all cells of 36 sites or fewer plus the 48-site cells (`cand_*.json`), Bloch-LSWT energies, cluster-expansion fit and hull.

## Not yet updated
- `slides/` still uses the old step fields and crystals.
