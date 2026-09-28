# Quantum order by disorder among lozenge tilings

Code and data for the manuscript

> **Quantum order by disorder among lozenge tilings: a staircase of hub-free crystals between the Lieb ferrimagnet and the antiferromagnet**
> Changyeon Won, Department of Physics, Kyung Hee University

The manuscript (`manuscript/main.pdf`) studies the spin-S Heisenberg antiferromagnet on the bipartite networks defined by lozenge tilings of the triangular lattice. It covers the dice lattice, square-type tilings and everything in between. All tilings are degenerate classically. Quantum fluctuations and a magnetic field lift this degeneracy and select a staircase of periodic crystals (M_dice/6, M_dice/3, hub-free M_dice/2, 2M_dice/3).

**한국어 요약.** 로젠지 타일링이 만드는 이분 격자 위 Heisenberg 반강자성체의 논문 코드와 데이터입니다. 구성 요소는 SSE 양자 몬테카를로(스핀 S, 자기장), 정확 대각화, 임의 그래프 스핀파 이론(실공간·Bloch), 주기 타일링 전수 열거, 어닐링, 타일링 몬테카를로, 그리고 곡률(높이 표현) 분석입니다. 모든 스크립트는 `data/` 폴더에서 실행합니다(아래 참조).

---

## Repository layout

```
code/         all Python modules and scripts (flat, they import each other by name), shell job runners
data/         working directory: every input/output file the scripts read or write
  paper/      data files used for the final manuscript figures (+ paper/figs/ output folder)
  hist_cells/ per-supercell histogram scans (cells <= 27 sites), from per_hist.py / per_cell.py
  cell_scan/  per-supercell scans with LSWT re-ranking (cells <= 27 sites), from per_cell.py
  figures/    output folder for exploratory figures (was an absolute path during the project)
manuscript/   main.tex (REVTeX 4.2), main.pdf, figs/ (final figure PDFs)
slides/       presentation deck sources (deck/: deck.json + one HTML file per slide) and slide images
```

## Requirements

Python >= 3.10 with `numpy`, `scipy`, `numba`, `matplotlib` (see `requirements.txt`). LaTeX with `revtex4-2` for the manuscript.

```bash
pip install -r requirements.txt
```

## How to run

Every script expects the **current directory to be `data/`**. Python puts the script's own folder (`code/`) on the import path, so the modules find each other:

```bash
cd data
python ../code/stair_boot.py          # example: bootstrap of the magnetization staircase
python ../code/paper_figs.py fig5     # regenerate one manuscript figure into data/paper/figs/
python ../code/paper_figs.py          # all manuscript figures
```

The first call of a numba-accelerated module compiles it (a few seconds). The QMC runs at L = 36 take one to two hours each. `qmc_ckpt.py` checkpoints every 90 s and resumes automatically (see `run_ckpt.sh`, `run_m18.sh`).

---

## Core modules

| module | content |
|---|---|
| `dice_string.py` | `Tiling(L)`: L×L triangular torus, lozenge tiling as a set of removed edges, monomer (string) creation/moves; classical spin minimiser |
| `tilt.py` | `strips(L, intervals)`: dice / square-type strip tilings (tilt family); CLI runs QMC on them |
| `tiling_moves.py` | hexagon flips, coordination numbers `degrees(T)` |
| `periodic.py` | supercells `Cell(p,q,s)`, exhaustive enumeration of periodic tilings, coordination model `EPS`, `C_M`, `replicate` to an L-torus |
| `lswt.py` | linear spin-wave energy on an arbitrary bipartite graph, Lieb sign structure |
| `lswt_fluct.py` | Colpa diagonalisation: spin deviations, Gaussian entanglement, modes |
| `bloch_lswt.py` | LSWT of a periodic tiling in the thermodynamic limit (Bloch theorem on the supercell) |
| `sse.py`, `sse_h.py` | S = 1/2 stochastic series expansion QMC with heat-bath directed loops (without / with field) |
| `sse_s.py` | spin-S SSE with field (validated against ED for S = 1/2, 1, 3/2) |
| `qmc_ckpt.py` | checkpointed, resumable S = 1/2 SSE run on a pickled tiling |
| `ed_dice.py` | Lanczos exact diagonalisation on tori (27-site dice clusters, fixed S^z sectors) |
| `tiling_mc.py` | finite-temperature Monte Carlo of the tiling with the QMC-scaled coordination free energy |
| `lift.py` | lift a tiling to its stepped surface in Z^3; discrete Gaussian curvature K = (π/2)(4 − z); staggered-curvature identity |

## Scripts by manuscript section / figure

| manuscript | scripts | main data |
|---|---|---|
| Fig. 1 tilings | `paper_figs.py fig1`, `paper_data.py` (builds `paper/structs_L36.pkl`) | `paper/structs_L36.pkl`, `ann_*.pkl` |
| Fig. 2 coordination model | `degfit.py`, `degfit_m.py`, `fit_scaled.py`, `paper_figs.py fig2` | `degfit_m.npz`, `scaled_model.json`, QMC logs |
| Fig. 3 tilt family, walls | `tilt.py`, `walls.py`, `fig_tilt.py`, `paper_figs.py fig3` | `q*.log`, `r_*.log`, `walls.json` |
| Sec. IV B moment cost / hub-free bound | `per_hist.py`, `per_cell.py` | `hist_cells/`, `cell_scan/` |
| Figs. 4–5, Tables I & III crystals and staircase | `periodic.py`, `per_big.py`, `big_lswt.py`, `qmc_pkl.py`, `qmc_ckpt.py`, `big_fig.py`, `stair_boot.py`, `paper_figs.py fig4 fig5` | `big_*.json`, `big_m*_L36.pkl`, `qbig_*.log`, `qc.log`, `big_staircase.json`, `paper/stair_boot.json` |
| irregular networks | `anneal.py` (LSWT annealing), `fast_anneal.py` (two-stage), `hull_qmc_b.py` | `ann_*.pkl`, `fa_*.pkl`, `hull_fa_*.json`, `hull_qmc_b.json` |
| Fig. 6 fluctuations/entanglement | `paper_data.py`, `lswt_fluct.py`, `paper_figs.py fig6` | `paper/fluct_structs.json` |
| Fig. 7 S(q, ω) | `paper_data.py`, `paper_figs.py fig7` | `paper/sqw_structs.npz` |
| Fig. 8, Table II spin S | `sse_s.py`, `qmc_s.py`, `spinS_analysis.py`, `paper_figs.py fig8` | `qs_S.log`, `spinS.json` |
| Fig. 9 finite T | `tiling_mc.py`, `melt.py`, `paper_figs.py fig9` | `melt.json`, `paper/Tm.json` |
| Fig. 10, Sec. VIII curvature | `lift.py`, `curv_fig.py` | `paper/curvature.json`, `paper/pleated_L36.pkl` |
| App. A LSWT hull | `hull_fa.py`, `paper_figs.py figA1` | `hull_fa_*.json` |
| App. B tests (Fig. A2) | `fit_pair.py`, `wall_d.py`, `wall_d2.py`, `cool_test.py`, `magnonF.py`, `magnonF_bloch.py`, `paper_figs.py figA2` | `paper/model_test.json`, `paper/wall_d*.json`, `paper/cool_test.json`, `paper/magnonF*.json` |
| App. C classical strings | `run_h0.py`, `fit_h0.py`, `torus.py`, `curved.py`, `fig.py`, `fig_curved.py` | `comp_*.pkl`, `eq*.pkl`, `r*_h0.1.pkl`, `curved.pkl` |
| ED (27 sites), windings, U-strings | `ed_dice.py`, `run_ed.py`, `wind.py`, `run_wind.py`, `hi_sectors.py`, `field_curves.py`, `ushape.py`, `run_u.py`, `u_analyze.py`, `fig_u.py` | `ed_*.log`, `ed_nh*.pkl`, `hi_sectors.pkl`, `field_curves.pkl` |
| slides | `slide_figs.py`, `curv_fig.py` (`slide_fig`) | writes `slides/img/` |

Other files in `code/`:
- `t_*.py`, `test_*.py`: small validation and debugging scripts written during development. Examples are ED-vs-QMC checks, loop-closure tests and LSWT checks.
- `*.sh`, `jobs*.txt`: batch runners. They `cd` into `data/`.

## Data conventions

- **QMC logs** (`*.log`): one JSON object per line, with keys `L, beta, h, S_lieb, e, e_err, m`. `e` is the energy per site at field `h`. The zero-field energy is `e0 = e + h (m + S_lieb/N)/2` (Sec. III of the manuscript). Lines starting with `t=` are checkpoint progress.
- **Tilings** (`*.pkl`): a pickled Python `set` of removed edges `(site_a, site_b)` for a `Tiling(L)`. Load it with

  ```python
  T = Tiling(L)
  T.removed = pickle.load(open(f, 'rb'))
  ```

- **File-name patterns:**
  - `ann_L_sector_h_seed_steps.pkl`: LSWT annealing
  - `fa_L_sector_h_seed.pkl`: two-stage annealing
  - `big_m{12,9,36,18}_L36.pkl`: the M/2, 2M/3, M/6 and M/3 crystals (m = 1/12, 1/9, 1/36, 1/18)
- **Enumeration results** (`periodic_*.json`, `big_*_s*.json`): candidate tilings per supercell, with coordination histogram `hist`, imbalance `m`, model energy `e_model` and, where computed, `e_lswt` / `m_rep`.
- **Energies:** all in units of J. The field h is in units of J (g μ_B = 1).

## Notes and limitations

- The crystal search is exhaustive only for supercells with at most 36 sites (see the manuscript, Sec. V and Limitations).
- Several scripts were written for exploratory steps and later superseded. The manuscript figures are produced by `paper_figs.py`, `curv_fig.py` and the scripts listed in the table.
- The largest files are the classical spin configurations (`comp_*.pkl`, `eq*.pkl`, `r*_h0.1.pkl`, about 8.6 MB each). They are below GitHub's 100 MB per-file limit. You may prefer Git LFS or to leave them out.
- License: not yet chosen. Add a `LICENSE` file before making the repository public.

## Citation

If you use this code, please cite the manuscript (reference to be added on publication).
