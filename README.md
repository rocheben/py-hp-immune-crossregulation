# Submission package

**Immune cross-regulation drives mutual facilitation in a malaria–helminth coinfection
and shapes the evolution of worm immunosuppression**

Prepared for *Proceedings of the Royal Society B*. The package contains the manuscript,
the electronic supplementary material, all figures, the code, the source and processed
data, and the fitted parameter files. Every figure and every number in the manuscript is
reproduced by the code in `code/`, starting from the experimental workbooks in
`data/source/`.

## Contents

```
README.md
manuscript/   Manuscript.docx / .pdf   main text (Proc R Soc B format)
              ESM.docx / .pdf          electronic supplementary material, appendices A-C
figures/      Figure1..Figure7         main-text figures
              FigureA1, FigureB1       supplementary figures
data/source/  consolidated experimental workbooks (.xlsx)
code/         analysis scripts, self-contained
              data_processed/          group-level summaries used for fitting
              fits/                    fitted parameter files
```

The scripts resolve `data_processed/`, `fits/` and `figures/` relative to their own
location, so all commands below are run from inside `code/`.

## From the experimental data to the fitted series

`prepare_data.py` documents and reproduces the link between the raw workbooks and the
three time series that enter the fits. Each output holds the per-group, per-timepoint
mean, standard deviation, sample size and standard error of one measured quantity.

| Processed file | Source workbook | Sheet | Column |
| --- | --- | --- | --- |
| `py_parasitemia_summary.csv` | PCI | `05_parasitemia py and coinfecte` | `parasitemia` |
| `rbc_summary.csv` | PCI | `06_rbc py and coinfected` | `rbc` |
| `hp_lfec_summary.csv` | Plos | `01_data` | `lfec` |

Running `python prepare_data.py` regenerates the three files byte-for-byte from
`data/source/`. The copies in `code/data_processed/` are those used for every result.

## Code

| Module | Role | Produces |
| --- | --- | --- |
| `prepare_data.py` | Source workbooks to processed time series | `data_processed/*.csv` |
| `models.py` | Reaction definitions of the three within-host models | library |
| `deterministic.py` | Mean-field ODE integrator shared by all models | library |
| `estimate.py` | Estimation, likelihood profiling, fit-figure functions | `fits/*.json` (+ Figures 1-3, A1 when refitting) |
| `evolution.py` | Worm fitness, selection gradient, stochastic invasion, figure functions | library (+ Figures 4-6 when run) |
| `interhost.py` | SIRS-Py by SIS-Hp between-host model | library |
| `run_interhost.py` | Coupling factors, epidemiological analyses, figure function | `fits/phi_interhost.json` (+ Figure 7 when run) |

One script per figure, each drawing from the archived `fits/*.json` (no refitting):

| Script | Figure | Output (`figures/`) | Time |
| --- | --- | --- | --- |
| `make_figure1.py` | 1, Py single-infection fit | `Figure1.png` | < 5 s |
| `make_figure2.py` | 2, Hp single-infection fit | `Figure2.png` | < 5 s |
| `make_figure3.py` | 3, coinfection fit | `Figure3.png` | < 5 s |
| `make_figure4.py` | 4, selection gradient | `Figure4.png` | ~5 s |
| `make_figure5.py` | 5, stochastic evolutionary trajectories | `Figure5.png` | ~2 min |
| `make_figure6.py` | 6, s12 sweep | `Figure6.png` | ~15 s |
| `make_figure7.py` | 7, between-host consequences | `Figure7.png` | ~5 s |
| `make_figureA1.py` | A1, likelihood profiles and 95% CIs | `FigureA1.png` | ~2 min |
| `make_figureB1.py` | B1, evolutionary-regime calibration | `FigureB1.png` | ~30 s |
| `make_all_figures.py` | all of the above (or a subset: `python make_all_figures.py 1 5 A1`) | | ~6 min |

Requirements: Python 3 with `numpy` (>= 2.0), `scipy`, `matplotlib`, plus `pandas` and
`openpyxl` for `prepare_data.py`.

## Reproducing the results

```bash
cd code
python prepare_data.py          # workbooks -> data_processed/*.csv
python make_all_figures.py      # every figure from the archived fits/*.json
```

To refit from scratch instead (overwrites `fits/*.json`, see the note below):

```bash
python estimate.py all          # fits -> fits/*.json, Figures 1-3, A1
python run_interhost.py         # fits/phi_interhost.json (delete it first to re-derive) and Figure 7
python make_all_figures.py      # all figures, consistent with the new fits
```

Note on the fits. Several parameters are only weakly identified, in particular the two
Th1/Th2 antagonism strengths, whose likelihood profiles are shallow (appendix A). The
multistart optimiser can therefore settle on different points within the same confidence
region from one run to another. The files in `code/fits/` are the ones used for every
figure and number in the manuscript. Re-running `estimate.py all` overwrites them, so if
you refit from scratch you should regenerate all downstream figures in the order above,
so that text and figures stay consistent.

## Values used in the manuscript

Within-host, Plasmodium single infection:
`r1 = 0.687, k1 = 0.150, b1 = 0.485, c1 = 197.81, rho = 77.95, d1 = 0.092, mu1 = 0.259`

Within-host, nematode single infection:
`r2 = 0.585, k2 = 0.809, egg-scaling factor = 809`

Coinfection interaction:
`K2 = 112, s12 = 2.29, s21 = 3.19`

Likelihood-profile 95% intervals:
`r1 [0.52, 1.38], s12 [0.92, 8.55], s21 [1.28, 8.77]`

Between-host coupling factors derived from the fitted within-host model:
`phi_betaPy = 1.287, phi_betaHp = 1.622, phi_gamma = 1.002`

Evolutionary regime (calibrated, not estimated, see appendix B):
`k2 = 6, s3 = 6, mu3 = 0.2, K2 = 300`
