"""Figure A1 (ESM) — Likelihood profiles and scaled 95% intervals for r1, s12
and s21, from the ARCHIVED fits (fits/single.json, coinf.json). 11 grid points,
the setting that gives the intervals reported in the manuscript
(r1 [0.52, 1.38], s12 [0.92, 8.55], s21 [1.28, 8.77]).
Output: figures/FigureA1.png"""
import estimate as e

e.profile(grid_n=11)
