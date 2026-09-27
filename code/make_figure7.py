"""Figure 7 — Population-level consequences (SIRS-Py x SIS-Hp): effective R0 of
Py (A), coinfected fraction (B), deworming (C). Uses the coupling factors in
fits/phi_interhost.json (derived from the within-host fit if absent).
Also prints the SIRS waning sensitivity quoted in the text.
Output: figures/Figure7.png"""
import run_interhost as R

R.figure7()
R.sirs_sensitivity()
