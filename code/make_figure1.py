"""Figure 1 — Plasmodium single infection: joint fit of parasitaemia and
uninfected RBC. Redrawn from the ARCHIVED parameters (fits/single.json), without
re-running the optimisation (to refit: python estimate.py single).
Output: figures/Figure1.png"""
import json, os
import estimate as e

single = json.load(open(os.path.join(e.FITS, "single.json")))
tp, yp, sp = e.load_para(e.T_SINGLE)
tr, yr, sr = e.load_rbc(e.T_RBC_S)
e._fig_single(single, ((tp, yp, sp), (tr, yr, sr), yr[0]))
