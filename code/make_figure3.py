"""Figure 3 — Coinfection joint fit (Py parasitaemia and Hp eggs, single vs
coinfected). Redrawn from the ARCHIVED parameters (fits/single.json, worm.json,
coinf.json), without re-running the optimisation (to refit: python estimate.py coinf).
Output: figures/Figure3.png"""
import json, os
import estimate as e

single = json.load(open(os.path.join(e.FITS, "single.json")))
worm = json.load(open(os.path.join(e.FITS, "worm.json")))
coinf = json.load(open(os.path.join(e.FITS, "coinf.json")))
e._fig_coinf(coinf, single, worm)
