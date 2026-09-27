"""Figure 2 — H. polygyrus single infection: log faecal egg count. Redrawn from
the ARCHIVED parameters (fits/worm.json), without re-running the optimisation
(to refit: python estimate.py worm).
Output: figures/Figure2.png"""
import json, os
import estimate as e

worm = json.load(open(os.path.join(e.FITS, "worm.json")))
tl, yl, sl = e.load_lfec(e.T_LFEC_S)
# same fixed parameters as in estimate.fit_worm
th = dict(e.WORM_FIXED, K2=4000.0, s3=0.05, g3=0.02, mu3=0.4, I3_0=1, Kc=50.0) | worm
e._fig_worm(th, (tl, yl, sl))
