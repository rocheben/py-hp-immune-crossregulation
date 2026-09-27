"""Figure 5 — Stochastic evolutionary trajectories of g3h (kappa = 0.8,
400 generations, 8 replicate lines per regime, g0 = 0.02, fixed seeds).
Takes ~2 min.
Output: figures/Figure5.png"""
import evolution as E

E.figure5(kappa=0.8, ngen=400, reps=8, g0=0.02)
