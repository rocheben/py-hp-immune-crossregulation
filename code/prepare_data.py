# -*- coding: utf-8 -*-
"""
Data preparation: source workbooks -> the three processed time series used for fitting.

This script documents and reproduces the link between the raw experimental data
(data/source/*.xlsx) and the group-level summaries consumed by estimate.py
(data_processed/*.csv). Each summary is the per-group, per-timepoint mean, standard
deviation, sample size and standard error of one measured quantity.

    py_parasitemia_summary.csv  <- PCI workbook, sheet "05_parasitemia py and coinfecte"
                                   column "parasitemia"  (proportion of infected RBC)
    rbc_summary.csv             <- PCI workbook, sheet "06_rbc py and coinfected"
                                   column "rbc"          (red blood cell count)
    hp_lfec_summary.csv         <- Plos workbook, sheet "01_data"
                                   column "lfec"         (log faecal egg count)

Usage (from the directory containing this script):
    python prepare_data.py [--source ../data/source] [--out data_processed]
"""
import os, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# (output csv, workbook, sheet, value column)
SPECS = [
    ("py_parasitemia_summary.csv", "donnees_consolideesPCI.xlsx",
     "05_parasitemia py and coinfecte", "parasitemia"),
    ("rbc_summary.csv", "donnees_consolideesPCI.xlsx",
     "06_rbc py and coinfected", "rbc"),
    ("hp_lfec_summary.csv", "donnees_consolideesPlos.xlsx",
     "01_data", "lfec"),
]

GROUP = ["Treatment", "time_post_infection"]


def summarise(df, value):
    """Group-level mean, sd, n and standard error at each sampling time."""
    g = df.groupby(GROUP)[value].agg(["mean", "std", "count"]).reset_index()
    g["se"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=os.path.join(HERE, "..", "data", "source"),
                    help="directory holding the consolidated .xlsx workbooks")
    ap.add_argument("--out", default=os.path.join(HERE, "data_processed"),
                    help="directory for the processed .csv summaries")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    for out_csv, book, sheet, value in SPECS:
        path = os.path.join(a.source, book)
        df = pd.read_excel(path, sheet_name=sheet)
        missing = [c for c in GROUP + [value] if c not in df.columns]
        if missing:
            raise SystemExit(f"{book}:{sheet} is missing column(s) {missing}")
        s = summarise(df, value)
        dest = os.path.join(a.out, out_csv)
        s.to_csv(dest, index=False)
        print(f"{out_csv:28s} <- {book} :: {sheet} [{value}]  "
              f"({len(s)} group-time rows, n={int(df[value].notna().sum())} observations)")


if __name__ == "__main__":
    main()
