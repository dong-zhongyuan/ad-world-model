#!/usr/bin/env python3
"""Independently reproduce the baseline BST2 Cox estimate with statsmodels."""
import argparse
import json
import math
from pathlib import Path

import pandas as pd
import statsmodels.api as sm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    d = pd.read_csv(args.cohort)
    x = pd.DataFrame({
        "expr": (d.BST2 - d.BST2.mean()) / d.BST2.std(),
        "baseline_mmse_z": (d.baseline_mmse - d.baseline_mmse.mean()) / d.baseline_mmse.std(),
    })
    diagnosis = pd.get_dummies(d.baseline_diagnosis.astype(int).astype(str), prefix="diagnosis", drop_first=True, dtype=float)
    x = pd.concat([x, diagnosis], axis=1)
    fit = sm.PHReg(d.time, x, status=d.event, ties="breslow").fit()
    i = fit.model.exog_names.index("expr")
    ci = fit.conf_int()[i]
    result = {
        "target": "BST2", "model": "statsmodels PHReg; Breslow ties",
        "n": len(d), "events": int(d.event.sum()),
        "HR": float(math.exp(fit.params[i])),
        "HR_lower": float(math.exp(ci[0])), "HR_upper": float(math.exp(ci[1])),
        "p": float(fit.pvalues[i]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
