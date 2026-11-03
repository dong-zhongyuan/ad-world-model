#!/usr/bin/env python3
"""Add a uniformly analysed ADNI association layer to the evidence wheel."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_adni_target_survival import bh, fit_model, load_diagnosis, load_expression, load_mmse, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--mmse", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--definitions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    gates = pd.read_csv(args.gates)
    targets = sorted(gates.protein.unique())
    expression, probes, samples = load_expression(args.expression, targets)
    clinical = load_mmse(args.mmse).merge(load_diagnosis(args.diagnosis), on="subject", how="left")
    clinical = clinical[clinical.subject.isin(samples)].copy()

    rows = []
    for target in targets:
        if target not in expression.index:
            rows.append(dict(target=target, status="not_measured"))
            continue
        data = clinical.copy()
        data["expr"] = data.subject.map(expression.loc[target])
        data = data.dropna(subset=["expr"])
        result = fit_model(data, True)
        result.update(target=target, status="estimated", probe_id=probes[target])
        rows.append(result)
    results = pd.DataFrame(rows)
    indices = results.index[results.status.eq("estimated")]
    results.loc[indices, "q_bh"] = bh(results.loc[indices, "p_wald"])
    results.to_csv(args.out / "wheel_candidate_survival_results.csv", index=False)

    definitions = json.loads(args.definitions.read_text())
    results_by_target = results.set_index("target")
    status = []
    concordance = []
    for _, row in gates.iterrows():
        target = row.protein
        betas = pd.to_numeric(pd.Series([row.bell_beta, row.finn_beta]), errors="coerce").dropna()
        mr_sign = np.nan
        if len(betas) == 2 and (betas != 0).all() and np.sign(betas.iloc[0]) == np.sign(betas.iloc[1]):
            mr_sign = float(np.sign(betas.iloc[0]))
        if target not in results_by_target.index:
            status.append("unresolved")
            concordance.append(np.nan)
            continue
        result = results_by_target.loc[target]
        if result.status != "estimated":
            status.append("unresolved")
            concordance.append(np.nan)
            continue
        concordance.append(bool(np.sign(np.log(result.HR)) == mr_sign) if np.isfinite(mr_sign) else np.nan)
        status.append("pass" if result.q_bh < 0.05 else "not_met")
    gates["survival"] = status
    gates["survival_HR"] = [results_by_target.HR.get(target, np.nan) for target in gates.protein]
    gates["survival_q"] = [results_by_target.q_bh.get(target, np.nan) for target in gates.protein]
    gates["survival_direction_concordant"] = concordance
    layers = definitions["layers"][:7] + ["survival"] + definitions["layers"][7:]
    gates["passed_count"] = gates[layers].eq("pass").sum(axis=1)
    gates["unresolved_count"] = gates[layers].eq("unresolved").sum(axis=1)
    gates.to_csv(args.out / "evidence_gates_with_survival.csv", index=False)
    definitions["layers"] = layers
    definitions["rules"]["survival"] = "ADNI blood-expression adjusted Cox BH q < 0.05 for first observed MMSE decline >=3 points; unavailable expression is unresolved. Expression–clinical and protein-MR directions remain separately reported."
    definitions["outer_bars"] = "Number of passed criteria, from 0 to 10; not a weighted evidence score."
    definitions["survival_data"] = "727 ADNI participants; 347 first observed MMSE declines >=3; baseline MMSE and diagnosis adjusted."
    (args.out / "evidence_gate_definitions_with_survival.json").write_text(json.dumps(definitions, indent=2))
    summary = dict(status="complete", n_wheel_genes=len(targets),
                   n_measured=int(results.status.eq("estimated").sum()),
                   n_unresolved=int(gates.survival.eq("unresolved").sum()),
                   passed_targets=sorted(gates.loc[gates.survival.eq("pass"), "protein"].unique().tolist()),
                   criterion=definitions["rules"]["survival"],
                   source_hashes={"evidence_gates.csv": sha(args.gates)})
    (args.out / "wheel_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(results[results.target.eq("BST2")].to_string(index=False))


if __name__ == "__main__":
    main()
