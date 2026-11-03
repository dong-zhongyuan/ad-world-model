#!/usr/bin/env python3
"""Target-level clinical analysis of ADNI blood gene expression and MMSE decline."""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test

TARGETS = ["BIN1", "BLNK", "CR1", "GRN", "PILRB", "CD55", "ACP5", "ST13", "NESP55", "BST2"]
GENE_FOR_TARGET = {target: target for target in TARGETS}
GENE_FOR_TARGET["NESP55"] = "GNAS"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bh(values):
    p = np.asarray(values, float)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def visit_month(value):
    value = str(value).lower().strip()
    if value in {"bl", "sc", "v01"}:
        return 0.0
    match = re.fullmatch(r"m(\d+)", value)
    return float(match.group(1)) if match else np.nan


def load_expression(path, genes=None):
    with zipfile.ZipFile(path) as archive:
        with archive.open("ADNI_Gene_Expression_Profile.csv") as handle:
            raw = pd.read_csv(handle, skiprows=2, low_memory=False)
    qc = {"260/280", "260/230", "RIN", "Affy Plate", "YearofCollection", "ProbeSet"}
    raw = raw[~raw.SubjectID.isin(qc) & raw.SubjectID.astype(str).str.contains("_at", na=False)].copy()
    raw["symbol"] = raw["Unnamed: 2"].astype(str).str.split(r"\s*\|\|\s*").str[0]
    selected_genes = set(GENE_FOR_TARGET.values()) if genes is None else set(genes)
    raw = raw[raw.symbol.isin(selected_genes)].copy()
    sample_columns = [c for c in raw.columns if c not in {"SubjectID", "Unnamed: 1", "Unnamed: 2", "symbol", "probe"}]
    expression = raw[sample_columns].apply(pd.to_numeric, errors="coerce")
    expression["variance"] = expression.var(axis=1)
    expression["symbol"] = raw.symbol.values
    expression["probe"] = raw.SubjectID.values
    expression = expression.sort_values("variance", ascending=False).drop_duplicates("symbol")
    probe_map = expression.set_index("symbol").probe.to_dict()
    expression = expression.set_index("symbol")[sample_columns]
    return expression, probe_map, sample_columns


def load_mmse(path):
    mmse = pd.read_csv(path, low_memory=False)
    mmse["MMSCORE"] = pd.to_numeric(mmse.MMSCORE, errors="coerce")
    mmse["month"] = mmse.VISCODE2.map(visit_month)
    mmse = mmse.dropna(subset=["MMSCORE", "month"]).sort_values(["PTID", "month"])
    records = []
    for subject, group in mmse.groupby("PTID", sort=False):
        group = group.drop_duplicates("month", keep="first")
        if len(group) < 2:
            continue
        baseline = float(group.MMSCORE.iloc[0])
        baseline_month = float(group.month.iloc[0])
        follow = group[group.month > baseline_month]
        if follow.empty:
            continue
        decline = follow[follow.MMSCORE <= baseline - 3]
        event = int(not decline.empty)
        event_month = float(decline.month.iloc[0]) if event else float(follow.month.iloc[-1])
        records.append(dict(subject=subject, time=event_month - baseline_month, event=event,
                            baseline_mmse=baseline, baseline_month=baseline_month,
                            last_mmse=float(follow.MMSCORE.iloc[-1]), n_visits=len(group)))
    return pd.DataFrame(records)


def load_diagnosis(path):
    dx = pd.read_csv(path, low_memory=False)
    dx["month"] = dx.VISCODE2.map(visit_month)
    dx = dx.dropna(subset=["DIAGNOSIS", "month"]).sort_values(["PTID", "month"])
    dx = dx.drop_duplicates("PTID", keep="first")
    return dx[["PTID", "DIAGNOSIS"]].rename(columns={"PTID": "subject", "DIAGNOSIS": "baseline_diagnosis"})


def fit_model(data, adjusted):
    fit_data = data.copy()
    fit_data["expr"] = (fit_data.expr - fit_data.expr.mean()) / fit_data.expr.std()
    columns = ["time", "event", "expr"]
    if adjusted:
        fit_data["baseline_mmse_z"] = (fit_data.baseline_mmse - fit_data.baseline_mmse.mean()) / fit_data.baseline_mmse.std()
        diagnosis = pd.get_dummies(fit_data.baseline_diagnosis.astype(int).astype(str), prefix="diagnosis", drop_first=True)
        fit_data = pd.concat([fit_data, diagnosis], axis=1)
        columns += ["baseline_mmse_z"] + diagnosis.columns.tolist()
    fit_data = fit_data[columns].dropna()
    if fit_data.event.sum() < 5 or fit_data.expr.std() < 1e-8:
        return None
    model = CoxPHFitter().fit(fit_data, duration_col="time", event_col="event")
    row = model.summary.loc["expr"]
    result = dict(HR=float(row["exp(coef)"]), HR_lower=float(row["exp(coef) lower 95%"]),
                  HR_upper=float(row["exp(coef) upper 95%"]), p_wald=float(row["p"]),
                  n=len(fit_data), events=int(fit_data.event.sum()))
    if adjusted:
        test = proportional_hazard_test(model, fit_data, time_transform="rank")
        result["ph_p"] = float(test.summary.loc["expr", "p"])
    return result


def mr_directions(path):
    frame = pd.read_csv(path)
    result = {}
    for target in TARGETS:
        rows = frame[frame.protein.eq(target) if target != "NESP55" else frame.protein.eq("GNAS")]
        if rows.empty:
            result[target] = np.nan
            continue
        if "identity_match" in rows and rows.identity_match.eq(1).any():
            rows = rows[rows.identity_match.eq(1)]
        direction = rows.iloc[0].direction
        result[target] = 1 if direction == "Both positive" else -1 if direction == "Both negative" else np.nan
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--mmse", type=Path, required=True)
    parser.add_argument("--diagnosis", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    expression_path = args.expression
    mmse_path = args.mmse
    diagnosis_path = args.diagnosis
    evidence_path = args.evidence
    expression, probe_map, samples = load_expression(expression_path)
    clinical = load_mmse(mmse_path).merge(load_diagnosis(diagnosis_path), on="subject", how="left")
    clinical = clinical[clinical.subject.isin(samples)].copy()
    directions = mr_directions(evidence_path)
    rows = []
    for target in TARGETS:
        gene = GENE_FOR_TARGET[target]
        if gene not in expression.index:
            rows.append(dict(target=target, source_gene=gene, status="not_measured", model="adjusted"))
            continue
        base = clinical.copy()
        base["expr"] = base.subject.map(expression.loc[gene])
        base = base.dropna(subset=["expr"])
        for adjusted in [False, True]:
            result = fit_model(base, adjusted)
            if result is None:
                continue
            result.update(target=target, source_gene=gene, status="estimated",
                          model="adjusted" if adjusted else "unadjusted", probe_id=probe_map[gene],
                          mr_risk_sign=directions[target], clinical_risk_sign=int(np.sign(np.log(result["HR"]))),
                          direction_concordant=bool(np.sign(np.log(result["HR"])) == directions[target]))
            if not adjusted:
                high = base.expr > base.expr.median()
                test = logrank_test(base.loc[high, "time"], base.loc[~high, "time"],
                                    base.loc[high, "event"], base.loc[~high, "event"])
                result["p_logrank_median_split"] = float(test.p_value)
            rows.append(result)
    results = pd.DataFrame(rows)
    for model in ["unadjusted", "adjusted"]:
        indices = results.index[(results.model == model) & results.status.eq("estimated")]
        results.loc[indices, "q_bh"] = bh(results.loc[indices, "p_wald"])
    results.to_csv(out / "target_survival_results.csv", index=False)
    cohort = clinical.copy()
    for target in TARGETS:
        gene = GENE_FOR_TARGET[target]
        if gene in expression.index:
            cohort[target] = cohort.subject.map(expression.loc[gene])
    cohort.to_csv(out / "analysis_cohort.csv", index=False)
    primary = results[(results.model == "adjusted") & results.status.eq("estimated")]
    summary = dict(
        status="complete", n_expression_samples=len(samples), n_clinical_subjects=len(clinical),
        events=int(clinical.event.sum()), targets=TARGETS, measured_targets=sorted(primary.target.tolist()),
        missing_targets=sorted(set(TARGETS) - set(primary.target)),
        endpoint="time to first observed MMSE decline of at least 3 points from the earliest scored visit",
        time_origin="earliest MMSE record with a numeric month; the expression matrix does not provide a sample-visit field",
        hr_scale="per 1 SD of blood gene expression", adjusted_for=["baseline MMSE", "baseline diagnosis"],
        primary_model="adjusted", nominal_positive=primary[primary.p_wald < 0.05].target.tolist(),
        fdr_positive=primary[primary.q_bh < 0.05].target.tolist(),
        direction_concordant=primary[primary.direction_concordant].target.tolist(),
        source_hashes={expression_path.name: sha(expression_path), mmse_path.name: sha(mmse_path),
                       diagnosis_path.name: sha(diagnosis_path), evidence_path.name: sha(evidence_path)},
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(primary[["target", "HR", "HR_lower", "HR_upper", "p_wald", "q_bh", "ph_p", "n", "events", "direction_concordant"]].to_string(index=False))


if __name__ == "__main__":
    main()
