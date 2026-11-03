"""Sample-visit alignment for ADNI blood expression and clinical trajectories."""
from pathlib import Path
import zipfile

import pandas as pd


def load_expression_samples(path, genes):
    with zipfile.ZipFile(path) as archive:
        with archive.open("ADNI_Gene_Expression_Profile.csv") as handle:
            metadata = pd.read_csv(handle, header=None, nrows=3, low_memory=False)
        with archive.open("ADNI_Gene_Expression_Profile.csv") as handle:
            raw = pd.read_csv(handle, skiprows=2, low_memory=False)
    sample_columns = list(raw.columns[3:])
    assert len(sample_columns) == metadata.shape[1] - 3
    samples = pd.DataFrame({
        "sample_column": sample_columns,
        "subject": metadata.iloc[2, 3:].to_numpy(),
        "sample_visit": metadata.iloc[1, 3:].to_numpy(),
        "phase": metadata.iloc[0, 3:].to_numpy(),
    })
    qc = {"260/280", "260/230", "RIN", "Affy Plate", "YearofCollection", "ProbeSet"}
    raw = raw[~raw.SubjectID.isin(qc) & raw.SubjectID.astype(str).str.contains("_at", na=False)].copy()
    raw["symbol"] = raw["Unnamed: 2"].astype(str).str.split(r"\s*\|\|\s*").str[0]
    raw = raw[raw.symbol.isin(set(genes))].copy()
    expression = raw[sample_columns].apply(pd.to_numeric, errors="coerce")
    expression["variance"] = expression.var(axis=1)
    expression["symbol"] = raw.symbol.to_numpy()
    expression["probe"] = raw.SubjectID.to_numpy()
    expression = expression.sort_values("variance", ascending=False).drop_duplicates("symbol")
    probes = expression.set_index("symbol").probe.to_dict()
    expression = expression.set_index("symbol")[sample_columns]
    return expression, probes, samples


def build_index_cohort(mmse_path, diagnosis_path, samples):
    dx = pd.read_csv(diagnosis_path, low_memory=False)
    dx["sample_visit"] = dx.VISCODE.astype(str).str.lower().str.strip()
    dx["sample_date"] = pd.to_datetime(dx.EXAMDATE, errors="coerce")
    dx = dx.dropna(subset=["DIAGNOSIS", "sample_date"])
    dx = dx.sort_values(["PTID", "sample_date"]).drop_duplicates(["PTID", "sample_visit"], keep="first")
    index = samples.merge(
        dx[["PTID", "sample_visit", "sample_date", "DIAGNOSIS"]],
        left_on=["subject", "sample_visit"], right_on=["PTID", "sample_visit"], how="left",
        validate="one_to_one",
    ).drop(columns="PTID")

    mmse = pd.read_csv(mmse_path, low_memory=False)
    mmse["score"] = pd.to_numeric(mmse.MMSCORE, errors="coerce")
    mmse["date"] = pd.to_datetime(mmse.VISDATE, errors="coerce")
    mmse["sample_visit"] = mmse.VISCODE.astype(str).str.lower().str.strip()
    phase = mmse.PHASE.astype(str).str.upper()
    mmse.loc[phase.eq("ADNI2") & mmse.sample_visit.eq("v01"), "sample_visit"] = "v03"
    mmse.loc[phase.eq("ADNIGO") & mmse.sample_visit.eq("sc"), "sample_visit"] = "bl"
    mmse = mmse.dropna(subset=["score", "date"])
    mmse = mmse.sort_values(["PTID", "date"]).drop_duplicates(["PTID", "date"], keep="first")

    starts = index.merge(
        mmse[["PTID", "sample_visit", "date", "score", "VISCODE", "VISCODE2"]],
        left_on=["subject", "sample_visit"], right_on=["PTID", "sample_visit"],
        how="inner", validate="one_to_one",
    ).rename(columns={"date": "index_date", "score": "baseline_mmse",
                      "DIAGNOSIS": "baseline_diagnosis", "VISCODE": "mmse_visit"})
    starts = starts.dropna(subset=["sample_date", "baseline_diagnosis"])
    starts = starts.sort_values(["subject", "sample_date", "sample_column"]).drop_duplicates("subject", keep="first")

    follow = mmse.merge(
        starts[["subject", "sample_column", "sample_date", "baseline_mmse"]],
        left_on="PTID", right_on="subject", suffixes=("", "_start"),
    )
    follow = follow[follow.date > follow.sample_date]
    rows = []
    for row in starts.itertuples():
        group = follow[follow.sample_column.eq(row.sample_column)]
        if group.empty:
            continue
        decline = group[group.score <= row.baseline_mmse - 3]
        event = int(not decline.empty)
        event_date = decline.date.iloc[0] if event else group.date.max()
        rows.append({
            "subject": row.subject,
            "sample_column": row.sample_column,
            "sample_visit": row.sample_visit,
            "phase": row.phase,
            "mmse_visit": row.mmse_visit,
            "sample_date": row.sample_date,
            "index_date": row.index_date,
            "time": (event_date - row.sample_date).days / 30.4375,
            "event": event,
            "baseline_mmse": row.baseline_mmse,
            "baseline_diagnosis": row.baseline_diagnosis,
            "last_mmse": group.sort_values("date").score.iloc[-1],
            "n_visits": len(group) + 1,
        })
    cohort = pd.DataFrame(rows)
    crosswalk = starts.assign(days_from_sample_to_mmse=lambda x: (x.index_date - x.sample_date).dt.days)
    crosswalk = crosswalk.groupby(["phase", "sample_visit", "mmse_visit", "VISCODE2"], dropna=False).agg(
        n=("subject", "size"), median_days=("days_from_sample_to_mmse", "median"),
        min_days=("days_from_sample_to_mmse", "min"), max_days=("days_from_sample_to_mmse", "max"),
    ).reset_index()
    return cohort, crosswalk
