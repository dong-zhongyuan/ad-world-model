import json
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
DATA = WORK / "results/lps_dynamic_adaptation_v1/metrics_by_test_animal.csv"
OUT = WORK / "results/lps_dynamic_headline_v1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA)
    rows = []
    for method, frame in data.groupby("method"):
        model_mse = frame.mse.mean()
        zero_mse = frame.zero_response_mse.mean()
        rows.append({
            "method": method,
            "n_test_lps_animals": len(frame),
            "mse": model_mse,
            "zero_response_mse": zero_mse,
            "relative_mse_reduction": 1.0 - model_mse / zero_mse,
            "mean_delta_pearson": frame.delta_pearson.mean(),
            "mean_top100_overlap": frame.top100_overlap.mean(),
        })
    summary = pd.DataFrame(rows).sort_values("relative_mse_reduction", ascending=False)
    summary.to_csv(OUT / "headline_by_method.csv", index=False)
    combined = data.loc[data.method.eq("combined_linear")].copy()
    combined.to_csv(OUT / "combined_by_test_animal.csv", index=False)

    rng = np.random.default_rng(42)
    model = combined.mse.to_numpy()
    zero = combined.zero_response_mse.to_numpy()
    indices = rng.integers(0, len(combined), size=(10000, len(combined)))
    bootstrap = 1.0 - model[indices].mean(axis=1) / zero[indices].mean(axis=1)
    headline = summary.loc[summary.method.eq("combined_linear")].iloc[0]
    result = {
        "status": "complete",
        "protocol": "fixed combined source-plus-world ridge model, alpha=1; reciprocal processing-held-out transfer",
        "n_test_lps_animals": len(combined),
        "relative_mse_reduction": float(headline.relative_mse_reduction),
        "animal_bootstrap_95_low": float(np.quantile(bootstrap, 0.025)),
        "animal_bootstrap_95_high": float(np.quantile(bootstrap, 0.975)),
        "mean_delta_pearson": float(headline.mean_delta_pearson),
        "mean_top100_overlap": float(headline.mean_top100_overlap),
        "target_relative_mse_reduction": 0.50,
        "target_met": bool(headline.relative_mse_reduction >= 0.50),
    }
    (OUT / "headline_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(combined.to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
