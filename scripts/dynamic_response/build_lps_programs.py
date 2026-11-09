import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from project_paths import RESULTS, configured_path

OUT = RESULTS / 'scene'
GMT = configured_path('go_gmt')


def bh(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def read_gmt(path):
    terms = {}
    with path.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            terms[fields[0]] = set(fields[2:])
    return terms


def enrich(query, background, terms, query_name):
    query = set(query) & set(background)
    background = set(background)
    rows = []
    for term, genes in terms.items():
        term_genes = genes & background
        if not 5 <= len(term_genes) <= 500:
            continue
        overlap = query & term_genes
        if not overlap:
            continue
        k = len(overlap)
        p = float(hypergeom.sf(k - 1, len(background), len(term_genes), len(query)))
        rows.append({"query": query_name, "term": term, "overlap_n": k, "query_n": len(query), "term_n": len(term_genes), "background_n": len(background), "p": p, "genes": ";".join(sorted(overlap))})
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result["q"] = bh(result.p)
    return result.sort_values("p")


def main():
    table = pd.read_csv(OUT / "response_gene_table.csv")
    members = pd.read_csv(configured_path('members_file')).set_index("og_id")
    node_symbols = members.human_symbols.dropna().to_dict()
    background = sorted({symbol.upper() for node in table.node for symbol in str(node_symbols.get(node, "")).split(";") if symbol})
    observed_top = table.nlargest(100, "observed_abs_mean")
    predicted_top = table.nlargest(100, "predicted_abs_mean")
    concordant = table.loc[np.sign(table.observed_mean) == np.sign(table.predicted_mean)].copy()
    concordant["concordance_strength"] = np.minimum(concordant.observed_mean.abs(), concordant.predicted_mean.abs())
    concordant_top = concordant.nlargest(100, "concordance_strength")

    def symbols(frame):
        return sorted({symbol.upper() for node in frame.node for symbol in str(node_symbols.get(node, "")).split(";") if symbol})

    terms = read_gmt(GMT)
    results = []
    for name, frame in [("observed_top100", observed_top), ("predicted_top100", predicted_top), ("concordant_top100", concordant_top)]:
        result = enrich(symbols(frame), background, terms, name)
        if not result.empty:
            results.append(result)
    enrichment = pd.concat(results, ignore_index=True)
    enrichment.to_csv(OUT / "go_biological_process_enrichment.csv", index=False)
    concordant_top.to_csv(OUT / "top100_concordant_response_genes.csv", index=False)
    summary = {
        "status": "complete",
        "background_genes": len(background),
        "queries": {},
    }
    for name, frame in enrichment.groupby("query"):
        significant = frame.loc[frame.q <= 0.05]
        summary["queries"][name] = {
            "n_significant_terms": len(significant),
            "top_terms": significant.head(12)[["term", "q"]].to_dict("records"),
        }
    (OUT / "program_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
