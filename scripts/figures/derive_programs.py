import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from paths import REPO_ROOT, FIGURE_WORK


def bh(ps):
    ps = np.asarray(ps, float)
    if not len(ps):
        return ps.copy()
    order = np.argsort(ps)
    q = np.empty(len(ps))
    q[order] = np.minimum(1.0, np.minimum.accumulate((ps[order] * len(ps) / np.arange(1, len(ps) + 1))[::-1])[::-1])
    return q


def figure_programs():
    root = REPO_ROOT
    out = FIGURE_WORK
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    edges = pd.read_csv(root / "results/network_v3_repair/validated_edges_reverse.csv")
    members = pd.read_csv(root / "data/mappings/orthogroups_member_level.csv", dtype=str).set_index("og_id").human_symbols.dropna().map(lambda s: set(s.split(";"))).to_dict()
    genes = pd.read_csv(root / "data/processed/genes.csv")
    kinds = genes.set_index("node_id").kind.to_dict()
    names = genes.node_id.tolist()
    targets = ["BIN1", "CR1", "PILRB", "GRN", "BLNK", "CD55", "ACP5", "ST13", "GNAS", "BST2"]
    routes = ["B_Micro", "CD4T_Micro", "CD8T_Micro", "Mono_Micro", "NK_Micro", "Mono_Oligo"]
    path = out / "references/GO_Biological_Process_2023.gmt"
    gene_sets = []
    for line in path.read_text().splitlines():
        values = line.split("\t")
        gene_sets.append((values[0], set(values[2:])))
    gene_nodes = {}
    for node, symbols in members.items():
        for gene in symbols:
            gene_nodes.setdefault(gene, set()).add(node)
    term_nodes = {term: set().union(*(gene_nodes.get(g, set()) for g in symbols)) for term, symbols in gene_sets}
    rows = []
    metadata = []
    for route in routes:
        chunk = next((root / "results/network_v3_repair/edges" / ("rev_" + route) / "chunks").glob("*.npz"))
        with np.load(chunk) as arr:
            indices = arr["targets"]
        nodes = {names[i] for i in indices if kinds[names[i]] == "shared_og" and members.get(names[i])}
        for target in targets:
            selected = edges[edges.source.eq(target) & edges.combo.eq(route)]
            source_nodes = set(edges[edges.source.eq(target)].source_node)
            background = {n for n in nodes if any(n != s and any(x != y for x in members.get(s, set()) for y in members[n]) for s in source_nodes)}
            query = set(selected.target_node) & background
            metadata.append(dict(target=target, route=route, background=len(background), query=len(query)))
            for term, symbols in gene_sets:
                mapped = term_nodes[term] & background
                if not 5 <= len(mapped) <= 500:
                    continue
                hit = query & mapped
                p = float(hypergeom.sf(len(hit) - 1, len(background), len(mapped), len(query))) if query else 1.0
                rows.append(dict(target=target, route=route, term=term, p=p, overlap=len(hit), query=len(query), background=len(background), set_size=len(mapped), nodes=";".join(sorted(hit)), genes=";".join(sorted({g for n in hit for g in members[n] & symbols}))))
    frame = pd.DataFrame(rows)
    frame["q"] = bh(frame.p)
    frame["fold_enrichment"] = (frame.overlap / frame["query"].replace(0, np.nan)) / (frame.set_size / frame.background)
    frame.to_csv(data / "enrichment_all.csv.gz", index=False)
    frame[frame.q < 0.05].to_csv(data / "enrichment_significant.csv", index=False)
    terms = frame[frame.overlap >= 3].sort_values(["q", "p", "term"]).drop_duplicates("term").head(8).term.tolist()
    frame[frame.term.isin(terms)].to_csv(data / "enrichment_display.csv", index=False)
    pd.DataFrame(metadata).to_csv(data / "enrichment_query_registry.csv", index=False)
    provenance = dict(library="GO_Biological_Process_2023", library_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), source_url="https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Biological_Process_2023", unit="orthogroup; each GO set maps to unique tested OGs", family=len(frame), queries=60, correction="BH over all tested target-route-GO hypotheses", significant=int((frame.q < 0.05).sum()), background="Actual cached target indices and shared-OG/member-level nonself eligibility, per source and route", display_terms=terms)
    (data / "enrichment_provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    figure_programs()
