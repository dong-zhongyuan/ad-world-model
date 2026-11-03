from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches

WORK = Path(__file__).resolve().parent
OUT = WORK / "figures"
TEAL = "#2A9D8F"
PURPLE = "#7E6BB0"
RED = "#C95C5C"
DARK = "#263238"
LIGHT = "#E8ECEC"


def panel_label(ax, label):
    ax.text(-0.05, 1.05, label, transform=ax.transAxes, fontsize=20, fontweight="bold", va="top", color=DARK)


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": DARK, "text.color": DARK, "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK})
    cd4 = pd.read_csv(WORK / "cd4_target_coverage.csv")
    myeloid = pd.read_csv(WORK / "results/myeloid_action_v1/candidate_actions_by_lane.csv")
    with np.load(WORK / "results/blood_actions_v1/blood_action_log2fc.npz", allow_pickle=True) as data:
        lanes = data["lanes"].astype(str)
        blood_lps = data["lps_response"]
    with np.load(WORK / "results/cd4_real_action_brain_v1/brain_response_predictions.npz", allow_pickle=True) as data:
        nodes = data["nodes"].astype(str)
        targets = data["targets"].astype(str)
        conditions = data["conditions"].astype(str)
        brain_response = data["response__Micro"]
    with np.load(WORK / "results/real_action_brain_v1/brain_response_predictions.npz", allow_pickle=True) as data:
        myeloid_candidates = data["candidates"].astype(str)
        myeloid_conditions = data["conditions"].astype(str)
        myeloid_brain = data["candidate__Micro"]
    integrated = pd.read_csv(WORK / "results/integrated_target_map_v1/integrated_target_map.csv")
    genes = pd.read_csv("/public/home/mengxl/dzy/ad_omics_v3/data/processed/genes.csv").set_index("node_id")

    fig = plt.figure(figsize=(15, 22.5), facecolor="white")
    grid = fig.add_gridspec(7, 4, height_ratios=[0.75, 1.65, 1.0, 1.15, 1.1, 1.65, 1.1], hspace=0.60, wspace=0.40, left=0.14, right=0.97, top=0.955, bottom=0.055)
    fig.suptitle("Real peripheral perturbations define an actionable input space", fontsize=23, fontweight="bold", x=0.06, ha="left")

    ax = fig.add_subplot(grid[0, :])
    ax.axis("off")
    panel_label(ax, "A")
    stages = [(0.08, "CD4 Perturb-seq\n22M cells · 4 donors", TEAL), (0.30, "Myeloid Perturb-seq\nuntreated · LPS", PURPLE), (0.52, "Measured state\ntransitions", TEAL), (0.74, "Blood–brain\nworld model", PURPLE), (0.91, "Brain-cell\nprograms", RED)]
    for x, text, color in stages:
        ax.add_patch(patches.Circle((x, 0.55), 0.032, transform=ax.transAxes, facecolor=color, edgecolor="none", alpha=0.85))
        ax.text(x, 0.32, text, transform=ax.transAxes, ha="center", va="top", fontsize=10.5)
    for left, right in zip(stages[:-1], stages[1:]):
        ax.annotate("", xy=(right[0] - 0.032, 0.55), xytext=(left[0] + 0.032, 0.55), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.8, "color": DARK})
    ax.text(0.50, 0.90, "Observed perturbation responses replace input-coordinate zeroing", transform=ax.transAxes, ha="center", fontsize=12, color="#526068")

    ax = fig.add_subplot(grid[1, :2])
    panel_label(ax, "B")
    order = ["GRN", "BIN1", "PILRB", "BST2", "ST13", "ACP5", "CD55", "CR1", "GNAS", "BLNK"]
    matrix = np.full((len(order), 3), np.nan)
    for i, target in enumerate(order):
        for j, condition in enumerate(conditions):
            selected = cd4.loc[cd4.target_contrast_gene_name.eq(target) & cd4.culture_condition.eq(condition)]
            if not selected.empty:
                matrix[i, j] = abs(selected.ontarget_effect_size.iloc[0])
    cmap = plt.cm.YlGnBu.copy()
    cmap.set_bad(LIGHT)
    im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=0, vmax=32)
    ax.set_xticks(range(3), conditions)
    ax.set_yticks(range(len(order)), order)
    ax.set_title("CD4 on-target knockdown magnitude", loc="left", fontweight="bold")
    for i in range(len(order)):
        for j in range(3):
            if np.isfinite(matrix[i, j]) and matrix[i, j] > 0:
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8, color="white" if matrix[i, j] > 16 else DARK)
            elif order[i] == "BLNK":
                ax.text(j, i, "—", ha="center", va="center", color="#7C878D")
    fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03, label="|On-target log2FC|")

    ax = fig.add_subplot(grid[1, 2:])
    panel_label(ax, "C")
    cell_matrix = np.full((len(order), 3), np.nan)
    for i, target in enumerate(order):
        for j, condition in enumerate(conditions):
            selected = cd4.loc[cd4.target_contrast_gene_name.eq(target) & cd4.culture_condition.eq(condition)]
            if not selected.empty:
                cell_matrix[i, j] = selected.n_cells_target.iloc[0]
    im = ax.imshow(np.log10(np.ma.masked_invalid(cell_matrix)), aspect="auto", cmap="PuBuGn", vmin=1.5, vmax=3.2)
    im.cmap.set_bad(LIGHT)
    ax.set_xticks(range(3), conditions)
    ax.set_yticks(range(len(order)), [])
    ax.set_title("Perturbed cells per candidate", loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03, label="log10 cells")

    ax = fig.add_subplot(grid[2, :2])
    panel_label(ax, "D")
    selected = myeloid.loc[myeloid.target.isin(["ACP5", "CR1"])].copy()
    selected["knockdown"] = -selected.on_target_delta
    summary = selected.groupby(["target", "condition"]).knockdown.agg(["mean", "std", "count"]).reset_index()
    positions = {"untreated": -0.12, "LPS": 0.12}
    for target, color in [("ACP5", TEAL), ("CR1", PURPLE)]:
        frame = summary.loc[summary.target.eq(target)]
        x = [0 if target == "ACP5" else 1]
        for condition in ["untreated", "LPS"]:
            row = frame.loc[frame.condition.eq(condition)].iloc[0]
            ax.errorbar(x[0] + positions[condition], row["mean"], yerr=row["std"], fmt="o", ms=10, capsize=4, color=color, label=f"{target} {condition}")
            for lane in selected.loc[selected.target.eq(target) & selected.condition.eq(condition)].itertuples():
                ax.scatter(x[0] + positions[condition] + np.random.default_rng(42).uniform(-0.025, 0.025), lane.knockdown, s=12, color=color, alpha=0.35)
    ax.set_xticks([0, 1], ["ACP5", "CR1"])
    ax.set_ylabel("On-target knockdown (−Δ log expression)")
    ax.set_title("Myeloid perturbation is reproducible across four lanes", loc="left", fontweight="bold")
    ax.text(0.02, 0.03, "Filled circles: untreated/LPS means; points: individual lanes", transform=ax.transAxes, fontsize=9, color="#526068")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[2, 2:])
    panel_label(ax, "E")
    correlation = np.corrcoef(blood_lps)
    im = ax.imshow(correlation, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(4), [value.replace("Mac_", "") for value in lanes])
    ax.set_yticks(range(4), [value.replace("Mac_", "") for value in lanes])
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{correlation[i, j]:.2f}", ha="center", va="center", color="white" if correlation[i, j] > 0.6 else DARK)
    ax.set_title("Measured peripheral LPS response across lanes", loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05, label="Pearson r")

    ax = fig.add_subplot(grid[3, :])
    panel_label(ax, "F")
    selected_targets = [list(targets).index("GRN"), list(targets).index("BIN1")]
    selected = brain_response[selected_targets]
    strength = np.max(np.abs(selected), axis=(0, 1))
    top = np.argsort(-strength)[:20]
    matrix = selected[:, :, top].reshape(6, len(top))
    limit = np.quantile(np.abs(matrix), 0.95)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_yticks(range(6), [f"{target} {condition}" for target in ["GRN", "BIN1"] for condition in conditions])
    ax.set_xticks(range(len(top)), [str(genes.loc[nodes[i]].symbols).split("|")[-1] for i in top], rotation=60, ha="right", fontsize=8)
    ax.set_title("Measured GRN/BIN1 actions generate condition-specific brain responses", loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.02, label="Predicted brain log-expression response")

    ax = fig.add_subplot(grid[4, :2])
    panel_label(ax, "G")
    acp5 = myeloid_brain[list(myeloid_candidates).index("ACP5"), :, list(myeloid_conditions).index("LPS"), :].mean(axis=0)
    top = np.argsort(-np.abs(acp5))[:16][::-1]
    values = acp5[top]
    labels = [str(genes.loc[nodes[i]].symbols).split("|")[-1] for i in top]
    ax.barh(np.arange(len(top)), values, color=[TEAL if value >= 0 else PURPLE for value in values], height=0.65)
    ax.set_yticks(np.arange(len(top)), labels)
    ax.set_xlabel("Predicted microglial response")
    ax.set_title("ACP5 action under LPS: microglial program", loc="left", fontweight="bold")
    ax.axvline(0, color=DARK, lw=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)

    ax = fig.add_subplot(grid[4, 2:])
    panel_label(ax, "H")
    acp5_lps = blood_lps.mean(axis=0)
    top = np.argsort(-np.abs(acp5_lps))[:16][::-1]
    values = acp5_lps[top]
    labels = [str(genes.loc[nodes[i]].symbols).split("|")[-1] for i in top]
    ax.barh(np.arange(len(top)), values, color=[RED if value >= 0 else TEAL for value in values], height=0.65)
    ax.set_yticks(np.arange(len(top)), labels)
    ax.set_xlabel("Measured peripheral log2FC")
    ax.set_title("Matched peripheral LPS input", loc="left", fontweight="bold")
    ax.axvline(0, color=DARK, lw=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)

    ax = fig.add_subplot(grid[5:, :])
    panel_label(ax, "I")
    coverage = pd.DataFrame({
        "CD4 action": integrated.cd4_perturbation_present.to_numpy(),
        "Myeloid action": integrated.myeloid_perturbation_present.to_numpy(),
        "Independent brain": integrated.independent_brain_dataset.ne("").to_numpy(),
        "Two AD outcomes": (integrated.mr_bell_pass & integrated.mr_finn_pass).to_numpy(),
        "Candidate compound": integrated.compound.notna().to_numpy(),
    }, index=integrated.target).loc[order]
    im = ax.imshow(coverage.astype(int), aspect="auto", cmap=plt.cm.YlGnBu, vmin=0, vmax=1)
    ax.set_xticks(range(coverage.shape[1]), coverage.columns)
    ax.set_yticks(range(len(coverage)), coverage.index)
    for i in range(len(coverage)):
        for j in range(coverage.shape[1]):
            ax.text(j, i, "●" if coverage.iloc[i, j] else "—", ha="center", va="center", color="white" if coverage.iloc[i, j] else "#7C878D", fontsize=14)
    ax.set_title("Ten targets connected to real actions, human genetics and compounds", loc="left", fontweight="bold")
    ax.spines[:].set_visible(False)

    fig.savefig(OUT / "Figure3_real_action.pdf", facecolor="white")
    fig.savefig(OUT / "Figure3_real_action.png", dpi=150, facecolor="white")
    print(OUT / "Figure3_real_action.pdf")


if __name__ == "__main__":
    main()
