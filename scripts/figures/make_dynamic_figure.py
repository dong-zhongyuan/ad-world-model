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
GRAY = "#B0B7BC"


def panel_label(ax, label):
    ax.text(-0.04, 1.04, label, transform=ax.transAxes, fontsize=20, fontweight="bold", va="top", ha="left", color=DARK)


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.8, "axes.edgecolor": DARK, "text.color": DARK, "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK})
    headline = pd.read_csv(WORK / "results/lps_dynamic_headline_v1/headline_by_method.csv")
    per_animal = pd.read_csv(WORK / "results/lps_dynamic_headline_v1/combined_by_test_animal.csv")
    with np.load(WORK / "results/lps_dynamic_scene_v1/dynamic_response_vectors.npz", allow_pickle=True) as data:
        nodes = data["nodes"].astype(str)
        observed = data["observed"]
        predicted = data["predicted"]
        gsm = data["gsm"].astype(str)
    genes = pd.read_csv(WORK / "results/lps_dynamic_scene_v1/response_gene_table.csv")
    go = pd.read_csv(WORK / "results/lps_dynamic_scene_v1/go_biological_process_enrichment.csv")

    fig = plt.figure(figsize=(15, 21), facecolor="white")
    grid = fig.add_gridspec(6, 4, height_ratios=[0.85, 1.0, 1.65, 1.05, 1.8, 1.25], hspace=0.60, wspace=0.38, left=0.13, right=0.97, top=0.95, bottom=0.055)
    fig.suptitle("Peripheral state transitions are translated into microglial responses", fontsize=23, fontweight="bold", x=0.06, ha="left")

    ax = fig.add_subplot(grid[0, :])
    ax.axis("off")
    panel_label(ax, "A")
    ax.text(0.03, 0.83, "Same-animal inflammatory states", fontsize=14, fontweight="bold")
    stages = [(0.08, "Peripheral\nphagocytes", TEAL), (0.30, "Blood–brain\nworld model", PURPLE), (0.52, "Dynamic\nadaptation", PURPLE), (0.74, "Microglial\nresponse", TEAL), (0.91, "Held-out\nanimals", RED)]
    for x, text, color in stages:
        circle = patches.Circle((x, 0.50), 0.035, transform=ax.transAxes, facecolor=color, alpha=0.85, edgecolor="none")
        ax.add_patch(circle)
        ax.text(x, 0.30, text, transform=ax.transAxes, ha="center", va="top", fontsize=11)
    for (x1, _, _), (x2, _, _) in zip(stages[:-1], stages[1:]):
        ax.annotate("", xy=(x2 - 0.035, 0.50), xytext=(x1 + 0.035, 0.50), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.8, "color": DARK})
    ax.text(0.42, 0.83, "Training and testing use disjoint animals and processing conditions", transform=ax.transAxes, ha="left", fontsize=10, color="#526068")

    ax = fig.add_subplot(grid[1, 0:2])
    panel_label(ax, "B")
    order = ["training_mean", "world_linear", "source_linear", "combined_linear"]
    labels = ["Zero response", "World model", "Peripheral state", "Combined model"]
    frame = headline.set_index("method").loc[order]
    values = frame.relative_mse_reduction.to_numpy() * 100
    colors = [GRAY, PURPLE, TEAL, RED]
    bars = ax.barh(np.arange(4), values, color=colors, height=0.62)
    ax.set_yticks(np.arange(4), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 65)
    ax.set_xlabel("MSE reduction versus zero response (%)")
    ax.set_title("Reciprocal held-out transfer", loc="left", fontweight="bold")
    for bar, value in zip(bars, values):
        ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 2:])
    panel_label(ax, "C")
    metrics = [("Delta Pearson", per_animal.delta_pearson.mean() * 100, 1.0), ("Top-100 recovery", per_animal.top100_overlap.mean() * 100, 100.0)]
    positions = np.arange(2)
    values = [x[1] for x in metrics]
    bars = ax.bar(positions, values, color=[TEAL, PURPLE], width=0.52)
    ax.set_xticks(positions, ["Response correlation", "Top-100 gene recovery"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Biological response fidelity", loc="left", fontweight="bold")
    for bar, value, metric in zip(bars, values, metrics):
        label = f"{value / 100:.3f}" if metric[0] == "Delta Pearson" else f"{value:.2f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, label, ha="center", fontweight="bold", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)

    for i in range(4):
        ax = fig.add_subplot(grid[2, i])
        if i == 0:
            panel_label(ax, "D")
        x = observed[i]
        y = predicted[i]
        ax.scatter(x, y, s=5, alpha=0.20, color=TEAL, edgecolors="none", rasterized=True)
        limit = np.quantile(np.abs(np.concatenate([x, y])), 0.995)
        ax.plot([-limit, limit], [-limit, limit], color=DARK, lw=0.8, ls="--")
        row = per_animal.loc[per_animal.gsm.eq(gsm[i])].iloc[0]
        ax.set_title(f"{gsm[i]}\nr={row.delta_pearson:.2f}; MSE↓{row.relative_mse_reduction * 100:.1f}%", fontsize=11)
        ax.set_xlabel("Observed")
        if i == 0:
            ax.set_ylabel("Predicted response")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")

    ax = fig.add_subplot(grid[3, :2])
    panel_label(ax, "E")
    x = np.arange(len(per_animal))
    ax.bar(x - 0.18, per_animal.delta_pearson, width=0.36, color=TEAL, label="Delta Pearson")
    ax.bar(x + 0.18, per_animal.top100_overlap, width=0.36, color=PURPLE, label="Top-100 recovery")
    ax.set_xticks(x, [value.replace("GSM", "Animal ") for value in per_animal.gsm])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title("Four held-out LPS animals", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[3, 2:])
    panel_label(ax, "F")
    direction = per_animal.groupby("train_processing").relative_mse_reduction.mean() * 100
    overall = headline.loc[headline.method.eq("combined_linear"), "relative_mse_reduction"].iloc[0] * 100
    labels = ["DMSO → inhibitors", "Inhibitors → DMSO", "Combined"]
    values = [direction.get("dmso"), direction.get("inhibitors"), overall]
    bars = ax.bar(np.arange(3), values, color=[PURPLE, TEAL, RED], width=0.58)
    ax.axhline(50, color=DARK, ls="--", lw=0.8)
    ax.set_xticks(np.arange(3), labels, rotation=12, ha="right")
    ax.set_ylabel("MSE reduction (%)")
    ax.set_ylim(0, 75)
    ax.set_title("Generalization across processing conditions", loc="left", fontweight="bold")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[4, :])
    panel_label(ax, "G")
    same_sign = genes.loc[np.sign(genes.observed_mean) == np.sign(genes.predicted_mean)].copy()
    same_sign["strength"] = np.minimum(same_sign.observed_mean.abs(), same_sign.predicted_mean.abs())
    selected = same_sign.nlargest(18, "strength")
    matrix = selected[["observed_mean", "predicted_mean"]].to_numpy().T
    limit = np.quantile(np.abs(matrix), 0.95)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_yticks([0, 1], ["Observed", "Predicted"])
    ax.set_xticks(np.arange(len(selected)), [str(value).split("|")[-1] for value in selected.symbols], rotation=60, ha="right", fontsize=8)
    ax.set_title("Concordant high-amplitude microglial response genes", loc="left", fontweight="bold")
    colorbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
    colorbar.set_label("Log-expression response")

    ax = fig.add_subplot(grid[5, :])
    panel_label(ax, "H")
    terms = go.loc[go["query"].eq("concordant_top100")].sort_values("q").drop_duplicates("term").head(8).iloc[::-1]
    labels = [value.split(" (")[0] for value in terms.term]
    values = -np.log10(terms.q)
    ax.barh(np.arange(len(terms)), values, color=TEAL, height=0.65)
    ax.set_yticks(np.arange(len(terms)), labels, fontsize=9)
    ax.set_xlabel("−log10 adjusted P")
    ax.set_title("Shared response programs", loc="left", fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)

    fig.savefig(OUT / "Figure4_dynamic.pdf", facecolor="white")
    fig.savefig(OUT / "Figure4_dynamic.png", dpi=150, facecolor="white")
    print(OUT / "Figure4_dynamic.pdf")
    print(OUT / "Figure4_dynamic.png")


if __name__ == "__main__":
    main()
