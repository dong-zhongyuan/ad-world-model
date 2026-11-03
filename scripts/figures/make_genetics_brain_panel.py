from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
OUT = WORK / "figures/panels"
TEAL = "#2A9D8F"
PURPLE = "#7E6BB0"
DARK = "#263238"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grn = pd.read_csv(WORK / "results/grn_route_v1/grn_marker_response.csv")
    bin1 = pd.read_csv(WORK / "results/bin1_route_v1/bin1_marker_responses.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "text.color": DARK, "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK})
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), gridspec_kw={"wspace": 0.60})
    fig.subplots_adjust(left=0.12, right=0.985, top=0.79, bottom=0.22)

    names = ["Gpnmb", "Ctsd", "Ctsb", "Lamp1", "Cst7", "Apoe", "Tyrobp"]
    frame = grn.set_index("marker").loc[names]
    values = frame.mean_ko_response.to_numpy()
    errors = frame.sd_ko_response.to_numpy()
    ax = axes[0]
    ax.barh(np.arange(len(names)), values, xerr=errors, color=TEAL, height=0.62, error_kw={"lw": 0.6, "capsize": 2})
    ax.set_yticks(np.arange(len(names)), [value.upper() for value in names])
    ax.invert_yaxis()
    ax.set_xlabel("Grn-KO response (log expression)")
    ax.set_title("GRN | aged Grn-KO microglia\n5/5 KO mice direction concordant", loc="left", fontsize=9, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlim(0, 0.68)

    names = ["Cx3cr1", "Apoe", "Trem2", "Tyrobp", "Cst3", "Ctsd", "Ifitm3"]
    frame = bin1.set_index("marker").loc[names]
    values = frame.Bin1cKO_baseline.to_numpy()
    ax = axes[1]
    ax.barh(np.arange(len(names)), values, color=[PURPLE if value < 0 else TEAL for value in values], height=0.62)
    ax.set_yticks(np.arange(len(names)), [value.upper() for value in names])
    ax.invert_yaxis()
    ax.axvline(0, color=DARK, lw=0.6)
    ax.set_xlabel("Bin1cKO response (log expression)")
    ax.set_title("BIN1 | microglial Bin1cKO\n2,399 annotated microglial cells", loc="left", fontsize=9, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlim(-0.52, 0.62)

    fig.savefig(OUT / "4h.pdf", facecolor="white")
    fig.savefig(OUT / "4h.png", dpi=300, facecolor="white")
    print(OUT / "4h.pdf")


if __name__ == "__main__":
    main()
