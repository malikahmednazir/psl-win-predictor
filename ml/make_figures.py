"""Paper figures (PNG, 200 dpi) from the result CSVs in public/research-data/."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "public" / "research-data"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def fig_reproduction():
    p = OUT / "e1_paper_reproduction_ipl.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d.Phase == "1-20"].dropna(subset=["Paper_test_accuracy"]).sort_values("Paper_test_accuracy")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(d) + 1.2))
    ax.barh(y - 0.2, d.Paper_test_accuracy, 0.4, label="Paper (Goel et al. 2021)", color="#94a3b8")
    ax.barh(y + 0.2, d.Accuracy, 0.4, label="This reproduction", color="#10b981")
    ax.set_yticks(y, d.Model, fontsize=8)
    ax.set_xlim(50, 85)
    ax.set_xlabel("Test accuracy on IPL 2019 (%)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_reproduction_ipl2019.png", dpi=200)
    plt.close(fig)


def fig_novelty():
    p = OUT / "e3_novelty_psl_rolling.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    settings = [s for s in d.Setting.unique() if s[:2] in ("S1", "S2", "S3", "S4")]
    models = list(d.Model.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, lab in [(axes[0], "Accuracy", "Accuracy (%)"), (axes[1], "Brier", "Brier score (lower is better)")]:
        w = 0.8 / len(settings)
        for i, s in enumerate(settings):
            g = d[d.Setting == s].groupby("Model")[key]
            m, sd = g.mean().reindex(models), g.std().reindex(models)
            ax.bar(np.arange(len(models)) + i * w, m, w, yerr=sd, capsize=2, label=s.split(":")[0] + ": " + s.split(":", 1)[-1].strip()[:38])
        ax.set_xticks(np.arange(len(models)) + w * (len(settings) - 1) / 2, models, fontsize=8)
        ax.set_ylabel(lab)
        if key == "Accuracy":
            ax.set_ylim(55, 85)
        else:
            ax.axhline(0.25, color="k", ls="--", lw=1)
    axes[0].legend(fontsize=6.5, loc="lower left")
    fig.suptitle("PSL rolling evaluation, test seasons 2021-2026 (mean ± sd)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_novelty_psl_rolling.png", dpi=200)
    plt.close(fig)


def fig_phases():
    p = OUT / "e2_four_models.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, lg in zip(axes, ["IPL", "PSL"]):
        s = d[(d.League == lg) & d.Split.str.startswith("Season")]
        for _, r in s.iterrows():
            ax.plot(["1-5", "6-10", "11-15", "16-20"],
                    [r["Acc_overs_1-5"], r["Acc_overs_6-10"], r["Acc_overs_11-15"], r["Acc_overs_16-20"]],
                    marker="o", label=r.Model)
        ax.set_title(f"{lg} – {s.Split.iloc[0]}", fontsize=9)
        ax.set_xlabel("Overs of the second innings")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_accuracy_by_phase.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_reproduction()
    fig_novelty()
    fig_phases()
    print("figures written to", FIG)
