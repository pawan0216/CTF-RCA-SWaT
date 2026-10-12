from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = Path("data/processed")
OUT = BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# ============================================================
# 1. OVERALL PERFORMANCE
# ============================================================

metrics = ["Top-1", "Top-3", "Top-5", "MRR"]
overall = [44.44, 55.56, 69.44, 55.30]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(metrics, overall)

ax.set_ylabel("Performance (%)")
ax.set_title("Overall CTF-RCA Ranking Performance")
ax.set_ylim(0, 100)

for b, v in zip(bars, overall):
    ax.text(
        b.get_x() + b.get_width()/2,
        v + 2,
        f"{v:.2f}%",
        ha="center",
        va="bottom"
    )

ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "Figure_1_Overall_Performance.png", bbox_inches="tight")
plt.close(fig)


# ============================================================
# 2. ABLATION STUDY
# ============================================================

ablation = pd.read_csv(BASE / "rca_ablation_results.csv")

# Normalize possible column names
ablation.columns = [str(c).strip() for c in ablation.columns]

fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(ablation))
width = 0.25

ax.bar(x - width, ablation["Top1"] * 100, width, label="Top-1")
ax.bar(x,         ablation["Top3"] * 100, width, label="Top-3")
ax.bar(x + width, ablation["Top5"] * 100, width, label="Top-5")

ax.set_xticks(x)
ax.set_xticklabels(
    ablation["Experiment"],
    rotation=30,
    ha="right"
)

ax.set_ylabel("Accuracy (%)")
ax.set_title("Ablation Study of CTF-RCA Evidence Components")
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(OUT / "Figure_2_Ablation_Study.png", bbox_inches="tight")
plt.close(fig)


# ============================================================
# 3. BASELINE VS OPTIMIZED VS LOAO
# ============================================================

comparison = pd.DataFrame({
    "Configuration": [
        "Initial Fusion",
        "Optimized Weights",
        "LOAO Validation"
    ],
    "Top1": [44.44, 52.78, 41.67],
    "Top3": [55.56, 63.89, 55.56],
    "Top5": [69.44, 72.22, 69.44]
})

fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(comparison))
width = 0.24

ax.bar(x - width, comparison["Top1"], width, label="Top-1")
ax.bar(x,         comparison["Top3"], width, label="Top-3")
ax.bar(x + width, comparison["Top5"], width, label="Top-5")

ax.set_xticks(x)
ax.set_xticklabels(comparison["Configuration"])
ax.set_ylabel("Accuracy (%)")
ax.set_title("CTF-RCA Performance Across Evaluation Settings")
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUT / "Figure_3_Evaluation_Settings.png",
    bbox_inches="tight"
)
plt.close(fig)


# ============================================================
# 4. LOAO RANK DISTRIBUTION
# ============================================================

loao = pd.read_csv(BASE / "rca_leave_one_attack_out_results.csv")

rank_col = "HeldOut_Rank"
ranks = pd.to_numeric(loao[rank_col], errors="coerce").dropna()

rank_counts = [
    (ranks == 1).sum(),
    ((ranks > 1) & (ranks <= 3)).sum(),
    ((ranks > 3) & (ranks <= 5)).sum(),
    (ranks > 5).sum()
]

labels = [
    "Rank 1",
    "Rank 2–3",
    "Rank 4–5",
    "Rank >5"
]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, rank_counts)

ax.set_ylabel("Number of attacks")
ax.set_title("True-Component Rank Distribution Under LOAO Validation")

for b, v in zip(bars, rank_counts):
    ax.text(
        b.get_x() + b.get_width()/2,
        v + 0.4,
        str(v),
        ha="center"
    )

ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUT / "Figure_4_LOAO_Rank_Distribution.png",
    bbox_inches="tight"
)
plt.close(fig)


# ============================================================
# 5. COMPONENT-LEVEL PERFORMANCE
# ============================================================

component = pd.read_csv(
    BASE / "rca_component_level_performance.csv"
)

component = component.sort_values(
    "Top1",
    ascending=True
)

# Keep components with at least one occurrence
component = component[component["Attacks"] > 0]

fig, ax = plt.subplots(figsize=(9, 10))

ax.barh(
    component["True_Component"],
    component["Top1"] * 100
)

ax.set_xlabel("Top-1 Accuracy (%)")
ax.set_ylabel("True Component")
ax.set_title("Component-Level RCA Localization Performance")
ax.set_xlim(0, 100)
ax.grid(axis="x", alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUT / "Figure_5_Component_Level_Performance.png",
    bbox_inches="tight"
)
plt.close(fig)


# ============================================================
# 6. COMPONENT MRR
# ============================================================

component_mrr = component.sort_values(
    "MRR",
    ascending=True
)

fig, ax = plt.subplots(figsize=(9, 10))

ax.barh(
    component_mrr["True_Component"],
    component_mrr["MRR"]
)

ax.set_xlabel("Mean Reciprocal Rank")
ax.set_ylabel("True Component")
ax.set_title("Component-Level Mean Reciprocal Rank")
ax.set_xlim(0, 1)
ax.grid(axis="x", alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUT / "Figure_6_Component_MRR.png",
    bbox_inches="tight"
)
plt.close(fig)


# ============================================================
# 7. TOP-1 CONFUSION ANALYSIS
# ============================================================

conf = pd.read_csv(
    BASE / "rca_top1_confusion_records.csv"
)

# Aggregate most frequent confusions
conf.columns = [str(c).strip() for c in conf.columns]

if len(conf) > 0:

    counts = (
        conf.groupby(
            ["True_Component", "Predicted_Component"]
        )
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
        .head(15)
    )

    counts["Pair"] = (
        counts["True_Component"]
        + " → "
        + counts["Predicted_Component"]
    )

    counts = counts.sort_values("Count", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.barh(
        counts["Pair"],
        counts["Count"]
    )

    ax.set_xlabel("Number of errors")
    ax.set_ylabel("True → Predicted")
    ax.set_title("Most Frequent Top-1 RCA Confusions")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUT / "Figure_7_Top1_Confusions.png",
        bbox_inches="tight"
    )
    plt.close(fig)


# ============================================================
# 8. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

ci = pd.read_csv(
    BASE / "rca_bootstrap_confidence_intervals.csv"
)

print("=" * 90)
print("CTF-RCA PUBLICATION FIGURES")
print("=" * 90)

print(f"Output directory: {OUT}")
print()

for f in sorted(OUT.glob("*.png")):
    print(f"Created: {f.name}")

print()
print("=" * 90)
print("FIGURE GENERATION COMPLETE")
print("=" * 90)