from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("data/processed")
OUT = BASE / "tables"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("CTF-RCA PUBLICATION TABLE GENERATION")
print("=" * 90)

# ============================================================
# TABLE 1 — DATASET / ATTACK SUMMARY
# ============================================================

windows = pd.read_csv(BASE / "rca_attack_windows.csv")

table1 = windows.copy()

table1["Attack_ID"] = (
    pd.to_numeric(table1["Attack_ID"], errors="coerce")
    .astype("Int64")
)

cols = [
    "Attack_ID",
    "Start_Timestamp",
    "End_Timestamp",
    "Duration_seconds",
    "Ground_Truth_Root_Cause"
]

table1 = table1[
    [c for c in cols if c in table1.columns]
]

table1.to_csv(
    OUT / "Table_1_Attack_Episodes.csv",
    index=False
)


# ============================================================
# TABLE 2 — OVERALL PERFORMANCE
# ============================================================

table2 = pd.DataFrame({
    "Metric": [
        "Top-1 Accuracy",
        "Top-3 Accuracy",
        "Top-5 Accuracy",
        "Mean Reciprocal Rank"
    ],
    "Initial_Fusion": [
        44.44,
        55.56,
        69.44,
        0.5530
    ],
    "Optimized_Weights": [
        52.78,
        63.89,
        72.22,
        0.6088
    ],
    "LOAO_Unseen_Attacks": [
        41.67,
        55.56,
        69.44,
        0.5271
    ]
})

table2.to_csv(
    OUT / "Table_2_Overall_Performance.csv",
    index=False
)


# ============================================================
# TABLE 3 — ABLATION STUDY
# ============================================================

ablation = pd.read_csv(
    BASE / "rca_ablation_results.csv"
)

ablation.columns = [
    str(c).strip()
    for c in ablation.columns
]

table3 = ablation.copy()

for c in ["Top1", "Top3", "Top5"]:
    if c in table3.columns:
        table3[c] = table3[c] * 100

table3.to_csv(
    OUT / "Table_3_Ablation_Study.csv",
    index=False
)


# ============================================================
# TABLE 4 — WEIGHT OPTIMIZATION
# ============================================================

weights = pd.read_csv(
    BASE / "rca_weight_optimization.csv"
)

weights.columns = [
    str(c).strip()
    for c in weights.columns
]

table4 = weights.sort_values(
    "MRR",
    ascending=False
).head(20)

table4.to_csv(
    OUT / "Table_4_Top_20_Weight_Configurations.csv",
    index=False
)


# ============================================================
# TABLE 5 — LOAO VALIDATION
# ============================================================

loao = pd.read_csv(
    BASE / "rca_leave_one_attack_out_results.csv"
)

loao.columns = [
    str(c).strip()
    for c in loao.columns
]

table5 = loao.copy()

table5.to_csv(
    OUT / "Table_5_LOAO_Attack_Level_Results.csv",
    index=False
)


# ============================================================
# TABLE 6 — BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

ci = pd.read_csv(
    BASE / "rca_bootstrap_confidence_intervals.csv"
)

ci.columns = [
    str(c).strip()
    for c in ci.columns
]

ci.to_csv(
    OUT / "Table_6_Bootstrap_Confidence_Intervals.csv",
    index=False
)


# ============================================================
# TABLE 7 — COMPONENT LEVEL PERFORMANCE
# ============================================================

component = pd.read_csv(
    BASE / "rca_component_level_performance.csv"
)

component.columns = [
    str(c).strip()
    for c in component.columns
]

table7 = component.sort_values(
    ["Top1", "MRR"],
    ascending=[False, False]
)

table7.to_csv(
    OUT / "Table_7_Component_Level_Performance.csv",
    index=False
)


# ============================================================
# TABLE 8 — TOP-1 CONFUSIONS
# ============================================================

conf = pd.read_csv(
    BASE / "rca_top1_confusion_records.csv"
)

conf.columns = [
    str(c).strip()
    for c in conf.columns
]

if len(conf) > 0:

    table8 = (
        conf.groupby(
            ["True_Component", "Predicted_Component"]
        )
        .size()
        .reset_index(name="Confusion_Count")
        .sort_values(
            "Confusion_Count",
            ascending=False
        )
    )

else:
    table8 = pd.DataFrame(
        columns=[
            "True_Component",
            "Predicted_Component",
            "Confusion_Count"
        ]
    )

table8.to_csv(
    OUT / "Table_8_Top1_Confusions.csv",
    index=False
)


# ============================================================
# TABLE 9 — STATISTICAL COMPARISON
# ============================================================

paired = pd.read_csv(
    BASE / "rca_paired_statistical_comparison.csv"
)

paired.columns = [
    str(c).strip()
    for c in paired.columns
]

paired.to_csv(
    OUT / "Table_9_Paired_Statistical_Comparison.csv",
    index=False
)


# ============================================================
# TABLE 10 — PROCESS DEPENDENCY GRAPH
# ============================================================

dependency = pd.read_csv(
    BASE / "rca_process_dependency_graph.csv"
)

dependency.columns = [
    str(c).strip()
    for c in dependency.columns
]

dependency.to_csv(
    OUT / "Table_10_Process_Dependencies.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("Tables created:")
print()

for f in sorted(OUT.glob("*.csv")):
    print(f"  {f.name}")

print()
print("=" * 90)
print("TABLE GENERATION COMPLETE")
print("=" * 90)
