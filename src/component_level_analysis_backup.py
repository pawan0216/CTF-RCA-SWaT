from pathlib import Path
import pandas as pd
import numpy as np
import re

BASE = Path("data/processed")

ranking_file = BASE / "ctf_rca_attack_rankings.csv"
score_file = BASE / "ctf_rca_component_scores.csv"

print("=" * 90)
print("CTF-RCA COMPONENT-LEVEL PERFORMANCE ANALYSIS")
print("=" * 90)

# ------------------------------------------------------------
# Load ranking results
# ------------------------------------------------------------
ranking = pd.read_csv(ranking_file)

print("\nRanking file:")
print(ranking_file)

print("Rows:", len(ranking))

# ------------------------------------------------------------
# Locate columns
# ------------------------------------------------------------
attack_col = "Attack_ID"
gt_col = "Ground_Truth"

required = [
    attack_col,
    gt_col,
    "Top1_Prediction",
    "Top3_Prediction",
    "Top5_Predictions",
    "Reciprocal_Rank",
    "Best_True_Rank"
]

missing = [c for c in required if c not in ranking.columns]

if missing:
    print("\nMissing columns:", missing)
    print("Available:")
    print(list(ranking.columns))
    raise SystemExit(1)

# ------------------------------------------------------------
# Normalize component names
# ------------------------------------------------------------
def normalize_component(x):
    if pd.isna(x):
        return None

    x = str(x).strip()

    if not x or x.lower() == "nan":
        return None

    # Ground truth can contain multiple components
    x = x.replace("-", "")

    return x.upper()


def split_components(x):
    if pd.isna(x):
        return []

    x = str(x).strip()

    if not x or x.lower() == "nan":
        return []

    parts = re.split(r"[|,;]+", x)

    return [
        normalize_component(p)
        for p in parts
        if normalize_component(p) is not None
    ]


# ------------------------------------------------------------
# Build component-level records
# ------------------------------------------------------------
records = []

for _, row in ranking.iterrows():

    attack_id = row[attack_col]

    true_components = split_components(
        row[gt_col]
    )

    if not true_components:
        continue

    top1 = split_components(
        row["Top1_Prediction"]
    )

    top3 = split_components(
        "|".join(
            str(row["Top1_Prediction"]),
            # handled below
        )
    )

    # Explicitly parse top-3/top-5 lists
    top3_string = str(
        row["Top3_Prediction"]
    )

    top5_string = str(
        row["Top5_Predictions"]
    )

    top3_components = split_components(
        top3_string
    )

    top5_components = split_components(
        top5_string
    )

    # The Top3_Prediction column represents the third
    # predicted component, so include Top1 + Top2 + Top3.
    top1_component = normalize_component(
        row["Top1_Prediction"]
    )

    top2_component = normalize_component(
        row["Top2_Prediction"]
    )

    top3_component = normalize_component(
        row["Top3_Prediction"]
    )

    top3_set = {
        x for x in [
            top1_component,
            top2_component,
            top3_component
        ]
        if x
    }

    top5_set = set(
        split_components(
            row["Top5_Predictions"]
        )
    )

    # Add one record for each true component
    for true_component in true_components:

        records.append({
            "Attack_ID": attack_id,
            "True_Component": true_component,
            "Top1": int(
                true_component == top1_component
            ),
            "Top3": int(
                true_component in top3_set
            ),
            "Top5": int(
                true_component in top5_set
            ),
            "Reciprocal_Rank": float(
                row["Reciprocal_Rank"]
            )
            if pd.notna(row["Reciprocal_Rank"])
            else np.nan,
            "True_Rank": float(
                row["Best_True_Rank"]
            )
            if pd.notna(row["Best_True_Rank"])
            else np.nan
        })


component_records = pd.DataFrame(records)

if component_records.empty:
    raise SystemExit(
        "No component-level records could be constructed."
    )

# ------------------------------------------------------------
# Aggregate by true component
# ------------------------------------------------------------
summary = (
    component_records
    .groupby("True_Component")
    .agg(
        Attacks=("Attack_ID", "nunique"),
        Component_Occurrences=("True_Component", "size"),
        Top1=("Top1", "mean"),
        Top3=("Top3", "mean"),
        Top5=("Top5", "mean"),
        MRR=("Reciprocal_Rank", "mean"),
        Mean_Rank=("True_Rank", "mean"),
        Median_Rank=("True_Rank", "median")
    )
    .reset_index()
)

summary = summary.sort_values(
    ["Top1", "Top5", "MRR"],
    ascending=[False, False, False]
)

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------
print("\n" + "=" * 90)
print("COMPONENT-LEVEL PERFORMANCE")
print("=" * 90)

display_df = summary.copy()

display_df["Top1"] = (
    display_df["Top1"] * 100
).round(2)

display_df["Top3"] = (
    display_df["Top3"] * 100
).round(2)

display_df["Top5"] = (
    display_df["Top5"] * 100
).round(2)

display_df["MRR"] = display_df["MRR"].round(4)

display_df["Mean_Rank"] = (
    display_df["Mean_Rank"]
    .round(2)
)

display_df["Median_Rank"] = (
    display_df["Median_Rank"]
    .round(2)
)

print(
    display_df.to_string(
        index=False
    )
)

# ------------------------------------------------------------
# Best and worst components
# ------------------------------------------------------------
print("\n" + "=" * 90)
print("BEST LOCALIZED COMPONENTS")
print("=" * 90)

best = summary.sort_values(
    ["Top1", "Top5", "MRR"],
    ascending=False
).head(10)

print(
    best[
        [
            "True_Component",
            "Attacks",
            "Top1",
            "Top3",
            "Top5",
            "MRR",
            "Mean_Rank"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 90)
print("MOST DIFFICULT COMPONENTS")
print("=" * 90)

worst = summary.sort_values(
    ["Top1", "Top5", "MRR"],
    ascending=True
).head(10)

print(
    worst[
        [
            "True_Component",
            "Attacks",
            "Top1",
            "Top3",
            "Top5",
            "MRR",
            "Mean_Rank"
        ]
    ].to_string(index=False)
)

# ------------------------------------------------------------
# Confusion analysis
# ------------------------------------------------------------
confusion = []

for _, row in ranking.iterrows():

    true_components = split_components(
        row[gt_col]
    )

    if not true_components:
        continue

    predicted = normalize_component(
        row["Top1_Prediction"]
    )

    for true_component in true_components:

        if predicted != true_component:

            confusion.append({
                "Attack_ID": row[attack_col],
                "True_Component": true_component,
                "Predicted_Component": predicted
            })

confusion_df = pd.DataFrame(confusion)

if not confusion_df.empty:

    confusion_counts = (
        confusion_df
        .groupby(
            [
                "True_Component",
                "Predicted_Component"
            ]
        )
        .size()
        .reset_index(
            name="Count"
        )
        .sort_values(
            "Count",
            ascending=False
        )
    )

    print("\n" + "=" * 90)
    print("MOST FREQUENT TOP-1 CONFUSIONS")
    print("=" * 90)

    print(
        confusion_counts
        .head(20)
        .to_string(index=False)
    )

else:

    confusion_counts = pd.DataFrame()

# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------
summary.to_csv(
    BASE / "rca_component_level_performance.csv",
    index=False
)

component_records.to_csv(
    BASE / "rca_component_level_attack_records.csv",
    index=False
)

confusion_df.to_csv(
    BASE / "rca_top1_confusion_records.csv",
    index=False
)

if not confusion_counts.empty:

    confusion_counts.to_csv(
        BASE / "rca_top1_confusion_matrix.csv",
        index=False
    )

print("\n" + "=" * 90)
print("COMPONENT-LEVEL ANALYSIS COMPLETE")
print("=" * 90)

print("\nSaved:")
print(
    "data/processed/rca_component_level_performance.csv"
)
print(
    "data/processed/rca_component_level_attack_records.csv"
)
print(
    "data/processed/rca_top1_confusion_records.csv"
)

if not confusion_counts.empty:
    print(
        "data/processed/rca_top1_confusion_matrix.csv"
    )

