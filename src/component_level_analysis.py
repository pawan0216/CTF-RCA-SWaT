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
# Load data
# ------------------------------------------------------------
ranking = pd.read_csv(ranking_file)
score = pd.read_csv(score_file)

print("\nRanking file:", ranking_file)
print("Ranking rows:", len(ranking))
print("Score file:", score_file)
print("Score rows:", len(score))

# ------------------------------------------------------------
# Normalize component names
# ------------------------------------------------------------
def normalize_component(x):
    if pd.isna(x):
        return None

    x = str(x).strip()

    if not x or x.lower() == "nan":
        return None

    x = x.replace("-", "")
    return x.upper()


def split_components(x):
    if pd.isna(x):
        return []

    x = str(x).strip()

    if not x or x.lower() == "nan":
        return []

    parts = re.split(r"[|,;]+", x)

    result = []

    for p in parts:
        value = normalize_component(p)
        if value is not None:
            result.append(value)

    return result


# ------------------------------------------------------------
# Verify required score columns
# ------------------------------------------------------------
required_score = [
    "Attack_ID",
    "Component_norm",
    "RCA_Score",
    "Rank"
]

missing_score = [
    c for c in required_score
    if c not in score.columns
]

if missing_score:
    raise SystemExit(
        f"Missing required score columns: {missing_score}"
    )

# ------------------------------------------------------------
# Build authoritative per-attack component ranking lookup
# ------------------------------------------------------------
score["Attack_ID"] = pd.to_numeric(
    score["Attack_ID"],
    errors="coerce"
)

score["Rank"] = pd.to_numeric(
    score["Rank"],
    errors="coerce"
)

score["Component_norm"] = score[
    "Component_norm"
].apply(normalize_component)

score = score.dropna(
    subset=["Attack_ID", "Component_norm", "Rank"]
)

score["Attack_ID"] = score[
    "Attack_ID"
].astype(int)

score["Rank"] = score[
    "Rank"
].astype(int)

# Key = (Attack_ID, Component)
rank_lookup = {
    (int(row["Attack_ID"]), row["Component_norm"]):
        int(row["Rank"])
    for _, row in score.iterrows()
}

# ------------------------------------------------------------
# Build component-level records
# ------------------------------------------------------------
records = []

for _, row in ranking.iterrows():

    attack_id = int(row["Attack_ID"])

    true_components = split_components(
        row["Ground_Truth"]
    )

    if not true_components:
        continue

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
        if x is not None
    }

    top5_set = set(
        split_components(
            row["Top5_Predictions"]
        )
    )

    # --------------------------------------------------------
    # Individual rank for EACH true component
    # --------------------------------------------------------
    for true_component in true_components:

        key = (
            attack_id,
            true_component
        )

        true_rank = rank_lookup.get(key)

        if true_rank is None:
            print(
                f"WARNING: no rank found for "
                f"Attack {attack_id}, "
                f"component {true_component}"
            )
            continue

        reciprocal_rank = 1.0 / true_rank

        records.append({
            "Attack_ID": attack_id,
            "True_Component": true_component,
            "Top1": int(true_rank <= 1),
            "Top3": int(true_rank <= 3),
            "Top5": int(true_rank <= 5),
            "Reciprocal_Rank": reciprocal_rank,
            "True_Rank": true_rank
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
# Display results
# ------------------------------------------------------------
print("\n" + "=" * 90)
print("CORRECTED COMPONENT-LEVEL PERFORMANCE")
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

display_df["MRR"] = (
    display_df["MRR"]
    .round(4)
)

display_df["Mean_Rank"] = (
    display_df["Mean_Rank"]
    .round(2)
)

display_df["Median_Rank"] = (
    display_df["Median_Rank"]
    .round(2)
)

print(
    display_df.to_string(index=False)
)

# ------------------------------------------------------------
# Explicit validation of previously problematic components
# ------------------------------------------------------------
print("\n" + "=" * 90)
print("VALIDATION OF MULTI-COMPONENT CASES")
print("=" * 90)

check_components = [
    "AIT402",
    "AIT502",
    "MV302",
    "P602",
    "MV101",
    "P101",
    "P501"
]

check = display_df[
    display_df["True_Component"].isin(
        check_components
    )
]

print(
    check.to_string(index=False)
)

# ------------------------------------------------------------
# Verify internal mathematical consistency
# ------------------------------------------------------------
print("\n" + "=" * 90)
print("CONSISTENCY CHECK")
print("=" * 90)

expected_mrr = (
    component_records
    .groupby("True_Component")["Reciprocal_Rank"]
    .mean()
)

expected_mean_rank = (
    component_records
    .groupby("True_Component")["True_Rank"]
    .mean()
)

if np.allclose(
    summary.set_index("True_Component")["MRR"]
    .sort_index()
    .values,
    expected_mrr.sort_index().values
):
    print("MRR consistency: PASS")
else:
    print("MRR consistency: FAIL")

if np.allclose(
    summary.set_index("True_Component")["Mean_Rank"]
    .sort_index()
    .values,
    expected_mean_rank.sort_index().values
):
    print("Mean-rank consistency: PASS")
else:
    print("Mean-rank consistency: FAIL")

print("\nComponent records:", len(component_records))
print("Components:", len(summary))

print("\n" + "=" * 90)
print("COMPONENT-LEVEL ANALYSIS COMPLETE")
print("=" * 90)
summary = summary.sort_values(
    ["Top1", "Top5", "MRR"],
    ascending=[False, False, False]
)

summary.to_csv(
    BASE / "rca_component_level_performance.csv",
    index=False
)
