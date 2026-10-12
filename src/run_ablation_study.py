from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("data/processed")

# ---------------------------------------------------------------------
# Locate final RCA score file
# ---------------------------------------------------------------------
candidates = [
    BASE / "ctf_rca_component_scores.csv",
    BASE / "ctf_rca_scores.csv",
    BASE / "final_rca_scores.csv",
]

SCORE_FILE = None
for p in candidates:
    if p.exists():
        SCORE_FILE = p
        break

if SCORE_FILE is None:
    raise FileNotFoundError(
        "Could not find the final RCA score file. "
        "Expected one of: " + ", ".join(str(x) for x in candidates)
    )

print("=" * 90)
print("CTF-RCA ABLATION STUDY")
print("=" * 90)
print("Score file:", SCORE_FILE)

df = pd.read_csv(SCORE_FILE, low_memory=False)

required = [
    "Attack_ID",
    "Component_norm",
    "Ground_Truth_Root_Cause",
    "Anomaly_Evidence",
    "Onset_Evidence",
    "Persistence_Evidence",
    "Transition_Evidence",
    "Network_Evidence",
    "Dependency_Evidence",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------
for c in required:
    if c not in ["Attack_ID", "Ground_Truth_Root_Cause", "Component_norm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

df["Component_norm"] = (
    df["Component_norm"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# ---------------------------------------------------------------------
# Normalize ground truth into individual components
# ---------------------------------------------------------------------
def normalize_component(x):
    if pd.isna(x):
        return set()

    x = str(x).upper()

    parts = []
    for p in x.split("|"):
        p = p.strip()
        if not p:
            continue

        p = p.replace("-", "")
        parts.append(p)

    return set(parts)


gt_by_attack = {}

for attack_id, group in df.groupby("Attack_ID"):
    values = set()

    for x in group["Ground_Truth_Root_Cause"].dropna():
        values.update(normalize_component(x))

    gt_by_attack[attack_id] = values


# ---------------------------------------------------------------------
# Evidence configurations
# ---------------------------------------------------------------------
experiments = {

    "E1_Anomaly_Only": {
        "Anomaly_Evidence": 1.0,
    },

    "E2_Anomaly_Onset": {
        "Anomaly_Evidence": 0.70,
        "Onset_Evidence": 0.30,
    },

    "E3_Physical_Temporal": {
        "Anomaly_Evidence": 0.40,
        "Onset_Evidence": 0.20,
        "Persistence_Evidence": 0.20,
        "Transition_Evidence": 0.20,
    },

    "E4_Physical_Network": {
        "Anomaly_Evidence": 0.30,
        "Onset_Evidence": 0.15,
        "Persistence_Evidence": 0.15,
        "Transition_Evidence": 0.15,
        "Network_Evidence": 0.25,
    },

    "E5_Physical_Dependency": {
        "Anomaly_Evidence": 0.30,
        "Onset_Evidence": 0.15,
        "Persistence_Evidence": 0.15,
        "Transition_Evidence": 0.15,
        "Dependency_Evidence": 0.25,
    },

    "E6_Physical_Network_Dependency": {
        "Anomaly_Evidence": 0.25,
        "Onset_Evidence": 0.15,
        "Persistence_Evidence": 0.15,
        "Transition_Evidence": 0.15,
        "Network_Evidence": 0.15,
        "Dependency_Evidence": 0.15,
    },
}


# ---------------------------------------------------------------------
# Calculate ranking
# ---------------------------------------------------------------------
def calculate_score(group, weights):

    score = np.zeros(len(group))

    for col, weight in weights.items():
        score += group[col].to_numpy(dtype=float) * weight

    return score


def evaluate(weights):

    results = []

    for attack_id, group in df.groupby("Attack_ID"):

        gt = gt_by_attack.get(attack_id, set())

        # Exclude episodes without an identifiable physical GT
        if not gt:
            continue

        temp = group.copy()

        temp["_score"] = calculate_score(temp, weights)

        temp = temp.sort_values(
            "_score",
            ascending=False
        ).reset_index(drop=True)

        predictions = temp["Component_norm"].tolist()

        # Remove duplicate component names while preserving ranking
        unique_predictions = []
        seen = set()

        for p in predictions:
            if p not in seen:
                unique_predictions.append(p)
                seen.add(p)

        rank = None

        for i, pred in enumerate(unique_predictions, start=1):
            if pred in gt:
                rank = i
                break

        if rank is None:
            reciprocal = 0.0
        else:
            reciprocal = 1.0 / rank

        results.append({
            "Attack_ID": attack_id,
            "Rank": rank,
            "Top1": int(rank is not None and rank <= 1),
            "Top3": int(rank is not None and rank <= 3),
            "Top5": int(rank is not None and rank <= 5),
            "Reciprocal_Rank": reciprocal,
        })

    r = pd.DataFrame(results)

    if len(r) == 0:
        return {
            "Attacks": 0,
            "Top1": np.nan,
            "Top3": np.nan,
            "Top5": np.nan,
            "MRR": np.nan,
        }, r

    metrics = {
        "Attacks": len(r),
        "Top1": r["Top1"].mean(),
        "Top3": r["Top3"].mean(),
        "Top5": r["Top5"].mean(),
        "MRR": r["Reciprocal_Rank"].mean(),
    }

    return metrics, r


# ---------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------
summary = []
detail_tables = {}

for name, weights in experiments.items():

    metrics, detail = evaluate(weights)

    detail["Experiment"] = name

    detail_tables[name] = detail

    summary.append({
        "Experiment": name,
        "Attacks": metrics["Attacks"],
        "Top1": metrics["Top1"],
        "Top3": metrics["Top3"],
        "Top5": metrics["Top5"],
        "MRR": metrics["MRR"],
    })


summary_df = pd.DataFrame(summary)

# Percentage representation
for c in ["Top1", "Top3", "Top5"]:
    summary_df[c] = summary_df[c] * 100

# ---------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("ABLATION RESULTS")
print("=" * 90)

print(
    summary_df.to_string(
        index=False,
        formatters={
            "Top1": "{:.2f}%".format,
            "Top3": "{:.2f}%".format,
            "Top5": "{:.2f}%".format,
            "MRR": "{:.4f}".format,
        },
    )
)

# ---------------------------------------------------------------------
# Improvement relative to anomaly-only baseline
# ---------------------------------------------------------------------
baseline = summary_df.iloc[0]

comparison = summary_df.copy()

comparison["Top1_Improvement_pp"] = (
    comparison["Top1"] - baseline["Top1"]
)

comparison["Top3_Improvement_pp"] = (
    comparison["Top3"] - baseline["Top3"]
)

comparison["Top5_Improvement_pp"] = (
    comparison["Top5"] - baseline["Top5"]
)

comparison["MRR_Improvement"] = (
    comparison["MRR"] - baseline["MRR"]
)

print("\n" + "=" * 90)
print("IMPROVEMENT OVER ANOMALY-ONLY BASELINE")
print("=" * 90)

print(
    comparison[
        [
            "Experiment",
            "Top1_Improvement_pp",
            "Top3_Improvement_pp",
            "Top5_Improvement_pp",
            "MRR_Improvement",
        ]
    ].to_string(
        index=False,
        formatters={
            "Top1_Improvement_pp": "{:+.2f}".format,
            "Top3_Improvement_pp": "{:+.2f}".format,
            "Top5_Improvement_pp": "{:+.2f}".format,
            "MRR_Improvement": "{:+.4f}".format,
        },
    )
)

# ---------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------
out_summary = BASE / "rca_ablation_results.csv"
out_detail = BASE / "rca_ablation_attack_details.csv"

summary_df.to_csv(out_summary, index=False)

all_details = pd.concat(
    detail_tables.values(),
    ignore_index=True
)

all_details.to_csv(out_detail, index=False)

print("\n" + "=" * 90)
print("ABLATION STUDY COMPLETE")
print("=" * 90)
print("Summary :", out_summary)
print("Details :", out_detail)

# ---------------------------------------------------------------------
# Identify best model
# ---------------------------------------------------------------------
best = summary_df.sort_values(
    ["MRR", "Top1", "Top5"],
    ascending=False
).iloc[0]

print("\nBEST ABLATION CONFIGURATION")
print("-" * 90)
print("Experiment :", best["Experiment"])
print("Top-1      :", f"{best['Top1']:.2f}%")
print("Top-3      :", f"{best['Top3']:.2f}%")
print("Top-5      :", f"{best['Top5']:.2f}%")
print("MRR        :", f"{best['MRR']:.4f}")
