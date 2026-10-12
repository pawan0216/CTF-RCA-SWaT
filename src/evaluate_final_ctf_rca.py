import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("data/processed")

rank = pd.read_csv(BASE / "ctf_rca_attack_rankings.csv")
score = pd.read_csv(BASE / "ctf_rca_component_scores.csv")

print("=" * 90)
print("CTF-RCA FINAL EVALUATION")
print("=" * 90)

print("\nRanking columns:")
print(list(rank.columns))

print("\nScore columns:")
print(list(score.columns))

print("\n" + "=" * 90)
print("BASIC DATASET SUMMARY")
print("=" * 90)

print("Attack episodes:", len(rank))
print("Score rows:", len(score))

# Existing metrics
for col in ["Top1_Correct", "Top3_Correct", "Top5_Correct"]:
    if col in rank.columns:
        print(f"{col}: {rank[col].mean():.4f}")

if "Reciprocal_Rank" in rank.columns:
    print("MRR:", rank["Reciprocal_Rank"].mean())

# Rank distribution
if "Best_True_Rank" in rank.columns:
    print("\n" + "=" * 90)
    print("TRUE COMPONENT RANK DISTRIBUTION")
    print("=" * 90)

    r = rank["Best_True_Rank"]

    print("Rank = 1 :", (r == 1).sum())
    print("Rank <= 3:", (r <= 3).sum())
    print("Rank <= 5:", (r <= 5).sum())
    print("Rank > 5 :", (r > 5).sum())
    print("Missing  :", r.isna().sum())

    print("\nRank statistics:")
    print(r.describe())

# Detailed errors
if "Top1_Correct" in rank.columns:
    print("\n" + "=" * 90)
    print("TOP-1 ERRORS")
    print("=" * 90)

    errors = rank[rank["Top1_Correct"] == 0].copy()

    cols = [
        c for c in [
            "Attack_ID",
            "Ground_Truth",
            "Top1_Prediction",
            "Top2_Prediction",
            "Top3_Prediction",
            "Top5_Predictions",
            "Best_True_Rank"
        ] if c in errors.columns
    ]

    print(errors[cols].to_string(index=False))

# Multi-component attacks
if "Ground_Truth" in rank.columns:
    print("\n" + "=" * 90)
    print("SINGLE vs MULTI-COMPONENT ATTACKS")
    print("=" * 90)

    rank["Num_GT_Components"] = (
        rank["Ground_Truth"]
        .fillna("")
        .astype(str)
        .apply(lambda x: 0 if not x.strip() else len(x.split("|")))
    )

    single = rank[rank["Num_GT_Components"] == 1]
    multi = rank[rank["Num_GT_Components"] > 1]

    print("Single-component attacks:", len(single))
    if len(single):
        print("  Top-1:", single["Top1_Correct"].mean())
        print("  Top-3:", single["Top3_Correct"].mean())
        print("  Top-5:", single["Top5_Correct"].mean())
        print("  MRR  :", single["Reciprocal_Rank"].mean())

    print("\nMulti-component attacks:", len(multi))
    if len(multi):
        print("  Top-1:", multi["Top1_Correct"].mean())
        print("  Top-3:", multi["Top3_Correct"].mean())
        print("  Top-5:", multi["Top5_Correct"].mean())
        print("  MRR  :", multi["Reciprocal_Rank"].mean())

print("\n" + "=" * 90)
print("EVALUATION COMPLETE")
print("=" * 90)
