from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product

BASE = Path("data/processed")

SCORE_FILE = BASE / "ctf_rca_component_scores.csv"

df = pd.read_csv(SCORE_FILE, low_memory=False)

evidence = [
    "Anomaly_Evidence",
    "Onset_Evidence",
    "Persistence_Evidence",
    "Transition_Evidence",
    "Network_Evidence",
    "Dependency_Evidence",
]

for c in evidence:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

df["Component_norm"] = (
    df["Component_norm"]
    .astype(str)
    .str.upper()
    .str.strip()
)


def normalize_gt(x):
    if pd.isna(x):
        return set()

    return {
        p.strip().replace("-", "").upper()
        for p in str(x).split("|")
        if p.strip()
    }


gt = {}

for attack_id, g in df.groupby("Attack_ID"):
    s = set()

    for x in g["Ground_Truth_Root_Cause"].dropna():
        s.update(normalize_gt(x))

    gt[attack_id] = s


def evaluate(weights):

    records = []

    for attack_id, g in df.groupby("Attack_ID"):

        truth = gt.get(attack_id, set())

        if not truth:
            continue

        score = np.zeros(len(g))

        for col, w in zip(evidence, weights):
            score += g[col].to_numpy() * w

        tmp = g.copy()
        tmp["_score"] = score

        tmp = tmp.sort_values(
            "_score",
            ascending=False
        )

        ranked = []
        seen = set()

        for x in tmp["Component_norm"]:
            if x not in seen:
                ranked.append(x)
                seen.add(x)

        rank = None

        for i, x in enumerate(ranked, start=1):
            if x in truth:
                rank = i
                break

        records.append(rank)

    ranks = np.array([
        r for r in records if r is not None
    ])

    if len(records) == 0:
        return 0, 0, 0, 0

    top1 = np.mean([
        r is not None and r <= 1
        for r in records
    ])

    top3 = np.mean([
        r is not None and r <= 3
        for r in records
    ])

    top5 = np.mean([
        r is not None and r <= 5
        for r in records
    ])

    mrr = np.mean([
        1/r if r is not None else 0
        for r in records
    ])

    return top1, top3, top5, mrr


print("=" * 90)
print("CTF-RCA WEIGHT OPTIMIZATION")
print("=" * 90)

print("Evaluating evidence-weight combinations...")

results = []

# Coarse grid.
# Each weight is a multiple of 0.10.
grid = np.arange(0.0, 1.01, 0.10)

count = 0

for weights in product(grid, repeat=len(evidence)):

    if abs(sum(weights) - 1.0) > 1e-9:
        continue

    count += 1

    top1, top3, top5, mrr = evaluate(weights)

    results.append({
        "Anomaly": weights[0],
        "Onset": weights[1],
        "Persistence": weights[2],
        "Transition": weights[3],
        "Network": weights[4],
        "Dependency": weights[5],
        "Top1": top1,
        "Top3": top3,
        "Top5": top5,
        "MRR": mrr,
    })

print("Configurations evaluated:", count)

results = pd.DataFrame(results)

results = results.sort_values(
    ["MRR", "Top1", "Top5"],
    ascending=False
).reset_index(drop=True)

out = BASE / "rca_weight_optimization.csv"

results.to_csv(out, index=False)

print("\n" + "=" * 90)
print("TOP 20 WEIGHT CONFIGURATIONS")
print("=" * 90)

display_cols = [
    "Anomaly",
    "Onset",
    "Persistence",
    "Transition",
    "Network",
    "Dependency",
    "Top1",
    "Top3",
    "Top5",
    "MRR",
]

print(
    results.head(20)[display_cols].to_string(
        index=False,
        formatters={
            "Top1": "{:.4f}".format,
            "Top3": "{:.4f}".format,
            "Top5": "{:.4f}".format,
            "MRR": "{:.4f}".format,
        },
    )
)

best = results.iloc[0]

print("\n" + "=" * 90)
print("BEST WEIGHT CONFIGURATION")
print("=" * 90)

for c in evidence:
    print(
        f"{c:25s}: "
        f"{best[c.replace('_Evidence','')]:.2f}"
    )

print("\nPerformance")
print("-" * 90)
print(f"Top-1 : {best['Top1']*100:.2f}%")
print(f"Top-3 : {best['Top3']*100:.2f}%")
print(f"Top-5 : {best['Top5']*100:.2f}%")
print(f"MRR   : {best['MRR']:.4f}")

print("\nSaved:")
print(out)
