from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product

BASE = Path("data/processed")
INPUT = BASE / "ctf_rca_component_scores.csv"
OUTPUT = BASE / "rca_leave_one_attack_out_results.csv"

print("=" * 90)
print("CTF-RCA LEAVE-ONE-ATTACK-OUT WEIGHT VALIDATION")
print("=" * 90)

df = pd.read_csv(INPUT, low_memory=False)

evidence = [
    "Anomaly_Evidence",
    "Onset_Evidence",
    "Persistence_Evidence",
    "Transition_Evidence",
    "Network_Evidence",
    "Dependency_Evidence",
]

for c in evidence:
    df[c] = pd.to_numeric(
        df[c], errors="coerce"
    ).fillna(0.0)


def norm_component(x):
    if pd.isna(x):
        return None

    return str(x).strip().upper().replace("-", "")


def get_ground_truth(row_group):
    truth = set()

    for value in row_group["Ground_Truth_Root_Cause"].dropna():
        for item in str(value).split("|"):
            item = norm_component(item)
            if item:
                truth.add(item)

    return truth


# ---------------------------------------------------------------------
# BUILD WEIGHT GRID
# ---------------------------------------------------------------------

grid = np.arange(0.0, 1.01, 0.10)

weights = np.array(
    [
        w
        for w in product(grid, repeat=6)
        if abs(sum(w) - 1.0) < 1e-9
    ],
    dtype=np.float32,
)

print(f"Weight configurations: {len(weights):,}")

# ---------------------------------------------------------------------
# PREPARE ATTACK DATA
# ---------------------------------------------------------------------

attacks = {}

for attack_id, g in df.groupby("Attack_ID"):

    g = g.copy()

    g["Component_norm"] = (
        g["Component_norm"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    X = g[evidence].to_numpy(dtype=np.float32)

    components = g["Component_norm"].to_numpy()

    truth = get_ground_truth(g)

    attacks[int(attack_id)] = {
        "X": X,
        "components": components,
        "truth": truth,
    }


attack_ids = sorted(attacks)

print(f"Attack episodes: {len(attack_ids)}")

# ---------------------------------------------------------------------
# PRECOMPUTE RANK OF TRUE COMPONENT FOR EVERY
# ATTACK × WEIGHT CONFIGURATION
# ---------------------------------------------------------------------

print("\nCalculating configuration performance for each attack...")

rank_matrix = np.full(
    (len(attack_ids), len(weights)),
    np.nan,
    dtype=np.float32,
)

for ai, attack_id in enumerate(attack_ids):

    data = attacks[attack_id]

    X = data["X"]
    components = data["components"]
    truth = data["truth"]

    if not truth:
        continue

    # Score:
    # configurations × components
    scores = weights @ X.T

    # For each configuration, rank components
    # by descending RCA score.
    order = np.argsort(
        -scores,
        axis=1,
        kind="stable"
    )

    sorted_components = components[order]

    # Best rank among all ground-truth components.
    for config_idx in range(len(weights)):

        ranked = sorted_components[config_idx]

        best_rank = None

        for rank, component in enumerate(
            ranked,
            start=1
        ):
            if component in truth:
                best_rank = rank
                break

        if best_rank is not None:
            rank_matrix[ai, config_idx] = best_rank

    print(
        f"Processed attack "
        f"{attack_id:2d} "
        f"({ai + 1}/{len(attack_ids)})"
    )

# ---------------------------------------------------------------------
# LEAVE-ONE-ATTACK-OUT
# ---------------------------------------------------------------------

print("\n" + "=" * 90)
print("LEAVE-ONE-ATTACK-OUT OPTIMIZATION")
print("=" * 90)

results = []

for held_idx, held_attack in enumerate(attack_ids):

    training_indices = [
        i
        for i in range(len(attack_ids))
        if i != held_idx
    ]

    train_ranks = rank_matrix[
        training_indices,
        :
    ]

    # MRR on training attacks.
    train_rr = np.where(
        np.isnan(train_ranks),
        0.0,
        1.0 / train_ranks
    )

    train_mrr = np.mean(
        train_rr,
        axis=0
    )

    # Select best configuration using ONLY
    # the other attacks.
    best_idx = np.argmax(train_mrr)

    best_weights = weights[best_idx]

    held_rank = rank_matrix[
        held_idx,
        best_idx
    ]

    if np.isnan(held_rank):
        held_rank_value = None
    else:
        held_rank_value = int(held_rank)

    results.append({
        "HeldOut_Attack": held_attack,

        "Anomaly": float(best_weights[0]),
        "Onset": float(best_weights[1]),
        "Persistence": float(best_weights[2]),
        "Transition": float(best_weights[3]),
        "Network": float(best_weights[4]),
        "Dependency": float(best_weights[5]),

        "Training_MRR": float(train_mrr[best_idx]),

        "HeldOut_Rank": held_rank_value,

        "HeldOut_Top1": (
            int(held_rank_value is not None
                and held_rank_value <= 1)
        ),

        "HeldOut_Top3": (
            int(held_rank_value is not None
                and held_rank_value <= 3)
        ),

        "HeldOut_Top5": (
            int(held_rank_value is not None
                and held_rank_value <= 5)
        ),

        "HeldOut_RR": (
            1.0 / held_rank_value
            if held_rank_value is not None
            else 0.0
        ),
    })

    print(
        f"Attack {held_attack:2d} held out | "
        f"Training MRR={train_mrr[best_idx]:.4f} | "
        f"Selected weights="
        f"{best_weights.round(1)} | "
        f"Held-out rank={held_rank_value}"
    )

results = pd.DataFrame(results)

results.to_csv(
    OUTPUT,
    index=False
)

# ---------------------------------------------------------------------
# FINAL AGGREGATED RESULTS
# ---------------------------------------------------------------------

top1 = results["HeldOut_Top1"].mean()
top3 = results["HeldOut_Top3"].mean()
top5 = results["HeldOut_Top5"].mean()
mrr = results["HeldOut_RR"].mean()

valid_ranks = results[
    results["HeldOut_Rank"].notna()
]["HeldOut_Rank"]

print("\n" + "=" * 90)
print("FINAL UNSEEN-ATTACK PERFORMANCE")
print("=" * 90)

print(f"Attacks evaluated : {len(results)}")
print(f"Top-1 Accuracy    : {top1 * 100:.2f}%")
print(f"Top-3 Accuracy    : {top3 * 100:.2f}%")
print(f"Top-5 Accuracy    : {top5 * 100:.2f}%")
print(f"MRR               : {mrr:.4f}")

print("\n" + "-" * 90)
print("TRUE COMPONENT RANKS")
print("-" * 90)

print(
    f"Rank = 1 : "
    f"{(valid_ranks == 1).sum()}"
)

print(
    f"Rank <= 3: "
    f"{(valid_ranks <= 3).sum()}"
)

print(
    f"Rank <= 5: "
    f"{(valid_ranks <= 5).sum()}"
)

print(
    f"Rank > 5 : "
    f"{(valid_ranks > 5).sum()}"
)

print(
    f"Missing   : "
    f"{results['HeldOut_Rank'].isna().sum()}"
)

# ---------------------------------------------------------------------
# WEIGHT FREQUENCY
# ---------------------------------------------------------------------

print("\n" + "=" * 90)
print("MOST FREQUENTLY SELECTED WEIGHT CONFIGURATIONS")
print("=" * 90)

weight_cols = [
    "Anomaly",
    "Onset",
    "Persistence",
    "Transition",
    "Network",
    "Dependency",
]

frequency = (
    results
    .groupby(weight_cols)
    .size()
    .reset_index(name="Folds")
    .sort_values("Folds", ascending=False)
)

print(
    frequency.head(15).to_string(
        index=False
    )
)

print("\n" + "=" * 90)
print("LOAO VALIDATION COMPLETE")
print("=" * 90)

print(f"Output: {OUTPUT}")
