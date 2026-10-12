from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("data/processed")

PHYSICAL = BASE / "rca_evidence_matrix_v2.csv"
NETWORK = BASE / "rca_component_network_evidence.csv"
GRAPH = BASE / "rca_process_dependency_graph.csv"
WINDOWS = BASE / "rca_attack_windows.csv"
MAPPING = BASE / "rca_component_mapping.csv"

OUT_SCORES = BASE / "ctf_rca_component_scores.csv"
OUT_RANKING = BASE / "ctf_rca_attack_rankings.csv"


# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------

W_ANOMALY = 0.25
W_ONSET = 0.20
W_PERSISTENCE = 0.15
W_TRANSITION = 0.15
W_NETWORK = 0.15
W_DEPENDENCY = 0.10


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("=" * 90)
print("CTF-RCA FINAL EVIDENCE FUSION")
print("=" * 90)

physical = pd.read_csv(
    PHYSICAL,
    low_memory=False
)

network = pd.read_csv(
    NETWORK,
    low_memory=False
)

graph = pd.read_csv(
    GRAPH,
    low_memory=False
)

windows = pd.read_csv(
    WINDOWS,
    low_memory=False
)

mapping = pd.read_csv(
    MAPPING,
    low_memory=False
)

print(f"Physical evidence rows : {len(physical):,}")
print(f"Network evidence rows  : {len(network):,}")
print(f"Dependency edges       : {len(graph):,}")
print(f"Attack episodes        : {len(windows):,}")


# ------------------------------------------------------------
# NORMALIZE COMPONENT NAMES
# ------------------------------------------------------------

def normalize(x):

    if pd.isna(x):
        return None

    return (
        str(x)
        .strip()
        .upper()
        .replace("-", "")
    )


physical["Component_norm"] = (
    physical["Component"]
    .apply(normalize)
)

network["Component_norm"] = (
    network["Component"]
    .apply(normalize)
)

graph["Source_norm"] = (
    graph["Source"]
    .apply(normalize)
)

graph["Target_norm"] = (
    graph["Target"]
    .apply(normalize)
)


# ------------------------------------------------------------
# PHYSICAL EVIDENCE
# ------------------------------------------------------------

physical["Anomaly_Evidence"] = (
    physical["Attack_Z"]
    .clip(lower=0)
)

# Robust normalization within each attack.
def normalize_group(s):

    s = s.astype(float)

    if s.max() <= 0:
        return pd.Series(
            np.zeros(len(s)),
            index=s.index
        )

    return (
        s / s.max()
    )


physical["Anomaly_Evidence"] = (
    physical
    .groupby("Attack_ID")["Anomaly_Evidence"]
    .transform(normalize_group)
)

physical["Onset_Evidence"] = (
    physical["Onset_Score"]
    .clip(lower=0)
)

physical["Onset_Evidence"] = (
    physical
    .groupby("Attack_ID")["Onset_Evidence"]
    .transform(normalize_group)
)

physical["Persistence_Evidence"] = (
    physical["Persistence"]
    .clip(lower=0, upper=1)
)

physical["Transition_Evidence"] = (
    physical["Transition_Score"]
    .clip(lower=0)
)

physical["Transition_Evidence"] = (
    physical
    .groupby("Attack_ID")["Transition_Evidence"]
    .transform(normalize_group)
)


# ------------------------------------------------------------
# NETWORK EVIDENCE
# ------------------------------------------------------------

network["Network_Evidence"] = (
    network["Network_Event_Share"]
    .fillna(0)
    .clip(lower=0)
)

network["Network_Evidence"] = (
    network
    .groupby("Attack_ID")["Network_Evidence"]
    .transform(normalize_group)
)


# ------------------------------------------------------------
# MERGE PHYSICAL + NETWORK
# ------------------------------------------------------------

# The network evidence file contains the following original fields:
# Network_Events, Total_Attack_Network_Events,
# Network_Event_Share, Unique_Sources, Unique_Destinations,
# Unique_Modbus_Transactions, Modbus_Events.

network_merge = network[
    [
        "Attack_ID",
        "Component",
        "Network_Events",
        "Total_Attack_Network_Events",
        "Network_Event_Share",
        "Unique_Sources",
        "Unique_Destinations",
        "Unique_Modbus_Transactions",
        "Modbus_Events",
        "Ground_Truth_Component"
    ]
].copy()

# Normalize component key.
network_merge["Component_norm"] = (
    network_merge["Component"]
    .astype(str)
    .str.replace("-", "", regex=False)
    .str.upper()
)

# Rename network evidence fields so they cannot collide
# with physical evidence fields.
network_merge = network_merge.rename(
    columns={
        "Network_Events": "Network_Events_Cyber",
        "Total_Attack_Network_Events":
            "Total_Attack_Network_Events_Cyber",
        "Network_Event_Share":
            "Network_Event_Share_Cyber",
        "Unique_Sources":
            "Unique_Sources_Cyber",
        "Unique_Destinations":
            "Unique_Destinations_Cyber",
        "Unique_Modbus_Transactions":
            "Unique_Modbus_Transactions_Cyber",
        "Modbus_Events":
            "Modbus_Events_Cyber"
    }
)

score = physical.merge(
    network_merge,
    on=["Attack_ID", "Component_norm"],
    how="left"
)

network_cols = [
    "Network_Events_Cyber",
    "Total_Attack_Network_Events_Cyber",
    "Network_Event_Share_Cyber",
    "Unique_Sources_Cyber",
    "Unique_Destinations_Cyber",
    "Unique_Modbus_Transactions_Cyber",
    "Modbus_Events_Cyber"
]

for c in network_cols:
    score[c] = pd.to_numeric(
        score[c],
        errors="coerce"
    ).fillna(0)

# Convenient normalized network evidence score.
score["Network_Evidence"] = (
    score["Network_Event_Share_Cyber"]
)

# ------------------------------------------------------------
# PROCESS DEPENDENCY EVIDENCE
# ------------------------------------------------------------

# Build adjacency list.
neighbors = {}

for _, row in graph[
    graph["Usable"] == True
].iterrows():

    src = normalize(row["Source"])
    dst = normalize(row["Target"])

    neighbors.setdefault(src, set()).add(dst)
    neighbors.setdefault(dst, set()).add(src)


# Create quick physical evidence lookup.
physical_lookup = {}

for _, row in score.iterrows():

    key = (
        int(row["Attack_ID"]),
        row["Component_norm"]
    )

    physical_lookup[key] = float(
        row["Anomaly_Evidence"]
    )


dependency_values = []

for _, row in score.iterrows():

    attack_id = int(
        row["Attack_ID"]
    )

    component = row[
        "Component_norm"
    ]

    neigh = neighbors.get(
        component,
        set()
    )

    if not neigh:

        dependency_values.append(0.0)
        continue

    values = []

    for n in neigh:

        key = (
            attack_id,
            n
        )

        if key in physical_lookup:

            values.append(
                physical_lookup[key]
            )

    if values:

        dependency_values.append(
            float(np.mean(values))
        )

    else:

        dependency_values.append(0.0)


score["Dependency_Evidence"] = (
    dependency_values
)


# ------------------------------------------------------------
# FINAL RCA SCORE
# ------------------------------------------------------------

score["RCA_Score"] = (

    W_ANOMALY *
    score["Anomaly_Evidence"]

    +

    W_ONSET *
    score["Onset_Evidence"]

    +

    W_PERSISTENCE *
    score["Persistence_Evidence"]

    +

    W_TRANSITION *
    score["Transition_Evidence"]

    +

    W_NETWORK *
    score["Network_Evidence"]

    +

    W_DEPENDENCY *
    score["Dependency_Evidence"]
)


# ------------------------------------------------------------
# RANK COMPONENTS
# ------------------------------------------------------------

score["Rank"] = (
    score
    .groupby("Attack_ID")["RCA_Score"]
    .rank(
        ascending=False,
        method="first"
    )
    .astype(int)
)

score = score.sort_values(
    [
        "Attack_ID",
        "Rank"
    ]
)


# ------------------------------------------------------------
# GROUND-TRUTH
# ------------------------------------------------------------

def get_truth_set(
    attack_id
):

    rows = windows[
        windows["Attack_ID"] == attack_id
    ]

    if rows.empty:
        return set()

    truth = rows.iloc[0][
        "Ground_Truth_Root_Cause"
    ]

    if pd.isna(truth):
        return set()

    result = set()

    for x in str(truth).split("|"):

        x = normalize(x)

        # Dataset alias
        if x == "DIT301":
            x = "DPIT301"

        if x in set(
            score["Component_norm"]
        ):
            result.add(x)

    return result


# ------------------------------------------------------------
# ATTACK-LEVEL RANKING
# ------------------------------------------------------------

ranking_rows = []

for attack_id, group in score.groupby(
    "Attack_ID"
):

    group = group.sort_values(
        "RCA_Score",
        ascending=False
    )

    ranked = (
        group["Component_norm"]
        .tolist()
    )

    truth = get_truth_set(
        int(attack_id)
    )

    top1 = (
        len(ranked) >= 1
        and ranked[0] in truth
    )

    top3 = any(
        x in truth
        for x in ranked[:3]
    )

    top5 = any(
        x in truth
        for x in ranked[:5]
    )

    # Reciprocal rank.
    rr = 0.0

    for i, component in enumerate(
        ranked,
        start=1
    ):

        if component in truth:

            rr = 1.0 / i
            break

    # Best rank among multiple true components.
    true_ranks = [
        i
        for i, component in enumerate(
            ranked,
            start=1
        )
        if component in truth
    ]

    best_rank = (
        min(true_ranks)
        if true_ranks
        else np.nan
    )

    ranking_rows.append({

        "Attack_ID":
            int(attack_id),

        "Ground_Truth":
            "|".join(sorted(truth)),

        "Top1_Prediction":
            ranked[0] if ranked else "",

        "Top2_Prediction":
            ranked[1] if len(ranked) > 1 else "",

        "Top3_Prediction":
            ranked[2] if len(ranked) > 2 else "",

        "Top5_Predictions":
            "|".join(ranked[:5]),

        "Top1_Correct":
            int(top1),

        "Top3_Correct":
            int(top3),

        "Top5_Correct":
            int(top5),

        "Reciprocal_Rank":
            rr,

        "Best_True_Rank":
            best_rank
    })


ranking = pd.DataFrame(
    ranking_rows
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

score.to_csv(
    OUT_SCORES,
    index=False
)

ranking.to_csv(
    OUT_RANKING,
    index=False
)


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

top1 = ranking[
    "Top1_Correct"
].mean()

top3 = ranking[
    "Top3_Correct"
].mean()

top5 = ranking[
    "Top5_Correct"
].mean()

mrr = ranking[
    "Reciprocal_Rank"
].mean()

print("\n" + "=" * 90)
print("FINAL CTF-RCA RESULTS")
print("=" * 90)

print(
    f"Attacks evaluated : {len(ranking)}"
)

print(
    f"Top-1 Accuracy    : {top1 * 100:.2f}%"
)

print(
    f"Top-3 Accuracy    : {top3 * 100:.2f}%"
)

print(
    f"Top-5 Accuracy    : {top5 * 100:.2f}%"
)

print(
    f"MRR               : {mrr:.4f}"
)

print("\n" + "-" * 90)
print("ATTACK-LEVEL PREDICTIONS")
print("-" * 90)

print(
    ranking.to_string(
        index=False
    )
)

print("\n" + "-" * 90)
print("TOP RCA SCORES")
print("-" * 90)

for attack_id, group in score.groupby(
    "Attack_ID"
):

    print(
        f"\nAttack {attack_id}:"
    )

    print(
        group[
            [
                "Component_norm",
                "RCA_Score",
                "Anomaly_Evidence",
                "Onset_Evidence",
                "Persistence_Evidence",
                "Transition_Evidence",
                "Network_Evidence",
                "Dependency_Evidence"
            ]
        ]
        .head(5)
        .to_string(index=False)
    )


print("\n" + "=" * 90)
print("OUTPUT FILES")
print("=" * 90)

print(OUT_SCORES)
print(OUT_RANKING)

print("=" * 90)
