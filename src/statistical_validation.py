from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path("data/processed")

LOAO = BASE / "rca_leave_one_attack_out_results.csv"
ABLATION = BASE / "rca_ablation_attack_details.csv"

N_BOOT = 10000
SEED = 42

rng = np.random.default_rng(SEED)

print("=" * 90)
print("CTF-RCA STATISTICAL VALIDATION")
print("=" * 90)

# ---------------------------------------------------------------------
# LOAD LOAO RESULTS
# ---------------------------------------------------------------------
loao = pd.read_csv(LOAO)

print("\nLOAO columns:")
print(list(loao.columns))

# ---------------------------------------------------------------------
# IDENTIFY RANK COLUMN
# ---------------------------------------------------------------------
rank_col = "Held_Out_Rank"

if rank_col not in loao.columns:
    candidates = [
        "Held_Out_Rank",
        "HeldOut_Rank",
        "Best_True_Rank",
        "True_Rank",
        "Rank"
    ]

    found = [c for c in candidates if c in loao.columns]

    if not found:
        raise ValueError(
            "Could not identify the held-out rank column. "
            f"Available columns: {list(loao.columns)}"
        )

    rank_col = found[0]

print("\nUsing rank column:", rank_col)

rank = pd.to_numeric(
    loao[rank_col],
    errors="coerce"
).to_numpy()

# Missing true component = NaN
valid = ~np.isnan(rank)

rank_valid = rank[valid]

top1 = (rank_valid <= 1).astype(float)
top3 = (rank_valid <= 3).astype(float)
top5 = (rank_valid <= 5).astype(float)
rr = 1.0 / rank_valid

# ---------------------------------------------------------------------
# BOOTSTRAP FUNCTION
# ---------------------------------------------------------------------
def bootstrap_ci(values, n_boot=N_BOOT, seed=42):
    values = np.asarray(values, dtype=float)

    rng_local = np.random.default_rng(seed)
    n = len(values)

    boot = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng_local.choice(
            values,
            size=n,
            replace=True
        )
        boot[i] = np.mean(sample)

    return (
        float(np.mean(values)),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5))
    )

# ---------------------------------------------------------------------
# BOOTSTRAP CIs
# ---------------------------------------------------------------------
results = []

for name, values in [
    ("Top-1 Accuracy", top1),
    ("Top-3 Accuracy", top3),
    ("Top-5 Accuracy", top5),
    ("MRR", rr)
]:
    mean, low, high = bootstrap_ci(values)

    results.append({
        "Metric": name,
        "Estimate": mean,
        "CI_95_Lower": low,
        "CI_95_Upper": high,
        "N": len(values)
    })

ci_df = pd.DataFrame(results)

print("\n" + "=" * 90)
print("LOAO 95% BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 90)

for _, r in ci_df.iterrows():
    if "Accuracy" in r["Metric"]:
        print(
            f"{r['Metric']:20s}: "
            f"{r['Estimate']*100:.2f}% "
            f"[{r['CI_95_Lower']*100:.2f}%, "
            f"{r['CI_95_Upper']*100:.2f}%]"
        )
    else:
        print(
            f"{r['Metric']:20s}: "
            f"{r['Estimate']:.4f} "
            f"[{r['CI_95_Lower']:.4f}, "
            f"{r['CI_95_Upper']:.4f}]"
        )

# ---------------------------------------------------------------------
# RANK DISTRIBUTION
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("HELD-OUT RANK DISTRIBUTION")
print("=" * 90)

print(f"Valid attacks : {len(rank_valid)}")
print(f"Missing       : {np.isnan(rank).sum()}")

print(f"Rank = 1      : {(rank_valid == 1).sum()}")
print(f"Rank <= 3     : {(rank_valid <= 3).sum()}")
print(f"Rank <= 5     : {(rank_valid <= 5).sum()}")
print(f"Rank > 5      : {(rank_valid > 5).sum()}")

print(f"Mean rank     : {np.mean(rank_valid):.4f}")
print(f"Median rank   : {np.median(rank_valid):.4f}")

# ---------------------------------------------------------------------
# PAIRED BOOTSTRAP DIFFERENCES
#
# Compare LOAO final model against anomaly-only baseline IF the
# ablation attack-details file contains the corresponding per-attack
# results.
# ---------------------------------------------------------------------
print("\n" + "=" * 90)
print("ABLATION / PAIRED COMPARISON")
print("=" * 90)

ab = pd.read_csv(ABLATION)

print("Ablation columns:")
print(list(ab.columns))

# Try to identify experiment/rank information
experiment_col = None

for c in ["Experiment", "experiment", "Configuration"]:
    if c in ab.columns:
        experiment_col = c
        break

rank_candidates = [
    "Rank",
    "Best_True_Rank",
    "True_Rank",
    "RCA_Rank"
]

ab_rank_col = next(
    (c for c in rank_candidates if c in ab.columns),
    None
)

attack_col = next(
    (c for c in ["Attack_ID", "attack_id", "Attack"] if c in ab.columns),
    None
)

if experiment_col is None or ab_rank_col is None or attack_col is None:

    print("\nPer-attack ablation structure is not suitable for a paired test.")
    print("Therefore, no paired significance test is performed at this stage.")

else:

    print("Experiment column:", experiment_col)
    print("Rank column      :", ab_rank_col)
    print("Attack column    :", attack_col)

    # Find anomaly-only experiment
    experiments = ab[experiment_col].astype(str).unique()

    anomaly_exp = [
        e for e in experiments
        if "Anomaly_Only" in e
    ]

    if not anomaly_exp:
        print("\nAnomaly-only experiment not found.")
    else:

        anomaly_name = anomaly_exp[0]

        anomaly = ab[
            ab[experiment_col].astype(str) == anomaly_name
        ].copy()

        anomaly[ab_rank_col] = pd.to_numeric(
            anomaly[ab_rank_col],
            errors="coerce"
        )

        # Merge with LOAO results
        loao_attack_col = next(
            (
                c for c in [
                    "HeldOut_Attack",
                    "Attack_ID",
                    "attack_id",
                    "Attack"
                ]
                if c in loao.columns
            ),
            None
        )

        if loao_attack_col is None:
            print("\nCould not identify Attack_ID in LOAO results.")

        else:

            loao_cmp = pd.DataFrame({
                "Attack_ID": loao[loao_attack_col],
                "LOAO_Rank": rank
            })

            anomaly_cmp = anomaly[
                [attack_col, ab_rank_col]
            ].rename(
                columns={
                    attack_col: "Attack_ID",
                    ab_rank_col: "Baseline_Rank"
                }
            )

            merged = pd.merge(
                loao_cmp,
                anomaly_cmp,
                on="Attack_ID",
                how="inner"
            )

            merged = merged.dropna()

            if len(merged) > 0:

                merged["LOAO_Top1"] = (
                    merged["LOAO_Rank"] <= 1
                ).astype(float)

                merged["Baseline_Top1"] = (
                    merged["Baseline_Rank"] <= 1
                ).astype(float)

                merged["LOAO_MRR"] = (
                    1.0 / merged["LOAO_Rank"]
                )

                merged["Baseline_MRR"] = (
                    1.0 / merged["Baseline_Rank"]
                )

                diff_top1 = (
                    merged["LOAO_Top1"]
                    - merged["Baseline_Top1"]
                ).to_numpy()

                diff_mrr = (
                    merged["LOAO_MRR"]
                    - merged["Baseline_MRR"]
                ).to_numpy()

                def paired_bootstrap(diff, n_boot=10000):
                    n = len(diff)
                    boot = np.empty(n_boot)

                    for i in range(n_boot):
                        sample = rng.choice(
                            diff,
                            size=n,
                            replace=True
                        )
                        boot[i] = np.mean(sample)

                    return (
                        np.mean(diff),
                        np.percentile(boot, 2.5),
                        np.percentile(boot, 97.5)
                    )

                t_mean, t_low, t_high = paired_bootstrap(
                    diff_top1
                )

                m_mean, m_low, m_high = paired_bootstrap(
                    diff_mrr
                )

                print(
                    f"\nPaired attacks available: {len(merged)}"
                )

                print(
                    f"\nTop-1 difference "
                    f"(LOAO - anomaly-only): "
                    f"{t_mean*100:.2f} percentage points"
                )

                print(
                    f"95% CI: "
                    f"[{t_low*100:.2f}, "
                    f"{t_high*100:.2f}] pp"
                )

                print(
                    f"\nMRR difference "
                    f"(LOAO - anomaly-only): "
                    f"{m_mean:.4f}"
                )

                print(
                    f"95% CI: "
                    f"[{m_low:.4f}, {m_high:.4f}]"
                )

                merged.to_csv(
                    BASE / "rca_paired_statistical_comparison.csv",
                    index=False
                )

# ---------------------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------------------
ci_df.to_csv(
    BASE / "rca_bootstrap_confidence_intervals.csv",
    index=False
)

print("\n" + "=" * 90)
print("STATISTICAL VALIDATION COMPLETE")
print("=" * 90)

print("\nSaved:")
print(
    "data/processed/rca_bootstrap_confidence_intervals.csv"
)

if (BASE / "rca_paired_statistical_comparison.csv").exists():
    print(
        "data/processed/rca_paired_statistical_comparison.csv"
    )

