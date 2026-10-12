from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

BASE = Path("data/processed")
OUT = BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FILE = BASE / "ctf_rca_component_scores.csv"

print("=" * 90)
print("CTF-RCA ROC CURVE ANALYSIS")
print("=" * 90)

df = pd.read_csv(FILE)

score_col = "RCA_Score"
label_col = "Ground_Truth_Component_x"

required = [score_col, label_col]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# ------------------------------------------------------------
# Prepare ROC data
# ------------------------------------------------------------
roc_df = df[[score_col, label_col]].copy()

roc_df[score_col] = pd.to_numeric(
    roc_df[score_col],
    errors="coerce"
)

roc_df[label_col] = pd.to_numeric(
    roc_df[label_col],
    errors="coerce"
)

roc_df = roc_df.dropna()

roc_df[label_col] = roc_df[label_col].astype(int)

y_score = roc_df[score_col].to_numpy()
y_true = roc_df[label_col].to_numpy()

print(f"Total candidate rows : {len(roc_df)}")
print(f"Positive candidates  : {int(y_true.sum())}")
print(f"Negative candidates  : {int((y_true == 0).sum())}")

if len(np.unique(y_true)) != 2:
    raise ValueError(
        "ROC requires both positive and negative candidate labels."
    )

# ------------------------------------------------------------
# ROC and AUC
# ------------------------------------------------------------
fpr, tpr, thresholds = roc_curve(
    y_true,
    y_score
)

auc_value = roc_auc_score(
    y_true,
    y_score
)

print()
print("=" * 90)
print("ROC RESULTS")
print("=" * 90)

print(f"ROC-AUC : {auc_value:.4f}")

# ------------------------------------------------------------
# Youden operating point
# ------------------------------------------------------------
youden = tpr - fpr
best_idx = np.argmax(youden)

print(f"Best threshold : {thresholds[best_idx]:.6f}")
print(f"TPR            : {tpr[best_idx]:.4f}")
print(f"FPR            : {fpr[best_idx]:.4f}")

# ------------------------------------------------------------
# Publication figure
# ------------------------------------------------------------
plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2.5,
    label=f"CTF-RCA (AUC = {auc_value:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Random classifier"
)

plt.xlabel(
    "False Positive Rate",
    fontsize=12
)

plt.ylabel(
    "True Positive Rate",
    fontsize=12
)

plt.title(
    "ROC Curve for Component-Level Root-Cause Identification",
    fontsize=13
)

plt.xlim(0, 1)
plt.ylim(0, 1.02)

plt.grid(
    True,
    alpha=0.25
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

output = OUT / "Figure_8_ROC_Curve.png"

plt.savefig(
    output,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# ------------------------------------------------------------
# Save ROC coordinates
# ------------------------------------------------------------
roc_coordinates = pd.DataFrame({
    "False_Positive_Rate": fpr,
    "True_Positive_Rate": tpr,
    "Threshold": thresholds
})

roc_coordinates.to_csv(
    BASE / "roc_curve_coordinates.csv",
    index=False
)

# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------
summary = pd.DataFrame([{
    "ROC_AUC": auc_value,
    "Best_Threshold": thresholds[best_idx],
    "Best_TPR": tpr[best_idx],
    "Best_FPR": fpr[best_idx],
    "Positive_Candidates": int(y_true.sum()),
    "Negative_Candidates": int((y_true == 0).sum()),
    "Total_Candidates": len(y_true)
}])

summary.to_csv(
    BASE / "roc_auc_summary.csv",
    index=False
)

print()
print("=" * 90)
print("FILES CREATED")
print("=" * 90)

print(output)
print(BASE / "roc_curve_coordinates.csv")
print(BASE / "roc_auc_summary.csv")

print()
print("=" * 90)
print("ROC ANALYSIS COMPLETE")
print("=" * 90)
