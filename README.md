# CTF-RCA
## Counterfactual Trust Fusion for Root-Cause Attribution

This directory contains the implementation and experimental scripts for:

**Counterfactual Trust Fusion for Root-Cause Attribution Under Conflicting Cyber-Physical Evidence**

CTF-RCA is a multi-source evidence-fusion framework for root-cause attribution in cyber-physical systems under heterogeneous, uncertain, and potentially conflicting evidence.

## Dataset

The experiments use the Secure Water Treatment (SWaT) dataset provided by:

**iTrust, Centre for Research in Cyber Security, Singapore University of Technology and Design (SUTD)**

Official dataset page:

https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/swat/

The SWaT dataset is not redistributed in this repository. Users must obtain the dataset from the official source and follow its access and usage conditions.

## Experimental Dataset Characteristics

The reported experiments use:

- 51 candidate sensors/actuators
- More than 900,000 observations
- 36 attack episodes
- 1,836 attack-component candidate evaluations
- 40 process-dependency edges
- Six evidence dimensions

The six evidence dimensions are:

1. Anomaly
2. Onset
3. Persistence
4. Transition
5. Network
6. Dependency

## Repository Structure

```text
CTF-RCA-SWaT/
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── data_quality_audit.py
├── inspect_raw.py
└── src/
    ├── preprocess.py
    ├── preprocess_network.py
    ├── prepare_attack_metadata.py
    ├── build_attack_ground_truth.py
    ├── build_attack_windows.py
    ├── create_temporal_windows.py
    ├── create_component_mapping.py
    ├── build_rca_evidence_v2.py
    ├── build_component_network_evidence.py
    ├── build_process_dependency_graph.py
    ├── synchronize_ctf_rca.py
    ├── run_final_ctf_rca.py
    ├── optimize_rca_weights.py
    ├── run_ablation_study.py
    ├── leave_one_attack_out_validation.py
    ├── statistical_validation.py
    ├── generate_rca_tables.py
    ├── generate_rca_figures.py
    ├── generate_roc_curve.py
    └── component_level_analysis.py
```

## Data Preparation

Raw SWaT data are expected under:

```text
data/raw/
```

Processed data are generated under:

```text
data/processed/
```

### 1. SWaT preprocessing

```bash
python src/preprocess.py
```

This script processes the raw SWaT data and writes the processed data to `data/processed/`.

### 2. Network preprocessing

```bash
python src/preprocess_network.py
```

Network input data are read from `data/raw/Network/` and processed network features are written to `data/processed/network_features/`.

### 3. Attack metadata preparation

```bash
python src/prepare_attack_metadata.py
```

Input:

```text
data/raw/List_of_attacks_Final.xlsx
```

Output:

```text
data/processed/attack_metadata_processed.csv
```

### 4. Attack ground-truth construction

```bash
python src/build_attack_ground_truth.py
```

Output:

```text
data/processed/attack_ground_truth.csv
```

### 5. Attack-window construction

```bash
python src/build_attack_windows.py
```

Output:

```text
data/processed/rca_attack_windows.csv
```

### 6. Temporal-window construction

```bash
python src/create_temporal_windows.py
```

Input:

```text
data/processed/normal_processed.csv
```

Output:

```text
data/processed/normal_windows_60s.npz
```

### 7. Component mapping

```bash
python src/create_component_mapping.py
```

This script generates the component mapping used by the RCA evidence construction and fusion stages.

## RCA Evidence Construction

### Physical and temporal evidence

```bash
python src/build_rca_evidence_v2.py
```

Output:

```text
data/processed/rca_evidence_matrix_v2.csv
```

This file contains the physical and temporal evidence used by the CTF-RCA framework.

### Network evidence

```bash
python src/build_component_network_evidence.py
```

Output:

```text
data/processed/rca_component_network_evidence.csv
```

This file contains component-level network evidence for the candidate root causes.

### Process dependency graph

```bash
python src/build_process_dependency_graph.py
```

Output:

```text
data/processed/rca_process_dependency_graph.csv
```

The dependency graph provides process-level relationships used as dependency evidence during fusion.

## Final CTF-RCA Fusion

The final fusion stage combines the six evidence dimensions to produce a ranked list of candidate root causes for each attack episode.

Run:

```bash
python src/run_final_ctf_rca.py
```

The final fusion stage uses:

```text
data/processed/rca_evidence_matrix_v2.csv
data/processed/rca_component_network_evidence.csv
data/processed/rca_process_dependency_graph.csv
data/processed/rca_attack_windows.csv
data/processed/rca_component_mapping.csv
```

The generated outputs are:

```text
data/processed/ctf_rca_component_scores.csv
data/processed/ctf_rca_attack_rankings.csv
```

### Initial Fusion Weights

The initial fusion configuration is:

| Evidence Dimension | Weight |
|---|---:|
| Anomaly | 0.25 |
| Onset | 0.20 |
| Persistence | 0.15 |
| Transition | 0.15 |
| Network | 0.15 |
| Dependency | 0.10 |

## Weight Optimization

The fusion weights are optimized by evaluating 3,003 candidate weight configurations.

Run:

```bash
python src/optimize_rca_weights.py
```

The optimized configuration reported in the study is:

| Evidence Dimension | Optimized Weight |
|---|---:|
| Anomaly | 0.00 |
| Onset | 0.10 |
| Persistence | 0.00 |
| Transition | 0.30 |
| Network | 0.40 |
| Dependency | 0.20 |

Output:

```text
data/processed/rca_weight_optimization.csv
```

Optimized performance:

| Metric | Result |
|---|---:|
| Top-1 | 52.78% |
| Top-3 | 63.89% |
| Top-5 | 72.22% |
| MRR | 0.6088 |

## Ablation Study

The ablation study evaluates the contribution of different combinations of the six evidence dimensions.

Run:

```bash
python src/run_ablation_study.py
```

The study reports Top-1, Top-3, Top-5, and Mean Reciprocal Rank (MRR) for each evidence configuration.

## Leave-One-Attack-Out Validation

Leave-One-Attack-Out (LOAO) validation evaluates the generalization of the fusion-weight selection by holding out each attack episode in turn.

Run:

```bash
python src/leave_one_attack_out_validation.py
```

The corrected LOAO evaluation reports:

| Metric | Result |
|---|---:|
| Top-1 | 42.86% |
| Top-3 | 57.14% |
| Top-5 | 71.43% |
| MRR | 0.5422 |

Output:

```text
data/processed/rca_leave_one_attack_out_results.csv
```

## Statistical Validation

Additional statistical validation is implemented in:

```bash
python src/statistical_validation.py
```

## ROC Analysis

Candidate-level discrimination is evaluated using ROC-AUC.

Run:

```bash
python src/generate_roc_curve.py
```

The reported ROC-AUC is **0.8128**.

ROC-AUC is used as a complementary candidate-level measure alongside the attack-level ranking metrics.

## Component-Level Analysis

Component-level evaluation assesses the rank of each ground-truth component within the complete candidate-component ranking.

Run:

```bash
python src/component_level_analysis.py
```

The corrected component-level analysis contains:

- 52 ground-truth component occurrences
- 25 distinct ground-truth components
- 51 candidate components in the complete candidate universe

Component-level evaluation differs from attack-level evaluation. Each ground-truth component is evaluated using its own rank in the complete candidate ranking.

## Reported Results

### Initial Fusion

| Metric | Result |
|---|---:|
| Top-1 | 44.44% |
| Top-3 | 55.56% |
| Top-5 | 69.44% |
| MRR | 0.5530 |

### Optimized Fusion

| Metric | Result |
|---|---:|
| Top-1 | 52.78% |
| Top-3 | 63.89% |
| Top-5 | 72.22% |
| MRR | 0.6088 |

### Corrected LOAO

| Metric | Result |
|---|---:|
| Top-1 | 42.86% |
| Top-3 | 57.14% |
| Top-5 | 71.43% |
| MRR | 0.5422 |

## Evaluation Definition

Attack-level metrics use the best-ranked ground-truth component for each attack.

For attacks with multiple ground-truth components:

best_rank = min(rank of true components)

Attack-level MRR is defined as:

MRR = 1 / best_rank

## Reproducibility

The repository provides scripts covering data preprocessing, attack metadata preparation, ground-truth construction, temporal and network evidence construction, process-dependency modeling, CTF-RCA fusion, weight optimization, ablation analysis, LOAO validation, statistical validation, ROC analysis, and component-level evaluation.

The raw SWaT dataset is not redistributed in this repository and must be obtained separately from iTrust/SUTD.

For the complete methodological definitions and experimental protocol, refer to the associated manuscript.

## Citation

If you use this implementation or methodology, please cite:

Pawan Kumar Badhan, "Counterfactual Trust Fusion for Root-Cause Attribution Under Conflicting Cyber-Physical Evidence," Information Fusion.

## Repository

https://github.com/pawan0216/VR-ML
