# AML Mule Account Detection — Code Submission

## Github Link to Externally Hosted Model Files:

https://github.com/Adithya-Monish-Kumar-K/NFPC-Phase-2-Submission

## Overview

End-to-end anti-money laundering (AML) mule account detection pipeline that processes over 400 million transactions across 160,153 bank accounts to identify potential money mule accounts using a multi-model stacking ensemble.

## 1. Environment Setup

### Hardware Requirements

- **OS:** Windows / Linux / macOS
- **RAM:** Minimum 16GB (the pipeline uses batch-streaming to process the massive dataset within memory constraints).
- **GPU:** (Optional) NVIDIA GPU with CUDA support accelerates XGBoost training significantly.

### Software Dependencies

- **Python Version:** 3.10 or higher
- **Required Libraries:**

```bash
pip install numpy pandas pyarrow scikit-learn lightgbm xgboost catboost optuna networkx python-louvain shap cleanlab torch ruptures matplotlib
```

## 2. Steps to Reproduce Results

Before running the script, ensure the competition dataset files and folders (`transactions/`, `transactions_additional/`, `accounts.parquet`, `train_labels_with_flags.parquet`, `test_accounts.parquet`, etc.) are placed in the **same directory** as `solution.py`.

### Option A: Instant Inference (Scoring Mode)

Run this mode to evaluate the provided model weights against the test set.

*Note: Before running this, make sure that you have downloaded the models from github, or that you have run the full training pipeline.*

```bash
python solution.py --mode predict
```

- **How it Works**: The script automatically detects that pre-extracted features are missing from the environment. It runs the full 30-45 minute feature engineering extraction pipeline on the raw data from scratch, caches the results to RAM/disk, loads the pre-trained model weights from the included `models/` directory, and instantly generates the final `submission.csv`.

### Option B: Full Training Pipeline (Reproduction Mode)

Run this mode to completely recreate all models from the ground up.

```bash
python solution.py --mode train
```

- **How it Works**: Executes feature engineering, performs label denoising via Confident Learning, executes stratified cross-validation across 7 base models, optimizes hyperparameters via Optuna, trains the Tri-Track stack, applies Calibration, and saves the trained weights into the `models/` directory.

## 3. Description of Approach

Our approach attacks the problem across three distinct layers:

### A. Feature Engineering (289 Features)

We hand-crafted features specifically targeting 13 known money-mule typologies.
The vast majority (279) are strict **Label-Free / Structural** features computed before training:

- **Transaction Dynamics:** Pass-through velocity, temporal activity peaks, burst-dormancy ratios, Benford's law deviations, threshold structuring.
- **Graph & Network:** Degree centrality, PageRank, Louvain community detection, DeepWalk Node2Vec embeddings.
- **Temporal Graph:** 6-month historical graph evolution and rolling activity windows.
- **Infrastructure:** Geo-spatial IP diversity, shared-IP exposure, and account balance trajectories.

Additionally, to capture powerful guilt-by-association signals without leaking target data across validation folds, we compute **10 Dynamic Label-Augmented Features** STRICTLY computed at runtime using only the training-fold mule IDs:

1. `shared_cp_with_mules`: Count of direct counterparties that are known mules.
2. `suspicious_cp_count`: Count of counterparties shared by 5+ known mules.
3. `two_hop_mule_exposure`: 2-hop graph connectivity to known mules.
4. `shared_ips_with_mules`: Count of unique IP hashes shared with known mules.
5. `shared_ip_ratio`: Ratio of shared vs private IP hashes.
6. `branch_mule_rate`: Density of mules originating from the same physical bank branch.
7. `community_mule_count`: Absolute count of mules residing in the same Louvain cluster.
8. `community_mule_rate`: Percentage of the account's Louvain community that are mules.
9. `branch_collusion_score`: Composite interaction of branch density × shared suspicious counterparties.
10. `branch_susp_cp_score`: Composite interaction of branch density × direct mule counterparties.

### B. Label Denoising

Given the massive class imbalance and inherent noise in suspicious activity reporting, we utilized a three-pronged approach:

1. **Temporal Weighting**: Downweighting labels for accounts active outside the observable 2020-2025 data window.
2. **Alert-Reason Scrutiny**: Deferring to the model rather than the label for vague "Routine Investigation" flags.
3. **Confident Learning**: Using the `cleanlab` framework to identify and down-weight highly-probable mislabeled accounts.

### C. Tri-Track Meta-Stacking Ensemble Architecture

To maximize predictive power and generalization across both seen and unseen network clusters, we built a complex **35-model ensemble** leveraging 7 distinct base algorithms and a multi-track meta-learning system.

**Base Models (5-Fold Stratified CV):**

1. **LightGBM** (Label-Augmented)
2. **XGBoost** (Label-Augmented)
3. **CatBoost** (Label-Augmented)
4. **ExtraTrees** (Label-Augmented)
5. **HistGradientBoosting** (Label-Augmented)
6. **LF-ET (Label-Free ExtraTrees)**: Trained strictly on structural features without knowing target labels to prevent data leakage.
7. **Net-ET (Network-Signal ExtraTrees)**: Trained explicitly on subgraph connectivity features.

**Track Architectures:**

- **Track A (Core Stack)**: Stacks the predictions of the 5 label-augmented core models (LGB, XGB, CB, ET, HGB) using a LightGBM meta-learner. Captures primary label-driven signal diversity.
- **Track B (Full Stack)**: Stacks all 7 models alongside complex meta-features calculated dynamically (prediction spread, label-free delta, network vs LF delta, network confidence).
- **Track C (LF-Weighted Stack)**: Stacks all 7 models specifically weighted to emphasize label-free structural generalization on "unseen" network segments.

**Final Blending & Calibration:**
We utilize another **Meta-Meta-Learner** to blend the output of Track A, Track B, and Track C, alongside their disagreement vectors, into a final probability score. This score is then passed through an optimizer that selects the best calibration method (Isotonic Regression vs. Platt Scaling) based on Brier score minimization.

## Final Output

The pipeline produces a strictly formatted 4-column `submission.csv` (`account_id`, `is_mule`, `suspicious_start`, `suspicious_end`).
