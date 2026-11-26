# Imbalanced learning benchmark and poison-robust fixed-FPR thresholds

Code for experiments on class-imbalanced classification: 49 benchmark datasets (42 KEEL, 7 KRNN), resampling strategies (ROS, RUS, SMOTE, Borderline-SMOTE, ADASYN, SMOTE-ENN, SMOTE-Tomek, class weights), threshold tuning, and a temporal security protocol (rolling-origin windows, recall at fixed false positive rate) with label-flip poisoning and capped/gated validation-based threshold rules.

## Layout
- `code/` benchmark over the 49 datasets (`run_bench.py`, `analyze.py`), Slurm scripts, strategy selector (`meta_selector.py`).
- `code_sec/` security protocol, poisoning experiments, threshold rules, and the 49-dataset poisoning test (`bench49_poison.py`, `analyze_b49.py`).
- `data_info/DATASET_REGISTRY.csv` size, imbalance ratio, source and citation per dataset; `data_info/DATASET_SUMBER_DAN_LINK.md` data sources and licence notes (in Indonesian).
- `spec/` pre-specification files (`PRESPEC_*.md`) frozen before the runs they govern.
- `results/` summary result files; `results/hasil_final/` final tables used in the manuscript (text and LaTeX).

## Data
Datasets are not redistributed. Obtain them from their original sources: KEEL repository, the `imbalanced-databases` package, OpenML ids 42397 (credit card), 46858 (IEEE-CIS) and 45277 (CIC-IDS2017), UNSW-NB15, Elliptic, and Bank Account Fraud (BAF).

## Environment
Python 3, numpy, pandas, scipy, scikit-learn, xgboost, imbalanced-learn, statsmodels (see `code/hpc_setup.sh`). Experiments ran on a Slurm cluster; scripts are run from the project root.

## Status
Accompanies a manuscript under review. A versioned archive with a DOI will be deposited after acceptance.
