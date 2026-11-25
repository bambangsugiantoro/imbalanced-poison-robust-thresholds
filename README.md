# Imbalanced learning benchmark and poison-robust fixed-FPR thresholds

Code for experiments on class-imbalanced classification: 49 benchmark datasets (42 KEEL, 7 KRNN), resampling strategies (ROS, RUS, SMOTE, Borderline-SMOTE, ADASYN, SMOTE-ENN, SMOTE-Tomek, class weights), threshold tuning, and a temporal security protocol (rolling-origin windows, recall at fixed false positive rate) with label-flip poisoning and capped/gated validation-based threshold rules.

## Layout
- `code/` benchmark over the 49 datasets (`run_bench.py`, `analyze.py`), Slurm scripts, strategy selector (`meta_selector.py`).
- `code_sec/` security protocol, poisoning experiments, threshold rules, and the 49-dataset poisoning test (`bench49_poison.py`, `analyze_b49.py`).
- `data_info/DATASET_REGISTRY.csv` size, imbalance ratio, source and citation per dataset.
- `results/` summary result files (full result files will be added with the final version).

## Data
Datasets are not redistributed. Obtain them from their original sources: KEEL repository, the `imbalanced-databases` package, OpenML ids 42397 (creditcard) and 46858 (IEEE-CIS), and UNSW-NB15.

## Environment
Python 3, numpy, pandas, scipy, scikit-learn, xgboost, imbalanced-learn, statsmodels (see `code/hpc_setup.sh`). Experiments ran on a Slurm cluster; scripts are run from the project root.

## Status
Work in progress; results and the license will be finalized with the accompanying manuscript.
