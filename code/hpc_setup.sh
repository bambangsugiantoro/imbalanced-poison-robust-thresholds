#!/bin/bash
# Jalankan SEKALI di login node HPC (folder ~/imb_bench). Membuat venv dan memasang paket.
set -e
cd ~/imb_bench
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn xgboost imbalanced-learn statsmodels
pip install --no-deps imbalanced-databases      # --no-deps: paket ini meminta 'sklearn' (nama lama) yang gagal dibangun
python - <<'PY'
import imbalanced_databases as imbd, numpy as np
d=imbd.load_ecoli_0_vs_1(); print('OK', np.asarray(d['data']).shape)
PY
