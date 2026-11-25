#!/bin/bash
#SBATCH --job-name=sec_bench
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sec_%j.out
#SBATCH --error=logs/sec_%j.err
# usage: sbatch sec_job.sbatch DID TIMECOL NAME [CAP]
# TIDAK memakai --exclusive; node dibagi dengan pengguna lain.
cd ~/imb_bench; source venv/bin/activate
export NJOBS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CAP=${4:-60000}
python code_sec/sec_bench.py "$1" "$2" results_sec "$3"
