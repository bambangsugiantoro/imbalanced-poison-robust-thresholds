#!/bin/bash
#SBATCH --job-name=cq2
#SBATCH --partition=l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cq2_%j.out
#SBATCH --error=logs/cq2_%j.err
# usage: sbatch cq2_job.sbatch SOURCE OUTDIR [reps] [nsplits]   (env NOISE, STRATS, LEVELS exported via --export)
cd ~/imb_bench; source venv/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python code_sec/cq2_run.py "$1" "$2" ${3:-2} ${4:-5}
