#!/bin/bash
#SBATCH --job-name=imb_bench
#SBATCH --array=1-49%12
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=06:00:00
#SBATCH --output=logs/imb_%A_%a.out
#SBATCH --error=logs/imb_%A_%a.err
# TIDAK memakai --exclusive; node dibagi dengan pengguna lain. Sesuaikan --partition dengan hasil `sinfo`.
cd ~/imb_bench
source venv/bin/activate
export NJOBS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" datasets.txt)
python code/run_bench.py "$NAME" results 2 5
