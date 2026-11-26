#!/bin/bash
# PRESPEC_AST2 driver (BAF only): 9 jobs = 3 noises x 3 learners, 14 CPUs each (126 of 128), XGB on GPU. Resubmits missing outputs with more memory (max 3 attempts), then writes VERDICT_AST2.txt
cd ~/imb_bench; source venv/bin/activate
for att in 1 2 3; do
  miss=0
  for N in delay2 delay random; do for K in LR RF XGB; do
    d=results_cq8_baf_${N}_$K
    if [ ! -s $d/baf.csv ] && [ -z "$(ls $d/*.csv 2>/dev/null | grep -v partial)" ]; then
      miss=$((miss+1)); echo "$(date) submit $d attempt $att" >> auto_driver8.log
      G=""; X=""; [ $K = XGB ] && G="--gres=gpu:1" && X="XGB_GPU=1"
      sbatch -p l40s $G --mem=$((16*att))G --cpus-per-task=14 -J a8b$N$K -o logs/a8_%j.out --wrap "cd ~/imb_bench; CAP=60000 NOISE=$N STRATS=BASE,CW KINDS=$K NJOBS=14 $X venv/bin/python code_sec/cq8_run.py baf:baf $d" > /dev/null
    fi
  done; done
  [ $miss -eq 0 ] && break
  while [ "$(squeue -u bambang -h -o %j | grep -c '^a8b')" -gt 0 ]; do sleep 60; done
done
python code_sec/analyze_conf8.py results_cq8_baf_* > ast2_baf.txt 2>&1
echo "finished $(date) missing=$miss" > VERDICT_AST2.txt
