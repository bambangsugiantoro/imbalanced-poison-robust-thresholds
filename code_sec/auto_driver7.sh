#!/bin/bash
# Autonomous driver for PRESPEC_AUDIT runs: waits for a4 jobs, resubmits missing outputs with more memory (max 3 attempts), then runs analyses and writes VERDICT_AUDIT.txt
cd ~/imb_bench; source venv/bin/activate
declare -A SRC=([cc]="oml:42397:Time:creditcard" [ie]="oml:46858:TransactionDT:ieee" [un]="unsw:unsw" [el]="ell:ell" [ci]="cic:cic")
declare -A CAPV=([cc]=60000 [ie]=30000 [un]=60000 [el]=60000 [ci]=60000)
declare -A CPU=([cc]=8 [ie]=8 [un]=8 [el]=4 [ci]=16)
declare -A MEM=([cc]=12 [ie]=24 [un]=12 [el]=8 [ci]=24)
declare -A FN=([cc]=creditcard [ie]=ieee [un]=unsw [el]=elliptic [ci]=cic2017)
for att in 1 2 3; do
  while [ "$(squeue -u bambang -h -o %j | grep -c '^a7')" -gt 0 ]; do sleep 120; done
  miss=0
  for ds in cc ie un el ci; do for N in delay2 delay random; do for K in LR RF XGB; do
    d=results_cq7_${ds}_${N}_$K
    if [ ! -s $d/${FN[$ds]}.csv ]; then miss=$((miss+1)); echo "$(date) resubmit $d attempt $att" >> auto_driver6.log
      sbatch -p l40s --mem=$((MEM[$ds]*(att+1)))G --cpus-per-task=${CPU[$ds]} -J a7$ds$N$K -o logs/a7_%j.out --wrap "cd ~/imb_bench; CAP=${CAPV[$ds]} NOISE=$N STRATS=BASE,CW KINDS=$K NJOBS=${CPU[$ds]} venv/bin/python code_sec/cq7_run.py ${SRC[$ds]} $d" >> auto_driver6.log
    fi
  done; done; done
  [ $miss -eq 0 ] && break
done
python code_sec/analyze_conf7.py results_cq7_cc_* results_cq7_ie_* results_cq7_un_* > ast_dev.txt 2>&1
python code_sec/analyze_conf7.py results_cq7_el_* results_cq7_ci_* > ast_conf.txt 2>&1
echo "finished $(date) missing=$miss" > VERDICT_AST.txt
