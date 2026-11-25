#!/bin/bash
cd /tmp/an/imb/code
for d in $(cat ../pilot_list.txt); do KINDS=LR,HGB,XGB python3 run_bench.py $d /tmp/an/imb/pilot_res 1 5; done
