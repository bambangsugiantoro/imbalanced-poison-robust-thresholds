"""usage: python analyze_poison.py RESULTS_DIR OUT_PREFIX -> mean over windows by learner/strategy/poison/mode"""
import sys, glob, os, pandas as pd
rd, out = sys.argv[1], sys.argv[2]
df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(rd, '*.csv')) if 'partial' not in f], ignore_index=True)
M = ['Recall', 'FPR', 'alerts_per_10k_benign', 'F1', 'AP', 'Brier', 'ECE']
g = df.groupby(['dataset', 'learner', 'strategy', 'thr_mode', 'poison'])[M].mean().reset_index(); g.to_csv(out + '_means.csv', index=False)
pd.set_option('display.width', 250)
for m in ('fpr1', 'fpr01'):
    print('\n=== mode', m, ': Recall (rows learner/strategy; columns poison level), then AP ===')
    for k in ('Recall', 'AP', 'FPR'):
        p = g[g.thr_mode == m].pivot_table(index=['dataset', 'learner', 'strategy'], columns='poison', values=k)
        print('\n--', k, m); print(p.round(3).to_string())
