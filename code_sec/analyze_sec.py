"""Summarise security benchmark: mean over temporal windows; delta vs BASE (same learner, same decision mode).
usage: python analyze_sec.py RESULTS_DIR OUT_PREFIX
Note: only 4 windows per dataset -> no Wilcoxon (min p=0.125); report mean delta and windows won."""
import sys, glob, os
import numpy as np, pandas as pd
rd, out = sys.argv[1], sys.argv[2]
df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(rd, '*.csv')) if 'partial' not in f], ignore_index=True)
M = ['Recall', 'FPR', 'alerts_per_10k_benign', 'Prec', 'F1', 'MCC', 'Cost20', 'AP', 'AUC', 'Brier', 'ECE']
g = df.groupby(['dataset', 'learner', 'strategy', 'thr_mode'])[M].mean().reset_index()
g.to_csv(out + '_means.csv', index=False)
rows = []
for (d, l, m), t in df.groupby(['dataset', 'learner', 'thr_mode']):
    b = t[t.strategy == 'BASE'].set_index('window')
    for s, u in t.groupby('strategy'):
        if s == 'BASE': continue
        u = u.set_index('window'); r = dict(dataset=d, learner=l, thr_mode=m, strategy=s)
        for k in M:
            dd = (u[k] - b[k]).dropna()
            r['d_' + k] = dd.mean(); r['win_' + k] = int((dd < 0).sum() if k in ('FPR','alerts_per_10k_benign','Cost20','Brier','ECE') else (dd > 0).sum())
        r['n_windows'] = len(u)
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(out + '_delta_vs_base.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
for m in ('fpr1', 'fpr01', 'tuned', 't05'):
    print('\n=== mode', m, '(means over windows; per dataset/learner/strategy) ===')
    print(g[g.thr_mode == m][['dataset', 'learner', 'strategy', 'Recall', 'FPR', 'alerts_per_10k_benign', 'F1', 'Cost20', 'AP', 'Brier', 'ECE']].round(4).to_string(index=False))
