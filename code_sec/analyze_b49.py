"""Across-dataset analysis for bench49_poison outputs. usage: python analyze_b49.py RESULTS_DIR OUT_PREFIX"""
import sys, glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon
rd, out = sys.argv[1], sys.argv[2]
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(rd + '/*.csv'))])
d = d[d.status == 'ok'] if 'status' in d else d
d['var'] = d.thr_mode.str.rsplit('_', n=1).str[0]; d['op'] = d.thr_mode.str.rsplit('_', n=1).str[1]
d['target'] = d.op.map({'fpr5': 0.05, 'fpr1': 0.01}); d['fpr_ratio'] = d.FPR / d.target
# per-dataset means over folds, learners, strategies
g = d.groupby(['dataset', 'op', 'poison', 'var'])[['Recall', 'FPR', 'fpr_ratio']].mean().reset_index()
g.to_csv(out + '_perdataset.csv', index=False)
print('datasets:', g.dataset.nunique())
S = g.groupby(['op', 'poison', 'var'])[['Recall', 'FPR', 'fpr_ratio']].mean().round(4)
print('\nMean over datasets:'); print(S.unstack('var').to_string())
variants = ['oracle', 'cap50', 'cap25', 'gcap50', 'gcap25']
res = []
for op in ['fpr5', 'fpr1']:
    for pz in sorted(g.poison.unique()):
        s = g[(g.op == op) & (g.poison == pz)].pivot(index='dataset', columns='var', values='Recall')
        r = g[(g.op == op) & (g.poison == pz)].pivot(index='dataset', columns='var', values='fpr_ratio')
        for v in variants:
            dd = (s[v] - s['naive']).dropna()
            p = 1.0 if (dd == 0).all() else wilcoxon(dd).pvalue
            res.append(dict(op=op, poison=pz, variant=v, mean_dRecall=dd.mean(), wins=int((dd > 1e-9).sum()), losses=int((dd < -1e-9).sum()), n=len(dd), p=p,
                            fpr_ratio_mean=r[v].mean(), fpr_ratio_naive=r['naive'].mean(), share_fpr_gt1p5=float((r[v] > 1.5).mean())))
R = pd.DataFrame(res)
R['p_holm'] = np.nan
for key, sub in R.groupby(['op', 'poison']):
    order = sub.p.sort_values().index; m = len(order); run = 0
    for i, ix in enumerate(order):
        run = max(run, min(1.0, (m - i) * R.loc[ix, 'p'])); R.loc[ix, 'p_holm'] = run
R.round(4).to_csv(out + '_tests.csv', index=False)
print('\nWilcoxon vs naive per dataset (Holm within op x poison):'); print(R.round(4).to_string(index=False))
