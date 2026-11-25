import glob, sys, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
res = sys.argv[1] if len(sys.argv) > 1 else 'results'
df = pd.concat([pd.read_csv(f) for f in glob.glob(res + '/*.csv')], ignore_index=True)
print('datasets', df.dataset.nunique(), 'rows', len(df))
# average over folds -> one value per dataset x learner x strategy x thr_mode
g = df.groupby(['dataset', 'learner', 'strategy', 'thr_mode']).mean(numeric_only=True).reset_index()
metrics = ['AP', 'AUC', 'F1', 'MCC', 'Brier', 'ECE', 'Cost5']
def paired(metric, ref=('BASE', 'tuned'), mode='tuned', higher=True):
    out = []
    for lrn in sorted(g.learner.unique()):
        r = g[(g.learner == lrn) & (g.strategy == ref[0]) & (g.thr_mode == ref[1])].set_index('dataset')[metric]
        for s in sorted(g.strategy.unique()):
            if s == ref[0] and mode == ref[1]: continue
            a = g[(g.learner == lrn) & (g.strategy == s) & (g.thr_mode == mode)].set_index('dataset')[metric].reindex(r.index)
            d = (a - r) * (1 if higher else -1)
            try: p = wilcoxon(d.dropna()).pvalue
            except Exception: p = np.nan
            out.append(dict(learner=lrn, strategy=s, mode=mode, metric=metric, mean_delta=d.mean(),
                            wins=int((d > 1e-9).sum()), ties=int((d.abs() <= 1e-9).sum()), losses=int((d < -1e-9).sum()), p=p))
    o = pd.DataFrame(out); o['p_holm'] = multipletests(o.p.fillna(1), method='holm')[1]; return o
tabs = []
for m, hi in [('F1', True), ('MCC', True), ('AP', True), ('Cost5', False), ('Brier', False), ('ECE', False)]:
    tabs.append(paired(m, mode='tuned', higher=hi))
    tabs.append(paired(m, mode='t05', higher=hi))   # vs tuned baseline: decision-rule effect included
T = pd.concat(tabs); T.to_csv(res + '_paired_vs_tuned_baseline.csv', index=False)
print(T[(T['mode'] == 'tuned') & (T.metric.isin(['F1', 'AP', 'Brier']))].round(3).to_string(index=False))
