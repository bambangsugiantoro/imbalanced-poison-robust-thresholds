"""Confirmation analysis (frozen rules, PRESPEC_KONFIRMASI.md). usage: python analyze_conf.py DIR [DIR ...]
Per dataset x noise x budget: paired Wilcoxon over units (split, learner, strategy), poison pooled 0.2 and 0.4 reported separately;
rule vs naive on abs calibration |log2(FPR/target)|; Holm over all tests; bootstrap 95% CI of mean difference. Units share windows (not independent): p-values are descriptive."""
import sys, glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
pd.set_option('display.width', 220)
fs = [f for d in sys.argv[1:] for f in glob.glob(os.path.join(d, '*.csv')) if '.partial' not in f]
D = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
D['acal'] = np.abs(np.log2(np.maximum(D['FPR'], 1e-4) / D['target']))
rng = np.random.RandomState(1)
rows = []
for (ds, nz, op, p), g in D[D.poison >= 0.2].groupby(['dataset', 'noise', 'op', 'poison']):
    pv = g.pivot_table(index=['split', 'learner', 'strategy'], columns='var', values='acal')
    for r in ['cq', 'trust', 'oracle', 'conf', 'npu', 'cl']:
        if r not in pv: continue
        d = (pv[r] - pv['naive']).dropna().values
        if len(d) < 6: continue
        try: pw = wilcoxon(d).pvalue
        except Exception: pw = np.nan
        bs = [rng.choice(d, len(d)).mean() for _ in range(1000)]
        rows.append((ds, nz, op, p, r, len(d), d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (d < 0).mean(), pw))
R = pd.DataFrame(rows, columns=['dataset', 'noise', 'op', 'poison', 'rule', 'n', 'mean_diff', 'ci_lo', 'ci_hi', 'share_better', 'p'])
m = R['p'].notna().sum(); R = R.sort_values('p')
R['p_holm'] = np.minimum(1, np.maximum.accumulate(R['p'].values * (m - np.arange(len(R)))))
print(R.sort_values(['dataset', 'noise', 'op', 'poison', 'rule']).round(4).to_string(index=False))
print('\nH3 check (cq vs naive, poison>=0.2): Holm p<0.05 and mean_diff<0')
h = R[R.rule == 'cq']; print(h.assign(H3=(h.p_holm < 0.05) & (h.mean_diff < 0))[['dataset', 'noise', 'op', 'poison', 'mean_diff', 'p_holm', 'H3']].sort_values(['dataset', 'noise', 'op', 'poison']).round(4).to_string(index=False))
