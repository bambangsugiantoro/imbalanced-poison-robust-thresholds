"""PRESPEC_AST analysis. usage: python analyze_conf7.py DIR [DIR ...]. HS1: ast_f vs aud_f (equal budget); HS2: ast_0.005 vs naive; poison>=0.2; paired Wilcoxon over (split,learner,strategy); Holm within hypothesis; descriptive p."""
import sys, glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
pd.set_option('display.width', 220)
fs = [f for d in sys.argv[1:] for f in glob.glob(os.path.join(d, '*.csv')) if '.partial' not in f]
D = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
D['acal'] = np.abs(np.log2(np.maximum(D['FPR'], 1e-4) / D['target']))
rows = []
for (ds, nz, op, p), g in D[D.poison >= 0.2].groupby(['dataset', 'noise', 'op', 'poison']):
    pv = g.pivot_table(index=['split', 'learner', 'strategy'], columns='var', values='acal')
    pairs = [('HS1_' + f, 'ast_' + f, 'aud_' + f) for f in ['0.002', '0.005', '0.01']] + [('HS2', 'ast_0.005', 'naive'), ('HS2b', 'ast_0.01', 'naive'), ('ref_oracle', 'oracle', 'naive')]
    for h, a, b in pairs:
        if a not in pv or b not in pv: continue
        d = (pv[a] - pv[b]).dropna().values
        if len(d) < 6: continue
        try: pw = wilcoxon(d).pvalue
        except Exception: pw = np.nan
        rows.append((h, ds, nz, op, p, len(d), d.mean(), (d < 0).mean(), pw, pv[a].mean(), pv[b].mean()))
R = pd.DataFrame(rows, columns=['hyp', 'dataset', 'noise', 'op', 'poison', 'n', 'mean_diff', 'share_better', 'p', 'acal_a', 'acal_b'])
R['p_holm'] = np.nan
for h, g in R.groupby('hyp'):
    pp = g['p'].fillna(1.0); o = pp.sort_values(); m = len(o)
    adj = np.minimum(1, np.maximum.accumulate(o.values * (m - np.arange(m))))
    R.loc[o.index, 'p_holm'] = adj
R['pass'] = (R.p_holm < 0.05) & (R.mean_diff < 0)
print(R.round(4).to_string(index=False))
print('\nPASS COUNTS per hypothesis (cells passed / cells):')
print(R.groupby('hyp')['pass'].agg(['sum', 'count']))
print('\nCells where ast_f worse than naive (HS2):'); print(R[(R.hyp.str.startswith('HS2')) & (R.mean_diff > 0)][['hyp', 'dataset', 'noise', 'op', 'poison', 'mean_diff']].round(3).to_string(index=False))
