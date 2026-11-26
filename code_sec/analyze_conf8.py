"""PRESPEC_AST2 analysis. usage: python analyze_conf8.py DIR [DIR ...]"""
import sys, glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
pd.set_option('display.width', 220)
fs = [f for d in sys.argv[1:] for f in glob.glob(os.path.join(d, '*.csv')) if '.partial' not in f]
D = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
D['acal'] = np.abs(np.log2(np.maximum(D['FPR'], 1e-4) / D['target']))
rows = []
for (ds, nz, op, p), g in D[D.poison >= 0.2].groupby(['dataset', 'noise', 'op', 'poison']):
    pv = g.pivot_table(index=['split', 'learner', 'strategy'], columns='var', values='acal')
    pairs = [('HV1_0.005', 'ast2_0.005', 'aud_0.005'), ('HV1_0.01', 'ast2_0.01', 'aud_0.01'), ('HV2', 'ast2_0.005', 'naive'), ('HV2b', 'ast2_0.01', 'naive'), ('HV3_0.005', 'ast2_0.005', 'ast_0.005'), ('HV3_0.01', 'ast2_0.01', 'ast_0.01'), ('v1_vs_naive', 'ast_0.005', 'naive'), ('ref_oracle', 'oracle', 'naive')]
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
    R.loc[o.index, 'p_holm'] = np.minimum(1, np.maximum.accumulate(o.values * (m - np.arange(m))))
R['pass'] = (R.p_holm < 0.05) & (R.mean_diff < 0)
print(R.round(4).to_string(index=False))
print('\nPASS COUNTS per hypothesis (needs >=7/12 for HV1, HV2):'); print(R.groupby('hyp')['pass'].agg(['sum', 'count']))
print('\nHV3: cells with ast2 lower mean acal than ast v1:'); print(R[R.hyp.str.startswith('HV3')].groupby('hyp').apply(lambda g: (g.mean_diff < 0).sum()).to_string())
A = D[D['var'].str.startswith('ast2_')].copy(); A['code'] = A['n_aud'] % 10
print('\nHV4 diagnostic codes (0 normal,1 fallback R/B,2 N0<=0,3 grid exhausted):'); print(pd.crosstab([A.op, A.poison], A.code))
print('\nCells where ast2 worse than naive:'); print(R[(R.hyp.str.startswith('HV2')) & (R.mean_diff > 0)][['hyp', 'noise', 'op', 'poison', 'mean_diff']].round(3).to_string(index=False))
