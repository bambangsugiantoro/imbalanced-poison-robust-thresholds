"""Audit-budget analysis (PRESPEC_AUDIT.md). usage: python analyze_audit.py DIR [DIR ...]
HA1: aud_f (f>=1%) vs naive, abs calibration, Wilcoxon per dataset x noise x op x poison(>=0.2), Holm over all tests.
HA2: log-log slope of mean abs calibration vs n_a*alpha (cells with n_a*alpha>=1), per noise, bootstrap CI over cells.
HA3: share of test windows with FPR > target for audnp_f (reported by noise, f)."""
import sys, glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
pd.set_option('display.width', 220)
fs = [f for d in sys.argv[1:] for f in glob.glob(os.path.join(d, '*.csv')) if '.partial' not in f]
D = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
D['acal'] = np.abs(np.log2(np.maximum(D['FPR'], 1e-4) / D['target']))
rng = np.random.RandomState(1); P = D[D.poison >= 0.2]; rows = []
for (ds, nz, op, p), g in P.groupby(['dataset', 'noise', 'op', 'poison']):
    pv = g.pivot_table(index=['split', 'learner', 'strategy'], columns='var', values='acal')
    for r in [c for c in pv.columns if c.startswith('aud_') and float(c[4:]) >= 0.01] + ['trust', 'oracle']:
        d = (pv[r] - pv['naive']).dropna().values
        if len(d) < 6: continue
        try: pw = wilcoxon(d).pvalue
        except Exception: pw = np.nan
        rows.append((ds, nz, op, p, r, len(d), d.mean(), pw))
R = pd.DataFrame(rows, columns=['dataset', 'noise', 'op', 'poison', 'rule', 'n', 'mean_diff', 'p'])
m = R['p'].notna().sum(); R = R.sort_values('p')
R['p_holm'] = np.minimum(1, np.maximum.accumulate(R['p'].fillna(1).values * (m - np.arange(len(R)))))
R['pass'] = (R.p_holm < 0.05) & (R.mean_diff < 0)
print('== HA1 per dataset x noise: passing cells / cells (aud_f f>=1%) =='); A = R[R.rule.str.startswith('aud_')]
print(A.groupby(['dataset', 'noise'])['pass'].agg(['sum', 'count']).to_string())
print('\n== HA1 full table =='); print(R.sort_values(['dataset', 'noise', 'op', 'poison', 'rule']).round(4).to_string(index=False))
print('\n== HA2 audit-budget law: slope of log(mean acal) on log(n_a*alpha) ==')
Q = P[P['var'].str.startswith('aud_') & (P.n_aud > 0)].copy()
Q['x'] = Q.n_aud * Q.target
C = Q.groupby(['dataset', 'noise', 'op', 'poison', 'var']).agg(x=('x', 'mean'), y=('acal', 'mean')).reset_index(); C = C[(C.x >= 1) & (C.y > 0)]
for nz, g in C.groupby('noise'):
    lx, ly = np.log(g.x.values), np.log(g.y.values)
    sl = np.polyfit(lx, ly, 1)[0]; bs = []
    for _ in range(1000):
        i = rng.randint(0, len(g), len(g)); bs.append(np.polyfit(lx[i], ly[i], 1)[0])
    print(nz, 'cells', len(g), 'slope', round(sl, 3), 'CI', np.round(np.percentile(bs, [2.5, 97.5]), 3), 'in[-0.6,-0.4]:', -0.6 <= sl <= -0.4)
print('\n== HA3 share of windows with FPR > target, audnp_f ==')
N = D[D['var'].str.startswith('audnp_')].assign(v=lambda d: d.FPR > d.target)
print(N.groupby(['dataset', 'noise', 'var'])['v'].mean().round(3).unstack().to_string())
