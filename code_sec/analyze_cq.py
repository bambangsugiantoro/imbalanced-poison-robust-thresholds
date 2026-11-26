"""Calibration analysis for CQ vs baselines. usage: python analyze_cq.py DIR [DIR2...] > out.txt
Reads NAME.csv files (not .partial). Metrics per (dataset, poison, op, var), averaged over splits/learners/strategies:
cal = mean log2(max(FPR,floor)/target) ; ok15 = share FPR<=1.5*target ; pen_recall = recall with 0 if FPR>1.5*target ;
regret = ideal recall - recall. Tests: per-dataset Wilcoxon (dataset-level means) CQ vs naive/oracle/cap25/gcap25, Holm."""
import sys, glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
fs = [f for d in sys.argv[1:] for f in glob.glob(os.path.join(d, '*.csv')) if '.partial' not in f]
D = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
D = D[D['var'].isin(['naive','oracle','cq','cap25','gcap25','ideal'])]
fl = 1e-4
D['cal'] = np.log2(np.maximum(D['FPR'], fl) / D['target'])
D['ok15'] = (D['FPR'] <= 1.5 * D['target']).astype(float)
D['ok1'] = (D['FPR'] <= D['target']).astype(float)
D['penR'] = np.where(D['FPR'] <= 1.5 * D['target'], D['Recall'], 0.0)
print('files', len(fs), 'rows', len(D), 'datasets', D['dataset'].nunique())
keys = ['poison', 'op', 'var']
g = D.groupby(keys).agg(n=('cal', 'size'), FPR=('FPR', 'mean'), cal=('cal', 'mean'), abscal=('cal', lambda x: np.abs(x).mean()), ok15=('ok15', 'mean'), ok1=('ok1', 'mean'), Recall=('Recall', 'mean'), penR=('penR', 'mean')).round(3)
print('\n== Overall (mean over all rows) =='); print(g.to_string())
print('\n== pi_hat vs pi_true (cq rows) ==')
c = D[D['var'] == 'cq'].groupby('poison').agg(pi_hat=('pi_hat', 'mean'), pi_true=('pi_true', 'mean'), mae=('pi_hat', lambda x: 0)).round(3)
c['mae'] = D[D['var'] == 'cq'].assign(e=lambda z: (z['pi_hat'] - z['pi_true']).abs()).groupby('poison')['e'].mean().round(3)
print(c.to_string())
print('\n== Tests per dataset (dataset-level mean abs calibration error; lower better), Holm ==')
for (p, op), s in D.groupby(['poison', 'op']):
    m = s.assign(a=s['cal'].abs()).groupby(['dataset', 'var'])['a'].mean().unstack()
    res = []
    for other in ['naive', 'oracle', 'cap25', 'gcap25']:
        if other not in m or 'cq' not in m: continue
        x = m[['cq', other]].dropna()
        if len(x) < 6 or np.allclose(x['cq'], x[other]): continue
        w = wilcoxon(x['cq'], x[other]); res.append((other, len(x), float((x['cq'] < x[other]).mean()), w.pvalue))
    ps = np.array([r[3] for r in res]); order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
    for rank, i in enumerate(order): run = max(run, min(1, ps[i] * (len(ps) - rank))); adj[i] = run
    for r, a in zip(res, adj): print('poison %.1f op %s CQ vs %-7s n=%d share CQ better %.2f  p=%.4g Holm=%.4g' % (p, op, r[0], r[1], r[2], r[3], a))
print('\n== Per temporal dataset (non-b49): FPR ratio mean / ok15 ==')
T = D[D['dataset'].isin(['creditcard', 'ieee', 'unsw'])]
print(T.groupby(['dataset', 'poison', 'op', 'var']).agg(ratio=('FPR', lambda x: 0), FPR=('FPR', 'mean'), ok15=('ok15', 'mean'), Recall=('Recall', 'mean')).drop(columns='ratio').round(4).to_string())
