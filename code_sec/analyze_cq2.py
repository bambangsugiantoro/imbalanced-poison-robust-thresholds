"""Analysis of cq2 runs (new baselines, strategies, delayed labels). usage: python analyze_cq2.py DIR [DIR ...]  (dirs may be results_cq2_b49, results_cq2_t_random, results_cq2_t_delay)"""
import sys, glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 30)
fs = [f for d in sys.argv[1:] for f in glob.glob(os.path.join(d, '*.csv')) if '.partial' not in f]
D = pd.concat([pd.read_csv(f).assign(src=os.path.basename(os.path.dirname(f))) for f in fs], ignore_index=True)
D['acal'] = np.abs(np.log2(np.maximum(D['FPR'], 1e-4) / D['target']))
D['ok15'] = (D['FPR'] <= 1.5 * D['target']).astype(float)
D['penR'] = np.where(D['FPR'] <= 1.5 * D['target'], D['Recall'], 0.0)
print('rows', len(D), 'files', len(fs)); print(D.groupby('src').size())
RULES = ['naive', 'oracle', 'cq', 'conf', 'npu', 'cl', 'trust', 'cap25', 'gcap25', 'ideal']
b = D[D.src == 'results_cq2_b49']
if len(b):
    print('\n== b49: mean abs calibration error (log2) by rule x poison (dataset means) ==')
    ds = b.groupby(['dataset', 'poison', 'var', 'op'])[['acal', 'ok15', 'penR']].mean().reset_index()
    t = ds.pivot_table(index=['op', 'poison'], columns='var', values='acal')[[r for r in RULES if r in set(ds['var'])]]
    print(t.round(2).to_string())
    print('\n== b49: ok15 (share within 1.5x budget) ==')
    print(ds.pivot_table(index=['op', 'poison'], columns='var', values='ok15')[[r for r in RULES if r in set(ds['var'])]].round(2).to_string())
    print('\n== b49 RQ5: abs cal by strategy, poison 0.4, op fpr1, rules naive/cq/trust/npu/oracle ==')
    s = b[(b.poison == 0.4) & (b.op == 'fpr1') & b['var'].isin(['naive', 'cq', 'trust', 'npu', 'oracle'])]
    print(s.groupby(['strategy', 'var'])['acal'].mean().unstack().round(2).to_string())
    print('\n== b49: naive FPR/target ratio median by strategy (poison 0.4, fpr1) ==')
    n = b[(b.poison == 0.4) & (b.op == 'fpr1') & (b['var'] == 'naive')].copy(); n['ratio'] = n['FPR'] / n['target']
    print(n.groupby('strategy')['ratio'].median().round(2).to_string())
    print('\n== Wilcoxon (dataset-level mean abs cal), rule vs naive, Holm over tests ==')
    res = []
    for op in ['fpr5', 'fpr1']:
        for p in [0.0, 0.2, 0.4]:
            for r in ['cq', 'conf', 'npu', 'cl', 'trust', 'oracle']:
                a = ds[(ds.op == op) & (ds.poison == p) & (ds['var'] == r)].set_index('dataset')['acal']
                c = ds[(ds.op == op) & (ds.poison == p) & (ds['var'] == 'naive')].set_index('dataset')['acal']
                j = a.index.intersection(c.index)
                if len(j) < 10: continue
                d = (a[j] - c[j])
                try: pv = wilcoxon(d).pvalue
                except Exception: pv = np.nan
                res.append((op, p, r, len(j), d.mean(), (d < 0).mean(), pv))
    R = pd.DataFrame(res, columns=['op', 'poison', 'rule', 'n', 'mean_diff_vs_naive', 'share_better', 'p'])
    m = R['p'].notna().sum(); order = R['p'].rank(method='first'); R['p_holm'] = np.minimum(1, R['p'] * (m - order + 1))
    # monotone Holm
    R = R.sort_values('p'); R['p_holm'] = np.minimum(1, np.maximum.accumulate(R['p'].values * (m - np.arange(len(R)))))
    print(R.sort_values(['op', 'poison', 'rule']).round(4).to_string(index=False))
t_ = D[D.src != 'results_cq2_b49']
if len(t_):
    print('\n== temporal: abs cal / ok15 / penR / recall by noise x dataset x rule (poison 0.4, op fpr1) ==')
    x = t_[(t_.poison == 0.4) & (t_.op == 'fpr1')]
    g = x.groupby(['noise', 'dataset', 'var'])[['acal', 'ok15', 'penR', 'Recall']].mean().round(3)
    print(g.reindex([r for r in RULES], level=2).to_string())
    print('\n== temporal: realised FPR/target of naive, by noise x dataset x poison (op fpr1) ==')
    y = t_[(t_.op == 'fpr1') & (t_['var'] == 'naive')].copy(); y['ratio'] = y['FPR'] / y['target']
    print(y.groupby(['noise', 'dataset', 'poison'])['ratio'].mean().unstack().round(2).to_string())
    print('\n== temporal: pi_hat vs pi_true (cq rows), by noise x dataset (poison>0) ==')
    z = t_[(t_['var'] == 'cq') & (t_.poison > 0) & (t_.op == 'fpr1')]
    print(z.groupby(['noise', 'dataset'])[['pi_hat', 'pi_true']].mean().round(3).to_string())
