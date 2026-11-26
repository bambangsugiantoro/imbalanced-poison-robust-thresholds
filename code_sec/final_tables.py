"""Aggregates frozen-code runs (cq7: creditcard, IEEE-CIS, UNSW, Elliptic, CIC; cq8: BAF) into final_tables.csv: mean over windows, learners, strategies."""
import glob, numpy as np, pandas as pd
fs = [f for p in ('results_cq7_*', 'results_cq8_baf_*') for d in glob.glob(p) for f in glob.glob(d + '/*.csv') if 'partial' not in f]
D = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
D['acal'] = np.abs(np.log2(np.maximum(D.FPR, 1e-4) / D.target)); D['lr'] = np.log2(np.maximum(D.FPR, 1e-4) / D.target); D['safe'] = (D.FPR <= 1.5 * D.target).astype(float)
R = ['naive', 'oracle', 'cq', 'conf', 'npu', 'cl', 'trust', 'aud_0.002', 'aud_0.005', 'aud_0.01', 'aud_0.05', 'ast_0.005', 'ast2_0.005', 'ideal']
G = D[D['var'].isin(R)].groupby(['dataset', 'noise', 'op', 'poison', 'var']).agg(acal=('acal', 'mean'), lr=('lr', 'mean'), safe=('safe', 'mean'), recall=('Recall', 'mean'), n=('acal', 'size'), pi_true=('pi_true', 'mean'), pi_hat=('pi_hat', 'mean')).reset_index()
G.round(4).to_csv('final_tables.csv', index=False)
print(len(D), 'rows', D.dataset.nunique(), 'datasets', len(G), 'table rows'); print(D.groupby('dataset').size())
