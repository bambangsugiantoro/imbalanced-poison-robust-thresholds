"""Strategy selector (leave-one-dataset-out). usage: python meta_selector.py RESULTS_DIR OUT_PREFIX
env METRIC (default F1), MODE (default tuned). Run from ~/imb_bench; imports code/run_bench.py for load()."""
import sys, glob, os, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.getcwd(), 'code'))
import run_bench as rb
RD = sys.argv[1] if len(sys.argv) > 1 else 'results'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'meta'
METRIC = os.environ.get('METRIC', 'F1'); MODE = os.environ.get('MODE', 'tuned')

def meta_features(X, y):
    X = np.nan_to_num(np.asarray(X, float)); y = np.asarray(y)
    labs, cnt = np.unique(y, return_counts=True)
    yb = (y == labs[np.argmin(cnt)]).astype(int)
    n, d = X.shape; nm = int(yb.sum())
    m1, m0 = X[yb == 1].mean(0), X[yb == 0].mean(0)
    v1, v0 = X[yb == 1].var(0), X[yb == 0].var(0)
    f = (m1 - m0) ** 2 / (v1 + v0 + 1e-12)
    disc = np.mean([len(np.unique(X[:, j])) <= 10 for j in range(d)])
    rs = np.random.RandomState(0)
    if n > 3000:
        i = rs.choice(n, 3000, replace=False)
        if yb[i].sum() < 5: i = np.concatenate([np.flatnonzero(yb == 1), rs.choice(np.flatnonzero(yb == 0), 2500, replace=False)])
        Xs, ys = X[i], yb[i]
    else: Xs, ys = X, yb
    try: auc = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=300)), Xs, ys, cv=3, scoring='roc_auc').mean()
    except Exception: auc = 0.5
    return dict(n=n, logn=np.log10(n), d=d, IR=(n - nm) / max(nm, 1), logIR=np.log10((n - nm) / max(nm, 1)), n_min=nm, nmin_per_d=nm / d,
                disc_frac=disc, fisher_max=f.max(), fisher_mean=f.mean(), lr_auc=auc)

rows = []
for fn in sorted(glob.glob(RD + '/*.csv')):
    df = pd.read_csv(fn)
    df = df[(df.status == 'ok') & (df.thr_mode == MODE)]
    rows.append(df[['dataset', 'fold', 'learner', 'strategy', METRIC]])
R = pd.concat(rows)
g = R.groupby(['dataset', 'learner', 'strategy'])[METRIC].mean().unstack('strategy').dropna(subset=['BASE'])
D = g.sub(g['BASE'], axis=0).drop(columns='BASE').stack().rename('delta').reset_index().dropna(subset=['delta'])
names = sorted(D.dataset.unique())
cache = OUT + '_metafeat.csv'
if os.path.exists(cache): M = pd.read_csv(cache, index_col=0)
else:
    M = {}
    for nm in names:
        try:
            X, y = rb.load(nm); M[nm] = meta_features(X, y)
        except Exception as e: print('skip', nm, e)
    M = pd.DataFrame(M).T; M.to_csv(cache)
D = D[D.dataset.isin(M.index)]
F = D.join(M, on='dataset')
mcols = list(M.columns)
X = pd.concat([F[mcols].astype(float), pd.get_dummies(F.learner, prefix='L').astype(float), pd.get_dummies(F.strategy, prefix='S').astype(float)], axis=1)
y = F.delta.values; grp = F.dataset.values
pred = np.zeros(len(F))
for ds in np.unique(grp):
    te = grp == ds
    rf = RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(X[~te], y[~te]); pred[te] = rf.predict(X[te])
F['pred'] = pred
strats = sorted(F.strategy.unique())
res = []
for (ds, ln), s in F.groupby(['dataset', 'learner']):
    best = s.loc[s.pred.idxmax()]
    pol = best.delta if best.pred > 0 else 0.0
    others = F[(F.dataset != ds) & (F.learner == ln)]
    fixed = others.groupby('strategy').delta.mean(); fb = fixed.idxmax()
    fv = s[s.strategy == fb].delta; fixed_pol = fv.iloc[0] if (fixed.max() > 0 and len(fv)) else 0.0
    d = dict(dataset=ds, learner=ln, policy=pol, fixed_lodo=fixed_pol, oracle=max(0.0, s.delta.max()), always_base=0.0,
             chose=(best.strategy if best.pred > 0 else 'BASE'))
    for st in strats:
        v = s[s.strategy == st].delta; d['always_' + st] = v.iloc[0] if len(v) else np.nan
    res.append(d)
P = pd.DataFrame(res); P.to_csv(OUT + '_policy_rows.csv', index=False)
print(f'metric={METRIC} mode={MODE} datasets={len(names)} rows={len(P)}')
cols = ['always_base', 'policy', 'fixed_lodo', 'oracle'] + ['always_' + s for s in strats]
print('\nMean delta vs BASE (tuned thr) per learner:'); print(P.groupby('learner')[cols].mean().round(4).T.to_string())
per_ds = P.groupby('dataset')[cols].mean()
print('\nOverall (mean of per-dataset means):'); print(per_ds.mean().round(4).to_string())
def wt(a, b):
    d = np.asarray(a) - np.asarray(b)
    return 1.0 if np.all(d == 0) else wilcoxon(d).pvalue
print('\nWilcoxon (per-dataset, n=%d): policy vs always_base p=%.4f; policy vs fixed_lodo p=%.4f' % (len(per_ds), wt(per_ds.policy, 0), wt(per_ds.policy, per_ds.fixed_lodo)))
print('policy wins/ties/losses vs base: ', int((per_ds.policy > 1e-12).sum()), int((per_ds.policy.abs() <= 1e-12).sum()), int((per_ds.policy < -1e-12).sum()))
print('choice counts:'); print(P.chose.value_counts().to_string())
rf = RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1).fit(X, y)
print('\nFeature importance:'); print(pd.Series(rf.feature_importances_, X.columns).sort_values(ascending=False).head(10).round(3).to_string())
