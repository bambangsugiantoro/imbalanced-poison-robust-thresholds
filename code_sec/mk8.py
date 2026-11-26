# Generates cq8_run.py from cq7_run.py: adds AST v2 (ast2_<f>), BAF loader (source baf:baf) and optional XGB-GPU (env XGB_GPU=1). Older rules unchanged.
s = open('cq7_run.py').read()
anchor = "from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split"
assert s.count(anchor) == 1
fn = '''AST2_CODE = {}
def ast2_thr(pv, yv, yt, B, alpha, seed, fallback, key):
    from sklearn.isotonic import IsotonicRegression
    rs = np.random.RandomState(seed)
    obs = np.where(yv == 0)[0]; so = pv[obs]; pos = pv[yv == 1]
    u = np.quantile(pos, 0.02) if len(pos) >= 20 else np.quantile(so, 1 - 5 * alpha)
    R = obs[so > u]
    if len(R) < 30 or B < 20: AST2_CODE[key] = 1; return fallback
    B1 = min(max(10, B // 2), len(R)); a1 = rs.choice(R, B1, replace=False)
    iso = IsotonicRegression(increasing=True, y_min=0, y_max=1, out_of_bounds='clip').fit(pv[a1], (yt[a1] == 1).astype(float))
    p_hat = lambda x: np.clip(iso.predict(x), 0.02, 0.98)
    grid = np.unique(np.quantile(so, np.linspace(0.02, 1, 2000)))
    order = np.argsort(so); ss = so[order]; pr = p_hat(so); Ntot = len(so) - pr.sum()
    cm = np.concatenate([[0], np.cumsum((1 - pr)[order][::-1])])[::-1]; idx = np.searchsorted(ss, grid, side='right')
    ok = np.where(cm[idx] / Ntot <= alpha)[0]; t0 = grid[ok[0]] if len(ok) else grid[-1]
    R2 = obs[pv[obs] >= t0]; w = np.sqrt(p_hat(pv[R2]) * (1 - p_hat(pv[R2])) + 0.02)
    B2 = B - B1
    if B2 > 0 and len(R2):
        w = 0.75 * w / w.mean() + 0.25   # defensive mixture with uniform sampling
        pi2 = np.minimum(1.0, B2 * w / w.sum())
    else: pi2 = np.zeros(len(R2))
    a2 = R2[rs.rand(len(R2)) < pi2]
    pi1f = np.zeros(len(pv)); pi1f[R] = B1 / len(R); pi2f = np.zeros(len(pv)); pi2f[R2] = pi2
    pit = 1 - (1 - pi1f) * (1 - pi2f); aud = np.zeros(len(pv), bool); aud[a1] = True; aud[a2] = True
    inc = pit[obs]; m = 1 - p_hat(so); cvec = (yt[obs] == 0).astype(float)
    est = m + np.where(aud[obs], (cvec - m) / np.maximum(inc, 1e-9), 0.0)
    N0 = est.sum()
    if N0 <= 0: AST2_CODE[key] = 2; return fallback
    cs = np.concatenate([[0], np.cumsum(est[order][::-1])])[::-1]
    ok = np.where(cs[idx] / N0 <= alpha)[0]
    if len(ok): AST2_CODE[key] = 0; return float(grid[ok[0]])
    AST2_CODE[key] = 3; return float(grid[-1])

def load_baf(path='data_new/baf_sub5.csv.gz'):
    """Bank Account Fraud (NeurIPS 2022) Base variant, systematic 1-in-5 subsample (200k rows). Ordered by month (stable); month and label dropped from features."""
    import pandas as pd
    d = pd.read_csv(path)
    o = np.argsort(d['month'].values, kind='stable'); y = d['fraud_bool'].astype(int).values
    X = d.drop(columns=['fraud_bool', 'month'])
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]): X[c] = X[c].astype('category').cat.codes.astype(float)
    X = X.astype('float32').replace([np.inf, -np.inf], np.nan); X = X.fillna(X.median()).fillna(0).values
    return X[o], y[o], 'baf'

if os.environ.get('XGB_GPU'):
    import functools; from xgboost import XGBClassifier as _XC
    rb.XGBClassifier = functools.partial(_XC, device='cuda', tree_method='hist')
'''
s = s.replace(anchor, fn + anchor)
old = "    elif kind0 == 'unsw':"
assert s.count(old) == 1
s = s.replace(old, "    elif kind0 == 'baf':\n        name = 'baf'; X, y, _ = load_baf(); X, y, sp = splits_temporal(X, y); qs_def = '0.99:fpr1,0.999:fpr01'\n" + old)
old2 = "aud_n['ast_%g' % fr] = Bf\n"
assert s.count(old2) == 1
s = s.replace(old2, old2 + "                            thrs['ast2_%g' % fr] = ast2_thr(pv, yv, y[iva], Bf, alpha, s_['seed'] + 377 + int(fr * 10000), thrs['naive'], ('ast2_%g' % fr)); aud_n['ast2_%g' % fr] = Bf * 10 + AST2_CODE.get('ast2_%g' % fr, 0)\n")
open('cq8_run.py', 'w').write(s); print('ok')
