s = open('cq4_run.py').read()
anchor = "from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split"
assert s.count(anchor) == 1
fn = '''def ast_thr(pv, yv, yt, B, alpha, seed, fallback):
    from sklearn.isotonic import IsotonicRegression
    rs = np.random.RandomState(seed)
    obs = np.where(yv == 0)[0]; so = pv[obs]; pos = pv[yv == 1]
    u = np.quantile(pos, 0.02) if len(pos) >= 20 else np.quantile(so, 1 - 5 * alpha)
    R = obs[so > u]
    if len(R) < 30 or B < 20: return fallback
    B1 = min(max(10, B // 2), len(R)); a1 = rs.choice(R, B1, replace=False)
    iso = IsotonicRegression(increasing=True, y_min=0, y_max=1, out_of_bounds='clip').fit(pv[a1], (yt[a1] == 1).astype(float))
    p_hat = lambda x: np.clip(iso.predict(x), 0.0, 1.0)
    grid = np.unique(np.quantile(so, np.linspace(0.5, 1, 1500)))
    order = np.argsort(so); ss = so[order]; pr = p_hat(so); Ntot = len(so) - pr.sum()
    cm = np.concatenate([[0], np.cumsum((1 - pr)[order][::-1])])[::-1]; idx = np.searchsorted(ss, grid, side='right')
    ok = np.where(cm[idx] / Ntot <= alpha)[0]; t0 = grid[ok[0]] if len(ok) else grid[-1]
    R2 = obs[pv[obs] >= t0]; w = np.sqrt(p_hat(pv[R2]) * (1 - p_hat(pv[R2])) + 0.02)
    B2 = B - B1; pi2 = np.minimum(1.0, B2 * w / w.sum()) if B2 > 0 and len(R2) else np.zeros(len(R2))
    a2 = R2[rs.rand(len(R2)) < pi2]
    pi1f = np.zeros(len(pv)); pi1f[R] = B1 / len(R); pi2f = np.zeros(len(pv)); pi2f[R2] = pi2
    pit = 1 - (1 - pi1f) * (1 - pi2f); aud = np.zeros(len(pv), bool); aud[a1] = True; aud[a2] = True
    inc = pit[obs]; m = 1 - p_hat(so); cvec = (yt[obs] == 0).astype(float)
    est = m + np.where(aud[obs], (cvec - m) / np.maximum(inc, 1e-9), 0.0)
    N0 = est.sum()
    if N0 <= 0: return fallback
    cs = np.concatenate([[0], np.cumsum(est[order][::-1])])[::-1]
    ok = np.where(cs[idx] / N0 <= alpha)[0]
    return float(grid[ok[0]]) if len(ok) else float(grid[-1])
'''
s = s.replace(anchor, fn + anchor)
old = "                            aud_n['aud_%g' % fr] = aud_n['audnp_%g' % fr] = int(len(aneg))\n"
assert s.count(old) == 1
new = old + "                            Bf = max(20, int(round(fr * len(iva)))); thrs['ast_%g' % fr] = ast_thr(pv, yv, y[iva], Bf, alpha, s_['seed'] + 277 + int(fr * 10000), thrs['naive']); aud_n['ast_%g' % fr] = Bf\n"
s = s.replace(old, new)
open('cq7_run.py', 'w').write(s); print('ok')
