# Generates cq5_run.py from cq4_run.py: adds prediction-powered threshold rules ppi_<f>. Existing rules unchanged.
s = open('cq4_run.py').read()
anchor = "from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split"
assert s.count(anchor) == 1
fn = '''def ppi_thr(pv, yv, yt, asel, alpha, fallback):
    # prediction-powered FPR estimate: observed-negative pool as proxy + audit rectifier; threshold = first grid point with estimated FPR <= alpha
    nA = int(asel.sum())
    if nA < 20: return fallback
    S0 = np.sort(pv[yv == 0]); N = len(pv); T0 = np.sort(pv[asel & (yt == 0)]); U0 = np.sort(pv[asel & (yv == 0)])
    den = len(S0) / N + (len(T0) - len(U0)) / nA
    if den <= 0: return fallback
    grid = np.unique(np.quantile(pv, np.linspace(0.5, 1.0, 2001)))
    num = (len(S0) - np.searchsorted(S0, grid, side='right')) / N + ((len(T0) - np.searchsorted(T0, grid, side='right')) - (len(U0) - np.searchsorted(U0, grid, side='right'))) / nA
    ok = np.where(num / den <= alpha)[0]
    return float(grid[ok[0]]) if len(ok) else float(grid[-1])
'''
s = s.replace(anchor, fn + anchor)
old = "                            aud_n['aud_%g' % fr] = aud_n['audnp_%g' % fr] = int(len(aneg))\n"
assert s.count(old) == 1
new = old + "                            thrs['ppi_%g' % fr] = ppi_thr(pv, yv, y[iva], asel, alpha, thrs['naive']); aud_n['ppi_%g' % fr] = int(asel.sum())\n"
s = s.replace(old, new)
open('cq5_run.py', 'w').write(s); print('ok')
