# Generates cq6_run.py from cq3_run.py: adds exponential-tilting contamination estimate and rule 'tilt'. Existing rules unchanged.
s = open('cq3_run.py').read()
anchor = "from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split"
assert s.count(anchor) == 1
fn = '''def tilt_pi_est(U, L, seed, fallback, maxn=20000):
    from scipy.optimize import minimize
    from scipy.special import expit
    if len(L) < 20 or len(U) < 50: return fallback
    rs = np.random.RandomState(seed + 999)
    if len(U) > maxn: U = U[rs.choice(len(U), maxn, replace=False)]
    if len(L) > maxn: L = L[rs.choice(len(L), maxn, replace=False)]
    sc = np.concatenate([U, L]); lab = np.concatenate([np.zeros(len(U)), np.ones(len(L))]); rho = len(L) / len(U)
    sc = np.log(np.clip(sc, 1e-6, 1 - 1e-6) / (1 - np.clip(sc, 1e-6, 1 - 1e-6))) if sc.min() >= 0 and sc.max() <= 1 else sc
    x = (sc - sc.mean()) / (sc.std() + 1e-12)
    def nll(p):
        a, b, lp = p; pi = expit(lp) * 0.9; z = np.clip(a + b * x, -30, 30)
        h = np.exp(z) / ((1 - pi) + pi * np.exp(z)); pr = np.clip(rho * h / (1 + rho * h), 1e-9, 1 - 1e-9)
        return -(lab * np.log(pr) + (1 - lab) * np.log(1 - pr)).sum()
    best = None
    for a0, lp0 in ((0.0, -2.0), (2.0, 0.0)):
        r = minimize(nll, [a0, 1.0, lp0], method='Nelder-Mead', options={'maxiter': 600, 'xatol': 1e-4, 'fatol': 1e-4})
        if best is None or r.fun < best.fun: best = r
    return float(expit(best.x[2]) * 0.9)
'''
s = s.replace(anchor, fn + anchor)
a1 = "                    pi_hat = cq_lib.estimate_pi(neg, posv)\n"
assert s.count(a1) == 1
s = s.replace(a1, a1 + "                    pi_tilt = tilt_pi_est(neg, posv, s_['seed'], pi_hat)\n")
a2 = "                        thrs['cq'] = cq_lib.cq_threshold(neg, posv, alpha, pi_hat)[0]\n"
assert s.count(a2) == 1
s = s.replace(a2, a2 + "                        thrs['tilt'] = cq_lib.cq_threshold(neg, posv, alpha, pi_tilt)[0]\n")
a3 = "pi_tpcf=pi_tpcf,"
assert s.count(a3) == 1
s = s.replace(a3, a3 + " pi_tilt=pi_tilt,")
open('cq6_run.py', 'w').write(s); print('ok')
