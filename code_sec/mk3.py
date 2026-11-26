s=open('code_sec/cq2_run.py').read()
H='''
def delay2_flip(tr, y, lv, seed):
    """Age-based stochastic label delay at the snapshot (end of training window): a positive at relative age a in [0,1] (0 = newest) is still unlabelled
    with probability exp(-a/lam); lam is chosen by bisection so that the mean unlabelled share of positives equals lv. Returns indices to flip (positive -> observed negative)."""
    pos = tr[y[tr] == 1]
    if lv <= 0 or len(pos) == 0: return pos[:0]
    a = 1.0 - (np.searchsorted(tr, pos) + 0.5) / len(tr)
    lo, hi = 1e-3, 50.0
    for _ in range(60):
        lam = 0.5 * (lo + hi); m = np.mean(np.exp(-a / lam))
        if m < lv: lo = lam
        else: hi = lam
    rs = np.random.RandomState(seed + 4242)
    return pos[rs.rand(len(pos)) < np.exp(-a / lam)]

def tpc_estimates(yp_tr, ival_start, K=10, guard=2.0, pi_fallback=0.0):
    """Temporal-reference contamination estimates from observed labels only (validation window = yp_tr[ival_start:]).
    tpc : p_ref = observed positive rate of the first half; pi = (p_ref - r_val)/(1 - r_val).
    tpcf: fit r(age) = p (1 - exp(-age/mu)) over K chunks; pi = (p_hat - r_val)/(1 - r_val).
    Guard: if the positive rates of the first two quarters differ by more than a factor `guard`, prevalence is judged non-stationary and the fallback (CQ pi) is returned."""
    from scipy.optimize import curve_fit
    n = len(yp_tr); r_val = float(yp_tr[ival_start:].mean())
    q1, q2 = float(yp_tr[:n // 4].mean()), float(yp_tr[n // 4:n // 2].mean())
    ref = float(yp_tr[:n // 2].mean())
    if min(q1, q2) <= 0 or max(q1, q2) / min(q1, q2) > guard: return pi_fallback, pi_fallback, False
    cl = lambda p: float(min(max(p, 0.0), 0.9))
    pi_ref = cl((ref - r_val) / max(1e-9, 1 - r_val))
    ed = np.linspace(0, n, K + 1).astype(int); rk = np.array([yp_tr[ed[i]:ed[i + 1]].mean() for i in range(K)]); ag = 1.0 - (np.arange(K) + 0.5) / K
    try:
        f = lambda a, p, mu: p * (1 - np.exp(-a / mu))
        (p_hat, mu), _ = curve_fit(f, ag, rk, p0=(max(rk.max(), 1e-6), 0.2), bounds=([max(rk.max(), 1e-9), 0.02], [1.0, 5.0]), maxfev=2000)
        pi_fit = cl((p_hat - r_val) / max(1e-9, 1 - r_val))
    except Exception:
        pi_fit = pi_ref
    return pi_ref, pi_fit, True

'''
s=s.replace('"""[cq2:','"""[cq3 = cq2 + NOISE=delay2 (age-based stochastic label delay) + rules tpc/tpcf (temporal-reference contamination estimates); all cq2 rules unchanged]\n[cq2:',1)
s=s.replace("def main():",H+"def main():",1)
o="yp = y.copy(); pm = np.sort(tr[y[tr] == 1])[::-1] if (NOISE == 'delay' and s_['mode'] == 'temporal') else s_['perm']; k = int(round(lv * len(pm))); yp[pm[:k]] = 0"
n="""yp = y.copy()
            if NOISE == 'delay2' and s_['mode'] == 'temporal': yp[delay2_flip(tr, y, lv, s_['seed'])] = 0
            else:
                pm = np.sort(tr[y[tr] == 1])[::-1] if (NOISE == 'delay' and s_['mode'] == 'temporal') else s_['perm']; k = int(round(lv * len(pm))); yp[pm[:k]] = 0"""
assert o in s; s=s.replace(o,n,1)
o="pi_hat = cq_lib.estimate_pi(neg, posv)"
n=o+"""
                    if s_['mode'] == 'temporal': pi_tpc, pi_tpcf, guard_ok = tpc_estimates(yp[tr], cut, pi_fallback=pi_hat)
                    else: pi_tpc, pi_tpcf, guard_ok = pi_hat, pi_hat, False"""
assert o in s; s=s.replace(o,n,1)
o="thrs['cq'] = cq_lib.cq_threshold(neg, posv, alpha, pi_hat)[0]"
n=o+"""
                        thrs['tpc'] = cq_lib.cq_threshold(neg, posv, alpha, pi_tpc)[0]; thrs['tpcf'] = cq_lib.cq_threshold(neg, posv, alpha, pi_tpcf)[0]"""
assert o in s; s=s.replace(o,n,1)
o="pi_hat=pi_hat, pi_true"
n="pi_hat=pi_hat, pi_tpc=pi_tpc, pi_tpcf=pi_tpcf, guard_ok=guard_ok, pi_true"
assert o in s; s=s.replace(o,n,1)
open('code_sec/cq3_run.py','w').write(s)
