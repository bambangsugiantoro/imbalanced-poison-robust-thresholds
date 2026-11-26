"""[cq3 = cq2 + NOISE=delay2 (age-based stochastic label delay) + rules tpc/tpcf (temporal-reference contamination estimates); all cq2 rules unchanged]
[cq2: adds baselines conf/npu/cl/trust and NOISE=delay; STRATS may be all nine]
Contamination-corrected quantile (CQ) vs naive/oracle/cap thresholds, one-sided label flipping.
usage: python cq_run.py SOURCE OUTDIR [reps] [nsplits]
SOURCE = b49:NAME | oml:DID:TIMECOL:NAME | unsw:unsw | cic:cic | ell:ell     (run from ~/imb_bench; imports code/run_bench.py, code_sec/sec_bench.py, code_sec/sec_unsw_cap.py, code_sec/cq_lib.py)
env: KINDS (LR,RF,XGB) STRATS (BASE,CW) LEVELS (0,0.2,0.4) QS (e.g. 0.99:fpr1,0.999:fpr01; default 0.95:fpr5,0.99:fpr1 for b49, 0.99:fpr1,0.999:fpr01 for temporal) CAP
Rules are fixed in cq_lib.py and cap rules (cap25, gcap25 = gate val AUC>=0.95) are kept only as reference."""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
for p in ('code', 'code_sec'): sys.path.insert(0, os.path.join(os.getcwd(), p))
import run_bench as rb
import sec_bench as sb
import cq_lib
from scipy.stats import binom
NOISE = os.environ.get('NOISE', 'random'); DELTA = 0.05; TRUST_FRAC = float(os.environ.get('TRUST_FRAC', '0.05'))

def npu_thr(neg, alpha, delta=DELTA):
    # Neyman-Pearson umbrella: smallest order statistic k with P(FPR>alpha) <= delta (Tong et al. 2018; Yao et al. 2021) on labelled negatives
    n = len(neg); s = np.sort(neg)
    for k in range(1, n + 1):
        if 1 - binom.cdf(k - 1, n, 1 - alpha) <= delta: return float(s[k - 1])   # P(FPR(s_k)>alpha)=P(Bin(n,1-alpha)>=k)
    return float(s[-1])
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

def splits_b49(name, reps, nspl):
    X, y = rb.load(name); y = np.asarray(y)
    out = []
    for fi, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=nspl, n_repeats=reps, random_state=20261125).split(X, y)):
        out.append(dict(tag=fi, tr=tr, te=te, perm=np.random.RandomState(9000 + fi).permutation(tr[y[tr] == 1]), seed=1000 + fi, mode='cv'))
    return X, y, out

def splits_temporal(X, y):
    n = len(y); out = []
    for wi in range(len(sb.EDGES) - 1):
        a, b = int(sb.EDGES[wi] * n), int(sb.EDGES[wi + 1] * n)
        tr = sb.cap_train(np.arange(a), y, 100 + wi); te = np.arange(a, b)
        if y[tr].sum() < 20 or y[te].sum() < 5: continue
        out.append(dict(tag=wi, tr=tr, te=te, perm=np.random.RandomState(7000 + wi).permutation(tr[y[tr] == 1]), seed=1000 + wi, mode='temporal'))
    return X, y, out


def load_cic(path='data_new/cic_45277.pq'):
    """CIC-IDS-2017 (OpenML 45277, parquet copy). Order by Timestamp; drop identifiers and label-bearing columns; positive = not Benign."""
    import pandas as pd
    d = pd.read_parquet(path)
    t = pd.to_datetime(d['Timestamp'], format='mixed', dayfirst=True, errors='coerce')  # d/m/Y, 3-7 July 2017; rows with missing Timestamp are dropped
    ben = d['Benign'].astype(str).str.strip().str.lower()
    yb = (~ben.isin(['benign', 'true', '1', '1.0'])).astype(int).values
    drop = ['Flow_ID', 'Source_IP', 'Destination_IP', 'Timestamp', 'Benign', 'Sub_Cat', 'Cat']
    X = d.drop(columns=[c for c in drop if c in d.columns])
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]): X[c] = X[c].astype('category').cat.codes.astype(float).replace(-1, np.nan)
    X = X.astype('float32').replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).fillna(0).values
    o = np.argsort(t.values, kind='stable'); o = o[~t.isna().values[o]]
    return X[o], yb[o], 'cic2017'

def load_elliptic(path='data_new/elliptic_labeled.csv.gz'):
    """Elliptic bitcoin: col0=label (1 illicit), col1=txId, col2=time step, then features. Order by time step."""
    import pandas as pd
    d = pd.read_csv(path, header=None)
    o = np.argsort(d[2].values, kind='stable')
    X = d.iloc[:, 3:].astype('float32').fillna(0).values; y = d[0].astype(int).values
    return X[o], y[o], 'elliptic'


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

def main():
    src, outdir = sys.argv[1], sys.argv[2]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    nspl = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    kinds = os.environ.get('KINDS', 'LR,RF,XGB').split(','); strats = os.environ.get('STRATS', 'BASE,CW').split(',')
    levels = [float(x) for x in os.environ.get('LEVELS', '0,0.2,0.4').split(',')]
    kind0 = src.split(':')[0]
    if kind0 == 'b49': name = src.split(':')[1]; X, y, sp = splits_b49(name, reps, nspl); qs_def = '0.95:fpr5,0.99:fpr1'
    elif kind0 == 'oml':
        _, did, tcol, name = src.split(':'); X, y, _ = sb.load_openml(did, tcol); X, y, sp = splits_temporal(X, np.asarray(y)); qs_def = '0.99:fpr1,0.999:fpr01'
    elif kind0 == 'cic':
        name = 'cic2017'; X, y, _ = load_cic(); X, y, sp = splits_temporal(X, y); qs_def = '0.99:fpr1,0.999:fpr01'
    elif kind0 == 'ell':
        name = 'elliptic'; X, y, _ = load_elliptic(); X, y, sp = splits_temporal(X, y); qs_def = '0.99:fpr1,0.999:fpr01'
    elif kind0 == 'unsw':
        import sec_unsw_cap as su; name = 'unsw'; X, y, _ = su.load_unsw(); X, y, sp = splits_temporal(X, y); qs_def = '0.99:fpr1,0.999:fpr01'
    else: raise SystemExit('bad source')
    QS = [(float(a), b) for a, b in (s.split(':') for s in os.environ.get('QS', qs_def).split(','))]
    os.makedirs(outdir, exist_ok=True); rows = []; t0 = time.time()
    for s_ in sp:
        tr, te = s_['tr'], s_['te']
        for lv in levels:
            yp = y.copy()
            if NOISE == 'delay2' and s_['mode'] == 'temporal': yp[delay2_flip(tr, y, lv, s_['seed'])] = 0
            else:
                pm = np.sort(tr[y[tr] == 1])[::-1] if (NOISE == 'delay' and s_['mode'] == 'temporal') else s_['perm']; k = int(round(lv * len(pm))); yp[pm[:k]] = 0
            if s_['mode'] == 'cv':
                try: itr, iva = train_test_split(tr, test_size=0.25, stratify=yp[tr], random_state=s_['tag'])
                except Exception: continue
                if yp[itr].sum() < 2 or yp[iva].sum() < 1 or (yp[iva] == 0).sum() < 10: continue
            else:
                cut = int(0.75 * len(tr)); itr, iva = tr[:cut], tr[cut:]
                if yp[itr].sum() < 10 or yp[iva].sum() < 3: itr, iva = train_test_split(tr, test_size=0.25, stratify=yp[tr], random_state=s_['tag'])
            n_flip_val = int(((y[iva] == 1) & (yp[iva] == 0)).sum()); n_negobs = int((yp[iva] == 0).sum())
            for kind in kinds:
                for s in strats:
                    pv, _ = rb.fit_predict(kind, s, X[itr], yp[itr], X[iva], s_['seed'])
                    pt, st = rb.fit_predict(kind, s, X[tr], yp[tr], X[te], s_['seed'])
                    yv = yp[iva]; neg = pv[yv == 0]; posv = pv[yv == 1]
                    auc = roc_auc_score(yv, pv) if 0 < yv.sum() < len(yv) else 0.5
                    pi_hat = cq_lib.estimate_pi(neg, posv)
                    if s_['mode'] == 'temporal': pi_tpc, pi_tpcf, guard_ok = tpc_estimates(yp[tr], cut, pi_fallback=pi_hat)
                    else: pi_tpc, pi_tpcf, guard_ok = pi_hat, pi_hat, False
                    negt = pt[y[te] == 0]
                    for q, qn in QS:
                        alpha = 1 - q; thrs = {}
                        thrs['naive'] = sb.fpr_thr(yv, pv, q); thrs['oracle'] = sb.fpr_thr(y[iva], pv, q)
                        thrs['cq'] = cq_lib.cq_threshold(neg, posv, alpha, pi_hat)[0]
                        thrs['tpc'] = cq_lib.cq_threshold(neg, posv, alpha, pi_tpc)[0]; thrs['tpcf'] = cq_lib.cq_threshold(neg, posv, alpha, pi_tpcf)[0]
                        if len(posv) >= 3:
                            c = float(np.quantile(posv, 0.25)); order = np.sort(neg)[::-1]
                            kept = order[min(int((neg >= c).sum()), len(posv)):]
                            th = float(np.quantile(kept, q)) if len(kept) > 50 else thrs['naive']
                            thrs['cap25'] = th; thrs['gcap25'] = th if auc >= 0.95 else thrs['naive']
                        else: thrs['cap25'] = thrs['gcap25'] = thrs['naive']
                        n_ = len(neg); lvl = min(1.0, np.ceil((n_ + 1) * q) / max(n_, 1)); thrs['conf'] = float(np.quantile(neg, lvl)) if n_ else thrs['naive']
                        thrs['npu'] = npu_thr(neg, alpha) if n_ > 20 else thrs['naive']
                        if len(posv) >= 3:
                            kept_cl = neg[neg < float(np.mean(posv))]; thrs['cl'] = float(np.quantile(kept_cl, q)) if len(kept_cl) > 50 else thrs['naive']
                        else: thrs['cl'] = thrs['naive']
                        rs = np.random.RandomState(s_['seed'] + 77); tsel = rs.rand(len(iva)) < TRUST_FRAC; tneg = pv[tsel & (y[iva] == 0)]
                        thrs['trust'] = float(np.quantile(tneg, q)) if len(tneg) >= 20 else thrs['naive']
                        thrs['ideal'] = float(np.quantile(negt, q)) if len(negt) else thrs['naive']     # test-negative quantile: unattainable reference
                        for var, th in thrs.items():
                            m = sb.sec_metrics(y[te], pt, th)
                            m.update(noise=NOISE, dataset=name, split=s_['tag'], poison=lv, learner=kind, strategy=s, var=var, op=qn, target=alpha, thr=th, val_auc=auc,
                                     pi_hat=pi_hat, pi_tpc=pi_tpc, pi_tpcf=pi_tpcf, guard_ok=guard_ok, pi_true=(n_flip_val / n_negobs if n_negobs else 0.0), n_neg_val=n_negobs, n_pos_val=len(posv), status=st, n_pos_test=int(y[te].sum()), n_neg_test=len(negt))
                            rows.append(m)
            print(name, 'split', s_['tag'], 'poison', lv, round(time.time() - t0), 's', flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(outdir, name + '.partial.csv'), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, name + '.csv'), index=False)
    print(name, 'done', round(time.time() - t0), 's', len(rows), 'rows', flush=True)
if __name__ == '__main__':
    main()
