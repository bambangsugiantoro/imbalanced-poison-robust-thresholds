"""Security benchmark: temporal rolling-origin evaluation, fixed-FPR operating points.
usage: python sec_bench.py DID TIMECOL OUTDIR [NAME]
env: CAP (majority cap in train, default 60000), NJOBS, KINDS, STRATS
"""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_bench as rb
import openml
from sklearn.metrics import confusion_matrix

CAP = int(os.environ.get('CAP', '60000'))
EDGES = [0.5, 0.625, 0.75, 0.875, 1.0]

def load_openml(did, timecol):
    d = openml.datasets.get_dataset(int(did), download_qualities=False, download_features_meta_data=False)
    X, y, cat, names = d.get_data(target=d.default_target_attribute)
    t = pd.to_numeric(X[timecol], errors='coerce').values
    X = X.drop(columns=[timecol])
    for c in X.columns:
        if X[c].dtype == 'O' or str(X[c].dtype) == 'category':
            X[c] = X[c].astype('category').cat.codes.astype(float).replace(-1, np.nan)
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.astype(float)
    X = X.fillna(X.median()).fillna(0).values
    vc = pd.Series(y.astype(str)).value_counts()
    yb = (y.astype(str).values == vc.index[-1]).astype(int)   # minority = positive
    o = np.argsort(t, kind='stable'); o = o[~np.isnan(t[o])]
    return X[o], yb[o], d.name

def best_thr_sec(y, p):
    """F1-optimal threshold; grid covers the extreme upper tail needed at prevalence << 2%."""
    qs = np.concatenate([np.linspace(0.02, 0.98, 97), [0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999]])
    ts = np.unique(np.quantile(p, qs))
    f = [rb.f1_score(y, p >= t, zero_division=0) for t in ts]
    f = np.array(f); best = np.flatnonzero(f == f.max())
    return float(ts[best[-1]])   # ties -> higher threshold (fewer alerts)

def fpr_thr(yv, pv, q):
    neg = pv[yv == 0]
    return float(np.quantile(neg, q)) if len(neg) else 0.5

def sec_metrics(y, p, thr):
    m = rb.metrics(y, p, thr)
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1]).ravel()
    m['FPR'] = fp / max(1, fp + tn); m['alerts_per_10k_benign'] = m['FPR'] * 1e4
    return m

def cap_train(idx, y, seed):
    pos = idx[y[idx] == 1]; neg = idx[y[idx] == 0]
    if len(neg) > CAP:
        neg = np.random.RandomState(seed).choice(neg, CAP, replace=False)
    return np.sort(np.concatenate([pos, neg]))

def main():
    did, timecol, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    kinds = os.environ.get('KINDS', 'LR,RF,HGB,XGB').split(',')
    strats = os.environ.get('STRATS', ','.join(rb.STRATS)).split(',')
    os.makedirs(outdir, exist_ok=True)
    X, y, name = load_openml(did, timecol)
    name = (sys.argv[4] if len(sys.argv) > 4 else name)
    n = len(y); rows = []; t0 = time.time()
    print(name, X.shape, 'pos', int(y.sum()), flush=True)
    for wi in range(len(EDGES) - 1):
        a, b = int(EDGES[wi] * n), int(EDGES[wi + 1] * n)
        tr = cap_train(np.arange(a), y, 100 + wi); te = np.arange(a, b)
        if y[tr].sum() < 20 or y[te].sum() < 5:
            print('skip window', wi, int(y[tr].sum()), int(y[te].sum()), flush=True); continue
        cut = int(0.75 * len(tr)); itr, iva = tr[:cut], tr[cut:]
        if y[itr].sum() < 10 or y[iva].sum() < 5:
            from sklearn.model_selection import train_test_split
            itr, iva = train_test_split(tr, test_size=0.25, stratify=y[tr], random_state=wi)
        seed = 1000 + wi
        for kind in kinds:
            for s in strats:
                pv, _ = rb.fit_predict(kind, s, X[itr], y[itr], X[iva], seed)
                thrs = {'t05': 0.5, 'tuned': best_thr_sec(y[iva], pv),
                        'fpr1': fpr_thr(y[iva], pv, 0.99), 'fpr01': fpr_thr(y[iva], pv, 0.999)}
                pt, st = rb.fit_predict(kind, s, X[tr], y[tr], X[te], seed)
                for mode, th in thrs.items():
                    m = sec_metrics(y[te], pt, th)
                    m.update(dataset=name, window=wi, learner=kind, strategy=s, thr_mode=mode, thr=th, status=st,
                             n_train=len(tr), n_pos_train=int(y[tr].sum()), n_test=len(te), n_pos_test=int(y[te].sum()))
                    rows.append(m)
            print(name, 'window', wi, kind, round(time.time() - t0), 's', flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(outdir, name + '.partial.csv'), index=False)
    df = pd.DataFrame(rows); df['secs'] = time.time() - t0
    df.to_csv(os.path.join(outdir, name + '.csv'), index=False)
    print(name, 'done', round(time.time() - t0), 's', len(df), 'rows', flush=True)
if __name__ == '__main__':
    main()
