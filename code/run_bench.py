"""Benchmark: does imbalance handling help strong classifiers?  One dataset per call.
usage: python run_bench.py DATASET_NAME OUTDIR [N_REPEATS] [N_SPLITS]
"""
import sys, os, time, warnings, json
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
    matthews_corrcoef, brier_score_loss, confusion_matrix)
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
import imbalanced_databases as imbd

NJ = int(os.environ.get('NJOBS', '1'))
def load(name):
    d = getattr(imbd, 'load_' + name)()
    X = np.asarray(d['data'], dtype=float); y = np.asarray(d['target'])
    # minority = positive class
    cl, c = np.unique(y, return_counts=True)
    pos = cl[np.argmin(c)]
    y = (y == pos).astype(int)
    X = np.nan_to_num(X)
    return X, y

def make_clf(kind, seed, cw=False):
    if kind == 'LR':
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced' if cw else None))
    if kind == 'RF':
        return RandomForestClassifier(300, n_jobs=NJ, random_state=seed, class_weight='balanced_subsample' if cw else None)
    if kind == 'HGB':
        return HistGradientBoostingClassifier(random_state=seed, class_weight='balanced' if cw else None)
    if kind == 'XGB':
        return XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                             n_jobs=NJ, random_state=seed, verbosity=0, eval_metric='logloss')
STRATS = ['BASE', 'CW', 'ROS', 'RUS', 'SMOTE', 'BSMOTE', 'ADASYN', 'SMOTEENN', 'SMOTETOMEK']
def sampler(s, seed, ymin):
    k = max(1, min(5, ymin - 1))
    return {'ROS': lambda: RandomOverSampler(random_state=seed),
            'RUS': lambda: RandomUnderSampler(random_state=seed),
            'SMOTE': lambda: SMOTE(k_neighbors=k, random_state=seed),
            'BSMOTE': lambda: BorderlineSMOTE(k_neighbors=k, m_neighbors=max(2, min(10, ymin + 1)), random_state=seed),
            'ADASYN': lambda: ADASYN(n_neighbors=k, random_state=seed),
            'SMOTEENN': lambda: SMOTEENN(smote=SMOTE(k_neighbors=k, random_state=seed), random_state=seed),
            'SMOTETOMEK': lambda: SMOTETomek(smote=SMOTE(k_neighbors=k, random_state=seed), random_state=seed)}[s]()

def fit_predict(kind, strat, Xtr, ytr, Xte, seed):
    """fit with strategy on (Xtr,ytr); return P(minority) on Xte; falls back to BASE if a sampler fails"""
    cw = strat == 'CW'
    Xs, ys = Xtr, ytr
    status = 'ok'
    if strat not in ('BASE', 'CW'):
        try:
            Xs, ys = sampler(strat, seed, int(ytr.sum())).fit_resample(Xtr, ytr)
        except Exception as e:
            Xs, ys = Xtr, ytr; status = 'fallback'
    if kind == 'XGB' and cw:
        clf = make_clf(kind, seed); clf.set_params(scale_pos_weight=(len(ytr) - ytr.sum()) / max(1, ytr.sum()))
    else:
        clf = make_clf(kind, seed, cw)
    clf.fit(Xs, ys)
    return clf.predict_proba(Xte)[:, 1], status

def best_thr(y, p):
    ts = np.unique(np.quantile(p, np.linspace(0.02, 0.98, 97)))
    f = [f1_score(y, p >= t, zero_division=0) for t in ts]
    return float(ts[int(np.argmax(f))]) if len(ts) else 0.5

def ece(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[0] -= 1e-9; q[-1] += 1e-9
    idx = np.clip(np.digitize(p, q[1:-1]), 0, bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e

def metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    N = len(y)
    out = dict(AP=average_precision_score(y, p), AUC=roc_auc_score(y, p), F1=f1_score(y, yh, zero_division=0),
               MCC=matthews_corrcoef(y, yh), Brier=brier_score_loss(y, p), ECE=ece(y, p),
               Recall=tp / max(1, tp + fn), Prec=tp / max(1, tp + fp))
    for c in (1, 5, 20):
        out['Cost%d' % c] = (c * fn + fp) / N
    return out

def main():
    name, outdir = sys.argv[1], sys.argv[2]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    nspl = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    kinds = os.environ.get('KINDS', 'LR,RF,HGB,XGB').split(',')
    os.makedirs(outdir, exist_ok=True)
    X, y = load(name)
    rskf = RepeatedStratifiedKFold(n_splits=nspl, n_repeats=reps, random_state=20261124)
    rows = []; t0 = time.time()
    for fi, (tr, te) in enumerate(rskf.split(X, y)):
        seed = 1000 + fi
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        # inner split for threshold selection
        itr, iva = train_test_split(np.arange(len(tr)), test_size=0.25, stratify=ytr, random_state=seed)
        for kind in kinds:
            for s in STRATS:
                pv, st1 = fit_predict(kind, s, Xtr[itr], ytr[itr], Xtr[iva], seed)
                thr = best_thr(ytr[iva], pv)
                pt, st2 = fit_predict(kind, s, Xtr, ytr, Xte, seed)
                for mode, th in (('t05', 0.5), ('tuned', thr)):
                    # resampled models are calibrated to a balanced prior; t05 is the default decision rule
                    m = metrics(yte, pt, th); m.update(dataset=name, fold=fi, learner=kind, strategy=s, thr_mode=mode,
                                                       thr=th, status=st2, n_test=len(te), n_pos_test=int(yte.sum()))
                    rows.append(m)
    df = pd.DataFrame(rows); df['secs'] = time.time() - t0
    df.to_csv(os.path.join(outdir, name + '.csv'), index=False)
    print(name, 'done', round(time.time() - t0, 1), 's', len(df), 'rows', flush=True)
if __name__ == '__main__':
    main()
