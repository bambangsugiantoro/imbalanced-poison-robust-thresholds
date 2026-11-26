"""Capped/gated poison-robust FPR thresholds (pre-specified: cut=median or q25 of validation positives; at most n_pos_val negatives removed; gate val AUC>=0.95). Label-flipping poisoning (attack -> benign) on the training pool; clean temporal test windows.
usage: python sec_unsw_cap.py DID TIMECOL OUTDIR NAME   (DID and TIMECOL are ignored; data is read from data_unsw/)
env: CAP, NJOBS, KINDS (default LR,RF,XGB), STRATS, LEVELS (default 0,0.2,0.4)
Flip order is a fixed random permutation of training positives per window, so 40% contains the 20% set."""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_bench as rb
import sec_bench as sb

def load_unsw():
    """UNSW-NB15 raw CSVs (4 files, no header) ordered by Stime; minority = attack (Label=1). Needs data_unsw/ in the working dir."""
    f = pd.read_csv('data_unsw/NUSW-NB15_features.csv', encoding='latin1'); names = [str(x).strip() for x in f.iloc[:, 1]]
    d = pd.concat([pd.read_csv('data_unsw/UNSW-NB15_%d.csv' % i, header=None, names=names, low_memory=False, encoding='latin1') for i in range(1, 5)], ignore_index=True)
    y = pd.to_numeric(d['Label'], errors='coerce').values; t = pd.to_numeric(d['Stime'], errors='coerce').values
    X = d.drop(columns=['srcip', 'dstip', 'attack_cat', 'Label', 'Stime', 'Ltime'])
    for c in X.columns:
        if c in ('sport', 'dsport'): X[c] = pd.to_numeric(X[c], errors='coerce')
        elif X[c].dtype == 'O': X[c] = X[c].astype('category').cat.codes.astype(float).replace(-1, np.nan)
        else: X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.astype(float); X = X.fillna(X.median()).fillna(0).values
    o = np.argsort(t, kind='stable'); o = o[~np.isnan(t[o]) & ~np.isnan(y[o])]
    return X[o], y[o].astype(int), 'unsw'

def main():
    did, timecol, outdir, name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    kinds = os.environ.get('KINDS', 'LR,RF,XGB').split(',')
    strats = os.environ.get('STRATS', 'BASE,CW').split(',')
    levels = [float(x) for x in os.environ.get('LEVELS', '0,0.2,0.4').split(',')]
    os.makedirs(outdir, exist_ok=True)
    X, y, _ = load_unsw()
    n = len(y); rows = []; t0 = time.time()
    for wi in range(len(sb.EDGES) - 1):
        a, b = int(sb.EDGES[wi] * n), int(sb.EDGES[wi + 1] * n)
        tr = sb.cap_train(np.arange(a), y, 100 + wi); te = np.arange(a, b)
        if y[tr].sum() < 20 or y[te].sum() < 5: continue
        pos = tr[y[tr] == 1]; perm = np.random.RandomState(7000 + wi).permutation(pos)
        for lv in levels:
            yp = y.copy(); k = int(round(lv * len(pos))); yp[perm[:k]] = 0
            cut = int(0.75 * len(tr)); itr, iva = tr[:cut], tr[cut:]
            if yp[itr].sum() < 10 or yp[iva].sum() < 3:
                from sklearn.model_selection import train_test_split
                itr, iva = train_test_split(tr, test_size=0.25, stratify=yp[tr], random_state=wi)
            seed = 1000 + wi
            for kind in kinds:
                for s in strats:
                    pv, _ = rb.fit_predict(kind, s, X[itr], yp[itr], X[iva], seed)
                    pt, st = rb.fit_predict(kind, s, X[tr], yp[tr], X[te], seed)
                    yv = yp[iva]; neg = pv[yv == 0]; posv = pv[yv == 1]
                    from sklearn.metrics import roc_auc_score
                    auc = roc_auc_score(yv, pv) if 0 < yv.sum() < len(yv) else 0.5
                    thrs = {}
                    order = np.sort(neg)[::-1]
                    for qn, q in (('fpr1', 0.99), ('fpr01', 0.999)):
                        thrs['naive_' + qn] = sb.fpr_thr(yv, pv, q)
                        thrs['oracle_' + qn] = sb.fpr_thr(y[iva], pv, q)          # clean validation labels (upper bound)
                        for cn, c in (('cap50', float(np.median(posv)) if len(posv) >= 3 else None),
                                      ('cap25', float(np.quantile(posv, 0.25)) if len(posv) >= 3 else None)):
                            if c is None: thrs[cn + '_' + qn] = thrs['naive_' + qn]; thrs['g' + cn + '_' + qn] = thrs['naive_' + qn]; continue
                            # trim negatives scoring >= c, but never more than the number of remaining validation positives
                            ntrim = min(int((neg >= c).sum()), len(posv))
                            kept = order[ntrim:]
                            th = float(np.quantile(kept, q)) if len(kept) > 50 else thrs['naive_' + qn]
                            thrs[cn + '_' + qn] = th
                            thrs['g' + cn + '_' + qn] = th if auc >= 0.95 else thrs['naive_' + qn]   # gate: only if validation AUC >= 0.95
                    for mode, th in thrs.items():
                        m = sb.sec_metrics(y[te], pt, th)       # evaluate against CLEAN labels
                        m.update(val_auc=auc, dataset=name, window=wi, poison=lv, learner=kind, strategy=s, thr_mode=mode, thr=th,
                                 status=st, n_pos_train_poisoned=int(yp[tr].sum()), n_pos_test=int(y[te].sum()))
                        rows.append(m)
            print(name, 'window', wi, 'poison', lv, round(time.time() - t0), 's', flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(outdir, name + '.partial.csv'), index=False)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(outdir, name + '.csv'), index=False)
    print(name, 'done', round(time.time() - t0), 's', len(df), 'rows', flush=True)
if __name__ == '__main__':
    main()
