"""Poison-robust fixed-FPR thresholds on the 49 benchmark datasets (random stratified CV, label-flip poisoning).
usage: python bench49_poison.py NAME OUTDIR [reps] [nsplits]   (run from ~/imb_bench; imports code/run_bench.py, code_sec/sec_bench.py)
Pre-specified rules: cut = median (cap50) or q25 (cap25) of validation-positive scores; remove negatives >= cut,
at most n_pos_val of them; g* = apply only if validation AUC >= 0.95 else naive. Operating points FPR 5% and 1%."""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.getcwd(), 'code')); sys.path.insert(0, os.path.join(os.getcwd(), 'code_sec'))
import run_bench as rb
import sec_bench as sb
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

def main():
    name, outdir = sys.argv[1], sys.argv[2]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    nspl = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    kinds = os.environ.get('KINDS', 'LR,RF,XGB').split(','); strats = os.environ.get('STRATS', 'BASE,CW').split(',')
    levels = [float(x) for x in os.environ.get('LEVELS', '0,0.2,0.4').split(',')]
    os.makedirs(outdir, exist_ok=True)
    X, y = rb.load(name); y = np.asarray(y)
    rskf = RepeatedStratifiedKFold(n_splits=nspl, n_repeats=reps, random_state=20261125)
    rows = []; t0 = time.time()
    for fi, (tr, te) in enumerate(rskf.split(X, y)):
        pos = tr[y[tr] == 1]; perm = np.random.RandomState(9000 + fi).permutation(pos)
        for lv in levels:
            yp = y.copy(); k = int(round(lv * len(pos))); yp[perm[:k]] = 0
            try: itr, iva = train_test_split(tr, test_size=0.25, stratify=yp[tr], random_state=fi)
            except Exception: continue
            if yp[itr].sum() < 2 or yp[iva].sum() < 1 or (yp[iva] == 0).sum() < 10: continue
            seed = 1000 + fi
            for kind in kinds:
                for s in strats:
                    pv, _ = rb.fit_predict(kind, s, X[itr], yp[itr], X[iva], seed)
                    pt, st = rb.fit_predict(kind, s, X[tr], yp[tr], X[te], seed)
                    yv = yp[iva]; neg = pv[yv == 0]; posv = pv[yv == 1]
                    auc = roc_auc_score(yv, pv) if 0 < yv.sum() < len(yv) else 0.5
                    order = np.sort(neg)[::-1]; thrs = {}
                    for qn, q in (('fpr5', 0.95), ('fpr1', 0.99)):
                        thrs['naive_' + qn] = sb.fpr_thr(yv, pv, q)
                        thrs['oracle_' + qn] = sb.fpr_thr(y[iva], pv, q)
                        for cn, c in (('cap50', float(np.median(posv)) if len(posv) >= 1 else None),
                                      ('cap25', float(np.quantile(posv, 0.25)) if len(posv) >= 1 else None)):
                            if c is None:
                                thrs[cn + '_' + qn] = thrs['naive_' + qn]; thrs['g' + cn + '_' + qn] = thrs['naive_' + qn]; continue
                            ntrim = min(int((neg >= c).sum()), len(posv))
                            kept = order[ntrim:]
                            th = float(np.quantile(kept, q)) if len(kept) >= 10 else thrs['naive_' + qn]
                            thrs[cn + '_' + qn] = th
                            thrs['g' + cn + '_' + qn] = th if auc >= 0.95 else thrs['naive_' + qn]
                    for mode, th in thrs.items():
                        m = sb.sec_metrics(y[te], pt, th)
                        m.update(val_auc=auc, dataset=name, fold=fi, poison=lv, learner=kind, strategy=s, thr_mode=mode, thr=th,
                                 status=st, n_pos_train_poisoned=int(yp[tr].sum()), n_pos_test=int(y[te].sum()))
                        rows.append(m)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(outdir, name + '.csv'), index=False)
    print(name, 'done', round(time.time() - t0, 1), 's', len(df), 'rows', flush=True)
if __name__ == '__main__':
    main()
