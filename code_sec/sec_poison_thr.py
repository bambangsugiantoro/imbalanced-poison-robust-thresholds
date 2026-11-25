"""Poison-robust FPR thresholds. Label-flipping poisoning (attack -> benign) on the training pool; clean temporal test windows.
usage: python sec_poison.py DID TIMECOL OUTDIR NAME
env: CAP, NJOBS, KINDS (default LR,RF,XGB), STRATS, LEVELS (default 0,0.2,0.4)
Flip order is a fixed random permutation of training positives per window, so 40% contains the 20% set."""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_bench as rb
import sec_bench as sb

def main():
    did, timecol, outdir, name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    kinds = os.environ.get('KINDS', 'LR,RF,XGB').split(',')
    strats = os.environ.get('STRATS', 'BASE,CW').split(',')
    levels = [float(x) for x in os.environ.get('LEVELS', '0,0.2,0.4').split(',')]
    os.makedirs(outdir, exist_ok=True)
    X, y, _ = sb.load_openml(did, timecol)
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
                    cuts = {}
                    if len(posv) >= 3:
                        cuts['trim50'] = float(np.median(posv)); cuts['trim25'] = float(np.quantile(posv, 0.25))
                    thrs = {}
                    for qn, q in (('fpr1', 0.99), ('fpr01', 0.999)):
                        thrs['naive_' + qn] = sb.fpr_thr(yv, pv, q)
                        thrs['oracle_' + qn] = sb.fpr_thr(y[iva], pv, q)          # clean validation labels (upper bound)
                        for cn, c in cuts.items():
                            kept = neg[neg < c]
                            thrs[cn + '_' + qn] = float(np.quantile(kept, q)) if len(kept) > 50 else thrs['naive_' + qn]
                    for mode, th in thrs.items():
                        m = sb.sec_metrics(y[te], pt, th)       # evaluate against CLEAN labels
                        m.update(dataset=name, window=wi, poison=lv, learner=kind, strategy=s, thr_mode=mode, thr=th,
                                 status=st, n_pos_train_poisoned=int(yp[tr].sum()), n_pos_test=int(y[te].sum()))
                        rows.append(m)
            print(name, 'window', wi, 'poison', lv, round(time.time() - t0), 's', flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(outdir, name + '.partial.csv'), index=False)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(outdir, name + '.csv'), index=False)
    print(name, 'done', round(time.time() - t0), 's', len(df), 'rows', flush=True)
if __name__ == '__main__':
    main()
