"""Contamination-corrected quantile (CQ) vs naive/oracle/cap thresholds, one-sided label flipping.
usage: python cq_run.py SOURCE OUTDIR [reps] [nsplits]
SOURCE = b49:NAME | oml:DID:TIMECOL:NAME | unsw:unsw     (run from ~/imb_bench; imports code/run_bench.py, code_sec/sec_bench.py, code_sec/sec_unsw_cap.py, code_sec/cq_lib.py)
env: KINDS (LR,RF,XGB) STRATS (BASE,CW) LEVELS (0,0.2,0.4) QS (e.g. 0.99:fpr1,0.999:fpr01; default 0.95:fpr5,0.99:fpr1 for b49, 0.99:fpr1,0.999:fpr01 for temporal) CAP
Rules are fixed in cq_lib.py and cap rules (cap25, gcap25 = gate val AUC>=0.95) are kept only as reference."""
import sys, os, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
for p in ('code', 'code_sec'): sys.path.insert(0, os.path.join(os.getcwd(), p))
import run_bench as rb
import sec_bench as sb
import cq_lib
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
    elif kind0 == 'unsw':
        import sec_unsw_cap as su; name = 'unsw'; X, y, _ = su.load_unsw(); X, y, sp = splits_temporal(X, y); qs_def = '0.99:fpr1,0.999:fpr01'
    else: raise SystemExit('bad source')
    QS = [(float(a), b) for a, b in (s.split(':') for s in os.environ.get('QS', qs_def).split(','))]
    os.makedirs(outdir, exist_ok=True); rows = []; t0 = time.time()
    for s_ in sp:
        tr, te = s_['tr'], s_['te']
        for lv in levels:
            yp = y.copy(); k = int(round(lv * len(s_['perm']))); yp[s_['perm'][:k]] = 0
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
                    negt = pt[y[te] == 0]
                    for q, qn in QS:
                        alpha = 1 - q; thrs = {}
                        thrs['naive'] = sb.fpr_thr(yv, pv, q); thrs['oracle'] = sb.fpr_thr(y[iva], pv, q)
                        thrs['cq'] = cq_lib.cq_threshold(neg, posv, alpha, pi_hat)[0]
                        if len(posv) >= 3:
                            c = float(np.quantile(posv, 0.25)); order = np.sort(neg)[::-1]
                            kept = order[min(int((neg >= c).sum()), len(posv)):]
                            th = float(np.quantile(kept, q)) if len(kept) > 50 else thrs['naive']
                            thrs['cap25'] = th; thrs['gcap25'] = th if auc >= 0.95 else thrs['naive']
                        else: thrs['cap25'] = thrs['gcap25'] = thrs['naive']
                        thrs['ideal'] = float(np.quantile(negt, q)) if len(negt) else thrs['naive']     # test-negative quantile: unattainable reference
                        for var, th in thrs.items():
                            m = sb.sec_metrics(y[te], pt, th)
                            m.update(dataset=name, split=s_['tag'], poison=lv, learner=kind, strategy=s, var=var, op=qn, target=alpha, thr=th, val_auc=auc,
                                     pi_hat=pi_hat, pi_true=(n_flip_val / n_negobs if n_negobs else 0.0), n_neg_val=n_negobs, n_pos_val=len(posv), status=st, n_pos_test=int(y[te].sum()), n_neg_test=len(negt))
                            rows.append(m)
            print(name, 'split', s_['tag'], 'poison', lv, round(time.time() - t0), 's', flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(outdir, name + '.partial.csv'), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, name + '.csv'), index=False)
    print(name, 'done', round(time.time() - t0), 's', len(rows), 'rows', flush=True)
if __name__ == '__main__':
    main()
