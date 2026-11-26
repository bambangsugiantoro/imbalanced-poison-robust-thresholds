"""Contamination-corrected quantile (CQ) thresholds for a fixed false-positive-rate budget under one-sided label flipping (attack -> benign).
Validation scores pv with (possibly corrupted) labels yv. Labeled positives are assumed to be true positives (flips only go positive -> negative),
so the labeled-negative score distribution is a mixture: F_obs = (1-pi) F0 + pi F1, with F1 estimated by the labeled positives.
estimate_pi: tail-ratio estimator pi_hat = min_t S_obs(t)/S1(t) over t in the upper quantiles of labeled positives (upper bound of pi, tight when F0 has no mass there).
cq_threshold: smallest t whose corrected negative tail (S_obs - pi S1)/(1-pi) is <= alpha (monotone-enforced)."""
import numpy as np

def estimate_pi(neg, pos, qgrid=(0.5, 0.6, 0.7, 0.8, 0.9), min_pos_tail=5, min_neg_tail=3):
    neg = np.asarray(neg); pos = np.asarray(pos)
    if len(pos) < 2 * min_pos_tail or len(neg) < 10: return 0.0
    best = 1.0
    for q in qgrid:
        t = np.quantile(pos, q)
        s1 = float((pos >= t).mean()); n_above = int((neg >= t).sum())
        if (pos >= t).sum() < min_pos_tail: continue
        r = (n_above / len(neg)) / s1
        best = min(best, r)
    return float(min(max(best, 0.0), 0.9))

def cq_threshold(neg, pos, alpha, pi=None):
    neg = np.asarray(neg); pos = np.asarray(pos)
    if pi is None: pi = estimate_pi(neg, pos)
    if pi < 1e-3 or len(pos) == 0:
        return float(np.quantile(neg, 1 - alpha)), pi
    cand = np.unique(neg)[::-1]                      # descending
    s_obs = np.array([(neg >= t).mean() for t in cand]); s1 = np.array([(pos >= t).mean() for t in cand])
    est = (s_obs - pi * s1) / (1 - pi)
    est = np.maximum.accumulate(est)                 # enforce monotone as t decreases
    ok = np.flatnonzero(est <= alpha)
    if len(ok) == 0: return float(cand[0]), pi        # even the top candidate exceeds alpha: most conservative
    return float(cand[ok[-1]]), pi                   # smallest t with est<=alpha
