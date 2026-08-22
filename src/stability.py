"""Vintage stability and confidence intervals.

Two things a scorecard needs before anyone acts on a comparison between models,
both of which were missing while this project reported point estimates only.

VINTAGE PSI. A scorecard is fit on one population and scored on next quarter's.
Population Stability Index per vintage, on the score and on every characteristic,
with the standard bands. The bands are stated because an unlabelled PSI is
meaningless: <0.10 stable, 0.10-0.25 monitor, >0.25 investigate.

CONFIDENCE INTERVALS. The champion/challenger recommendation in run_scorecard.py
rested on a +0.009 Gini difference and a +1.5pp swap-in default gap. Neither came
with an interval, so neither could be distinguished from noise -- and the README
said as much in words while the table still printed bare numbers. This computes
them by bootstrap, which needs no distributional assumption and handles the
swap-set statistic (a difference of rates over two disjoint, jointly-determined
subsamples) that has no clean closed form.

The DeLong test is the standard parametric alternative for comparing two AUCs on
the SAME sample, and it is stricter than the naive approach of overlapping
individual CIs -- two intervals can overlap while the paired difference is
significant, because the models' errors are correlated. So the AUC comparison
here is a PAIRED bootstrap on the difference, not two independent intervals.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    cuts = np.quantile(expected, np.linspace(0, 1, bins + 1))
    cuts[0], cuts[-1] = -np.inf, np.inf
    cuts = np.unique(cuts)
    e = np.histogram(expected, bins=cuts)[0].astype(float)
    a = np.histogram(actual, bins=cuts)[0].astype(float)
    e = np.clip(e / e.sum(), 1e-6, None)
    a = np.clip(a / a.sum(), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_band(v: float) -> str:
    return "stable" if v < 0.10 else ("monitor" if v < 0.25 else "INVESTIGATE")


def vintage_report(scores_by_vintage: dict, reference: np.ndarray) -> list[dict]:
    """PSI of each vintage's score distribution against the build sample."""
    rows = []
    for name, s in scores_by_vintage.items():
        v = psi(reference, np.asarray(s))
        rows.append({"vintage": name, "n": len(s), "psi": v, "band": psi_band(v)})
    return rows


def characteristic_stability(X_ref: np.ndarray, X_new: np.ndarray,
                             names: list[str]) -> list[dict]:
    out = []
    for i, n in enumerate(names):
        v = psi(X_ref[:, i], X_new[:, i])
        out.append({"characteristic": n, "psi": v, "band": psi_band(v)})
    return sorted(out, key=lambda d: -d["psi"])


# ------------------------------------------------------------------ intervals
def bootstrap_ci(stat_fn, *arrays, n_boot: int = 500, seed: int = 11,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """(point, lo, hi) for any statistic of aligned arrays."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    point = stat_fn(*arrays)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(stat_fn(*[a[idx] for a in arrays]))
        except ValueError:
            continue          # a resample with one class present
    if not vals:
        return point, float("nan"), float("nan")
    return (point, float(np.quantile(vals, alpha / 2)),
            float(np.quantile(vals, 1 - alpha / 2)))


def auc_ci(y: np.ndarray, s: np.ndarray, **kw):
    return bootstrap_ci(lambda yy, ss: roc_auc_score(yy, ss), y, s, **kw)


def gini_ci(y: np.ndarray, s: np.ndarray, **kw):
    return bootstrap_ci(lambda yy, ss: 2 * roc_auc_score(yy, ss) - 1, y, s, **kw)


def paired_auc_difference(y: np.ndarray, s_a: np.ndarray, s_b: np.ndarray,
                          n_boot: int = 500, seed: int = 12) -> dict:
    """PAIRED bootstrap of AUC(b) - AUC(a) on the same sample.

    Paired, not two independent intervals. Two overlapping individual CIs can
    still hide a significant paired difference, because the models make
    correlated errors on the same applicants -- resampling them together
    preserves that correlation and is the only comparison that answers "is B
    better than A on this population".
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    point = roc_auc_score(y, s_b) - roc_auc_score(y, s_a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        diffs.append(roc_auc_score(yy, s_b[idx]) - roc_auc_score(yy, s_a[idx]))
    lo, hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
    return {
        "difference": point, "lo": lo, "hi": hi,
        "significant": lo > 0 or hi < 0,
        "verdict": ("challenger is better" if lo > 0 else
                    "champion is better" if hi < 0 else
                    "indistinguishable -- the interval spans zero"),
    }


def swap_set_ci(y: np.ndarray, s_champ: np.ndarray, s_chall: np.ndarray,
                approval_rate: float, n_boot: int = 500, seed: int = 13) -> dict:
    """Bootstrap the swap-in minus swap-out default-rate gap.

    The swap set is re-derived inside every resample, because the thresholds and
    therefore the membership of both groups depend on the sample. Computing the
    groups once and resampling within them would hold fixed the very thing whose
    variability is being measured, and would report an interval far too narrow.
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    def gap(yy, a, b):
        ta, tb = np.quantile(a, approval_rate), np.quantile(b, approval_rate)
        app_a, app_b = a <= ta, b <= tb
        swap_in, swap_out = app_b & ~app_a, app_a & ~app_b
        if swap_in.sum() < 5 or swap_out.sum() < 5:
            raise ValueError("swap set too small")
        return float(yy[swap_in].mean() - yy[swap_out].mean())

    point = gap(y, s_champ, s_chall)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(gap(y[idx], s_champ[idx], s_chall[idx]))
        except ValueError:
            continue
    lo, hi = float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))
    return {
        "gap": point, "lo": lo, "hi": hi,
        "significant": lo > 0 or hi < 0,
        "verdict": ("challenger trades in WORSE risk" if lo > 0 else
                    "challenger trades in BETTER risk" if hi < 0 else
                    "indistinguishable -- the interval spans zero"),
    }
