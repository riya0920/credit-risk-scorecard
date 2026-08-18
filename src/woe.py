"""Weight-of-Evidence binning and Information Value.

Why a bank still ships this in 2026 when a GBM wins on AUC: a WoE scorecard is
monotonic by construction, its points are additive and auditable, adverse-action
reasons fall out of the arithmetic rather than being reconstructed after the
fact, and the whole model fits on one page that a regulator, a credit officer,
and a collections agent can all read. Those are product requirements, not
nostalgia. Sometimes the GBM wins that argument now -- but it has to win it.

WoE for bin i:   ln( (good_i / total_good) / (bad_i / total_bad) )
IV            :   sum over bins of (good_rate_i - bad_rate_i) * WoE_i

Convention used here: "bad" = default = target 1. WoE is positive where the bin
is safer than the portfolio, so a positive coefficient on WoE means safer, and
points go UP. Fixing the sign convention in one place and stating it is worth
more than it sounds -- half of scorecard bugs are sign errors.

Standard IV bands (stated because an unlabeled IV is meaningless):
  < 0.02  useless | 0.02-0.10 weak | 0.10-0.30 medium | 0.30-0.50 strong
  > 0.50  suspiciously strong -- check for leakage before celebrating
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 0.5      # Laplace-style correction so an empty cell doesn't produce inf


@dataclass
class Bin:
    label: str
    lo: float
    hi: float
    n: int
    n_bad: int
    bad_rate: float
    woe: float
    iv_contrib: float


@dataclass
class Binning:
    feature: str
    bins: list[Bin]
    iv: float
    monotonic: bool

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros(len(x), dtype=float)
        for b in self.bins:
            m = (x > b.lo) & (x <= b.hi)
            out[m] = b.woe
        return out

    def bin_of(self, value: float) -> Bin:
        for b in self.bins:
            if b.lo < value <= b.hi:
                return b
        return self.bins[-1]

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "bin": b.label, "n": b.n, "bad": b.n_bad,
            "bad_rate": round(b.bad_rate, 4), "woe": round(b.woe, 4),
            "iv": round(b.iv_contrib, 4)} for b in self.bins])


def _woe_of(n_bad: int, n_good: int, tot_bad: int, tot_good: int) -> tuple[float, float]:
    p_bad = (n_bad + EPS) / (tot_bad + EPS)
    p_good = (n_good + EPS) / (tot_good + EPS)
    woe = float(np.log(p_good / p_bad))
    return woe, float((p_good - p_bad) * woe)


def fit_binning(x: np.ndarray, y: np.ndarray, feature: str,
                max_bins: int = 6, min_share: float = 0.05,
                enforce_monotonic: bool = True) -> Binning:
    """Quantile bins, then merged until the bad-rate is monotonic in the bin
    order and every bin carries at least `min_share` of the population.

    Monotonicity is not cosmetic. A scorecard where risk goes down, then up, then
    down across income cannot be explained to an applicant, cannot be defended to
    a regulator, and is usually the model fitting noise in a thin bin.
    """
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    n = len(xs)
    tot_bad = int(ys.sum())
    tot_good = n - tot_bad

    edges = list(np.unique(np.quantile(xs, np.linspace(0, 1, max_bins + 1))))
    edges[0], edges[-1] = -np.inf, np.inf

    def build(edge_list):
        bins = []
        for lo, hi in zip(edge_list[:-1], edge_list[1:]):
            m = (xs > lo) & (xs <= hi)
            cnt = int(m.sum())
            if cnt == 0:
                continue
            bad = int(ys[m].sum())
            woe, ivc = _woe_of(bad, cnt - bad, tot_bad, tot_good)
            bins.append(Bin("({:.4g}, {:.4g}]".format(lo, hi), float(lo), float(hi),
                            cnt, bad, bad / cnt, woe, ivc))
        return bins

    bins = build(edges)

    # Merge thin bins, then merge to enforce monotonic bad rate.
    changed = True
    while changed and len(bins) > 2:
        changed = False
        for i, b in enumerate(bins):
            if b.n < min_share * n:
                j = i - 1 if i > 0 else i + 1
                edges.pop(max(min(i, j), 1) if i else 1)
                bins = build(edges)
                changed = True
                break

    if enforce_monotonic:
        while len(bins) > 2:
            rates = [b.bad_rate for b in bins]
            inc = all(a <= b for a, b in zip(rates, rates[1:]))
            dec = all(a >= b for a, b in zip(rates, rates[1:]))
            if inc or dec:
                break
            # merge the pair that violates the dominant direction most
            direction = 1 if rates[0] <= rates[-1] else -1
            worst, worst_i = -1.0, 1
            for i in range(len(rates) - 1):
                viol = (rates[i] - rates[i + 1]) * direction
                if viol > worst:
                    worst, worst_i = viol, i + 1
            edges.pop(worst_i)
            bins = build(edges)

    rates = [b.bad_rate for b in bins]
    monotonic = (all(a <= b for a, b in zip(rates, rates[1:]))
                 or all(a >= b for a, b in zip(rates, rates[1:])))
    return Binning(feature, bins, sum(b.iv_contrib for b in bins), monotonic)


def iv_band(iv: float) -> str:
    if iv < 0.02:
        return "useless"
    if iv < 0.10:
        return "weak"
    if iv < 0.30:
        return "medium"
    if iv < 0.50:
        return "strong"
    return "SUSPICIOUS - check leakage"
