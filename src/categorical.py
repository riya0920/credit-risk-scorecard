"""Categorical WoE binning, and the special-value / missing handling a real
scorecard cannot ship without.

Numeric binning was the easy half. A production card is mostly categorical
(employment type, residential status, purpose of loan) and mostly incomplete,
and both bring problems that quantile binning never has to face.

SPECIAL VALUES. In credit data a numeric field routinely carries codes that are
not quantities: -1 for "no bureau record", -7 for "consent withheld", 999 for
"never delinquent". Binning those as numbers puts "no record" at one end of the
risk ordering purely because -1 sorts below 0, which is not a statement anyone
made about risk. They get their own bins, always, and the list is per-feature
because the codes are per-bureau.

MISSING. A missing value is a fact about the applicant's file, not an absence of
one -- "no address history" is informative, and imputing it to the median throws
away the signal and pretends to a precision the data does not have. So NULL gets
its own bin with its own WoE, computed from its own bad rate.

RARE LEVELS. A category with nine applicants has a bad rate that is noise. Levels
below a population floor are merged into an OTHER bucket rather than being given
a WoE the model will happily trust. The floor is a parameter and stated.

The awkward case, handled explicitly: a level with ZERO bads. Its empirical WoE
is infinite. The Laplace correction in `woe.py` keeps it finite, but a finite
number computed from zero events is still not evidence, so such levels are
flagged in the binning table rather than silently smoothed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPS = 0.5
MIN_LEVEL_SHARE = 0.01          # levels below 1% of the population merge to OTHER
LOW_EVENT_FLOOR = 10            # fewer bads than this = not evidence


@dataclass
class CategoricalBin:
    level: str
    n: int
    n_bad: int
    bad_rate: float
    woe: float
    iv_contrib: float
    is_missing: bool = False
    is_special: bool = False
    is_other: bool = False
    low_event: bool = False

    @property
    def label(self) -> str:
        """Numeric bins expose `label`; categorical ones exposed `level`.

        The two are the same idea and the Scorecard reads `label`, so the alias
        is what lets one card carry both kinds of characteristic instead of
        needing a parallel code path. A parallel code path is how the points
        table and the reason codes drift apart.
        """
        return self.level


@dataclass
class CategoricalBinning:
    feature: str
    bins: list[CategoricalBin]
    iv: float
    mapping: dict = field(default_factory=dict)
    other_woe: float = 0.0

    def transform(self, x) -> np.ndarray:
        s = pd.Series(x).astype("object")
        out = np.full(len(s), self.other_woe, dtype=float)
        for i, v in enumerate(s):
            key = _key(v)
            out[i] = self.mapping.get(key, self.other_woe)
        return out

    def bin_of(self, value):
        key = _key(value)
        for b in self.bins:
            if b.level == key:
                return b
        return next((b for b in self.bins if b.is_other), self.bins[-1])

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "level": b.level, "n": b.n, "bad": b.n_bad,
            "bad_rate": round(b.bad_rate, 4), "woe": round(b.woe, 4),
            "iv": round(b.iv_contrib, 4),
            "flag": ("MISSING" if b.is_missing else "SPECIAL" if b.is_special
                     else "OTHER" if b.is_other else
                     "LOW-EVENT" if b.low_event else "")}
            for b in self.bins])


def _key(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "__MISSING__"
    return str(v)


def fit_categorical_binning(x, y: np.ndarray, feature: str,
                            special_values: list | None = None,
                            min_share: float = MIN_LEVEL_SHARE) -> CategoricalBinning:
    """WoE per level, with missing, special and rare levels handled explicitly."""
    specials = {str(v) for v in (special_values or [])}
    s = pd.Series(x).astype("object").map(_key)
    y = np.asarray(y)

    tot_bad = int(y.sum())
    tot_good = len(y) - tot_bad
    counts = s.value_counts()
    floor = max(int(min_share * len(s)), 1)

    keep = {lvl for lvl, n in counts.items()
            if n >= floor or lvl == "__MISSING__" or lvl in specials}

    bins: list[CategoricalBin] = []
    mapping: dict[str, float] = {}
    other_mask = ~s.isin(keep)

    def add(level: str, mask: np.ndarray, **flags) -> None:
        n = int(mask.sum())
        if n == 0:
            return
        bad = int(y[mask].sum())
        good = n - bad
        p_bad = (bad + EPS) / (tot_bad + EPS)
        p_good = (good + EPS) / (tot_good + EPS)
        woe = float(np.log(p_good / p_bad))
        bins.append(CategoricalBin(
            level=level, n=n, n_bad=bad, bad_rate=bad / n, woe=woe,
            iv_contrib=float((p_good - p_bad) * woe),
            low_event=bad < LOW_EVENT_FLOOR, **flags))
        if not flags.get("is_other"):
            mapping[level] = woe

    for lvl in sorted(keep):
        mask = (s == lvl).to_numpy()
        add(lvl, mask,
            is_missing=(lvl == "__MISSING__"),
            is_special=(lvl in specials))

    add("__OTHER__", other_mask.to_numpy(), is_other=True)
    other_woe = next((b.woe for b in bins if b.is_other), 0.0)

    return CategoricalBinning(feature, bins, sum(b.iv_contrib for b in bins),
                              mapping, other_woe)


def split_special_values(x: np.ndarray, special_values: list) -> tuple:
    """Separate special codes from genuine quantities in a numeric column.

    Returns (numeric_mask, specials_series). Binning -1 for "no bureau record"
    as a number puts it at the bottom of the risk ordering because -1 sorts
    below 0 -- which is an artefact of the encoding, not a statement anyone made
    about that applicant.
    """
    arr = np.asarray(x, dtype=float)
    special = np.isin(arr, np.asarray(special_values, dtype=float))
    missing = np.isnan(arr)
    return ~(special | missing), pd.Series(
        np.where(missing, "__MISSING__",
                 np.where(special, arr.astype(object), None)))


# ---------------------------------------------------------------- hybrid
@dataclass
class HybridBinning:
    """A numeric characteristic that also carries special codes.

    Credit age is a quantity, except when the bureau returns -9 for "no hit".
    Neither pure treatment is right:

      bin it all as a NUMBER   -9 sorts below every real credit age, so the
                               no-hit population lands in the youngest-file bin
                               and inherits its risk. That is an artefact of the
                               encoding, not a statement anyone made about the
                               applicant.
      bin it all as a CATEGORY every distinct age becomes its own level, the
                               rare ones merge into `__OTHER__`, and the whole
                               characteristic collapses to two bins with
                               identical points -- which is what happened here
                               before this class existed, and it made the card
                               carry a row that could not affect a decision.

    So the special codes get their own bins, the remainder is binned as the
    quantity it is, and the two sets of bins sit on one card under one name.
    """
    feature: str
    numeric: object                 # woe.Binning over the non-special rows
    specials: CategoricalBinning
    special_values: list

    @property
    def bins(self):
        return list(self.numeric.bins) + [
            b for b in self.specials.bins if b.is_special or b.is_missing]

    @property
    def iv(self) -> float:
        return float(self.numeric.iv + sum(
            b.iv_contrib for b in self.specials.bins
            if b.is_special or b.is_missing))

    def _split(self, x):
        arr = np.asarray(x, dtype=float)
        special = np.isin(arr, np.asarray(self.special_values, dtype=float))
        return arr, special | np.isnan(arr)

    def transform(self, x) -> np.ndarray:
        arr, is_special = self._split(x)
        out = self.numeric.transform(np.where(is_special, np.nan, arr))
        if is_special.any():
            out[is_special] = self.specials.transform(
                np.asarray(arr, dtype=object)[is_special])
        return out

    def bin_of(self, value):
        arr, is_special = self._split([value])
        if is_special[0]:
            return self.specials.bin_of(value)
        return self.numeric.bin_of(float(value))

    def table(self) -> pd.DataFrame:
        return pd.concat([self.numeric.table(),
                          self.specials.table()], ignore_index=True)


def fit_hybrid_binning(x, y: np.ndarray, feature: str, special_values: list,
                       **kwargs) -> HybridBinning:
    from .woe import fit_binning

    numeric_mask, _ = split_special_values(x, special_values)
    arr = np.asarray(x, dtype=float)
    numeric = fit_binning(arr[numeric_mask], np.asarray(y)[numeric_mask], feature,
                          **kwargs)
    specials = fit_categorical_binning(
        np.asarray(arr, dtype=object), y, feature,
        special_values=[float(v) for v in special_values])
    return HybridBinning(feature, numeric, specials, list(special_values))
