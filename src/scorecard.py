"""Points-based scorecard: logistic on WoE, scaled to points.

The scaling arithmetic, written out because "600 base, PDO 20" is quoted far
more often than it is derived:

    factor = PDO / ln(2)
    offset = base_score - factor * ln(base_odds)
    score  = offset + factor * ln(odds)

with odds = P(good)/P(bad). PDO ("points to double the odds") = 20 means every
20 points doubles the good:bad odds. Base 600 at base odds 20:1 means an
applicant scoring 600 is expected to be good 20 times for each bad.

Distributing the score across features (this is what makes the card a *card*):

    score = offset + factor*intercept + sum_j [ factor * beta_j * WoE_j(x) ]

so each attribute contributes a fixed, printable number of points. Adverse
action then needs no explainer model at all: the reasons are the attributes where
the applicant lost the most points against the maximum they could have scored.
That is the points-lost method, and it is exact rather than approximate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from .woe import Binning


@dataclass
class ScorecardConfig:
    base_score: float = 600.0
    base_odds: float = 20.0
    pdo: float = 20.0

    @property
    def factor(self) -> float:
        return self.pdo / math.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * math.log(self.base_odds)


class Scorecard:
    def __init__(self, binnings: dict[str, Binning], cfg: ScorecardConfig | None = None):
        self.binnings = binnings
        self.features = list(binnings)
        self.cfg = cfg or ScorecardConfig()
        self.model = LogisticRegression(max_iter=1000)

    def _woe_matrix(self, X: dict[str, np.ndarray]) -> np.ndarray:
        return np.column_stack([self.binnings[f].transform(X[f]) for f in self.features])

    def fit(self, X: dict[str, np.ndarray], y: np.ndarray) -> "Scorecard":
        self.model.fit(self._woe_matrix(X), y)
        return self

    def predict_proba(self, X: dict[str, np.ndarray]) -> np.ndarray:
        return self.model.predict_proba(self._woe_matrix(X))[:, 1]

    # -- points ------------------------------------------------------------
    def points_table(self) -> list[dict]:
        """The scorecard itself: one row per attribute, with its points.

        Note the sign. WoE is positive where the bin is SAFER, and the logistic
        coefficient on a WoE feature is negative when the target is default, so
        `-beta * woe * factor` yields more points for safer bins. Getting this
        backwards produces a card that reads perfectly and ranks upside down.
        """
        rows = []
        f = self.cfg.factor
        n_feat = len(self.features)
        intercept_pts = (self.cfg.offset + f * self.model.intercept_[0]) / n_feat
        for j, feat in enumerate(self.features):
            beta = float(self.model.coef_[0][j])
            for b in self.binnings[feat].bins:
                rows.append({
                    "feature": feat,
                    "bin": b.label,
                    "n": b.n,
                    "bad_rate": round(b.bad_rate, 4),
                    "woe": round(b.woe, 4),
                    "points": round(intercept_pts - f * beta * b.woe, 1),
                })
        return rows

    def score(self, row: dict[str, float]) -> float:
        f = self.cfg.factor
        n_feat = len(self.features)
        total = self.cfg.offset + f * self.model.intercept_[0]
        for j, feat in enumerate(self.features):
            beta = float(self.model.coef_[0][j])
            woe = self.binnings[feat].bin_of(row[feat]).woe
            total -= f * beta * woe
        return float(total)

    def scores(self, X: dict[str, np.ndarray]) -> np.ndarray:
        f = self.cfg.factor
        base = self.cfg.offset + f * self.model.intercept_[0]
        woe_m = self._woe_matrix(X)
        contrib = -(f * woe_m * self.model.coef_[0])
        return base + contrib.sum(axis=1)

    # -- adverse action ----------------------------------------------------
    def reason_codes(self, row: dict[str, float], top_k: int = 4) -> list[dict]:
        """Points-lost method: for each characteristic, the gap between the
        points this applicant earned and the maximum points available on that
        characteristic. The largest gaps are the principal reasons for the
        decline, which is exactly what ECOA Reg B (12 CFR 1002.9) requires the
        notice to state.
        """
        f = self.cfg.factor
        # Same per-attribute points the printed scorecard shows, intercept share
        # included. points_lost is unaffected by the share (it cancels), but the
        # earned/available figures must tie to the card an examiner is holding --
        # a reason code that cites points nobody can find on the scorecard is a
        # finding waiting to happen.
        intercept_pts = (self.cfg.offset + f * self.model.intercept_[0]) / len(self.features)
        out = []
        for j, feat in enumerate(self.features):
            beta = float(self.model.coef_[0][j])
            binning = self.binnings[feat]
            earned = intercept_pts - f * beta * binning.bin_of(row[feat]).woe
            best = max(intercept_pts - f * beta * b.woe for b in binning.bins)
            out.append({
                "feature": feat,
                "applicant_bin": binning.bin_of(row[feat]).label,
                "points_earned": round(earned, 1),
                "points_available": round(best, 1),
                "points_lost": round(best - earned, 1),
            })
        out.sort(key=lambda d: -d["points_lost"])
        for i, d in enumerate(out[:top_k], 1):
            d["rank"] = i
        return out[:top_k]
