"""Reject inference: modelling the population you will score, not the one you
happened to fund.

The problem, stated precisely. A lender observes repayment only for approved
loans. The model is therefore fit on `P(default | approved)` but deployed to
answer `P(default | applied)`. Those differ whenever approval was informative --
which is always, because the incumbent policy was not random. The rejected
population is not missing at random; it is missing *because it looked bad*.

Three standard methods, implemented so their assumptions are visible:

  PARCELLING       Score the rejects with the funded-only model, bucket them by
                   score, and assign inferred bad rates by scaling the funded bad
                   rate in each bucket by a factor > 1. The scaling factor is the
                   assumption -- it encodes "a reject in this score band defaults
                   k times as often as a funded loan in the same band" -- and it
                   cannot be estimated from the data. It is a judgement, and this
                   implementation makes you pass it explicitly.

  AUGMENTATION     Re-weight the funded population so it resembles the applicant
                   population. Assumes reject outcomes are predictable from
                   observed features alone (MAR). Cheapest, weakest.

  FUZZY            Duplicate each reject into a good copy and a bad copy with
                   fractional weights from the inferred PD. Uses the whole
                   distribution rather than a hard label, which is honest about
                   the uncertainty -- but propagates the same untestable
                   assumption as parcelling.

**None of these is a fix.** Reject inference cannot create information that was
never observed; it can only make an assumption explicit and propagate it
consistently. The only real fix is a randomised approval holdout -- approving a
small random slice of marginal applicants and observing what happens. That costs
money, which is why it is rare and why the honest answer in an interview is
"we bound the bias and we buy some ground truth where we can afford to".

This module can be VALIDATED here in a way a real lender cannot: the generator
produced counterfactual outcomes for rejected applicants, so the inferred bad
rates can be scored against what would actually have happened.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def parcelling(reject_scores: np.ndarray, funded_scores: np.ndarray,
               funded_y: np.ndarray, n_bands: int = 10,
               bad_rate_multiplier: float = 2.0) -> np.ndarray:
    """Inferred bad probability per reject.

    `bad_rate_multiplier` is the assumption and has no data-driven value. Common
    industry practice is 2x-4x; it is passed in rather than defaulted quietly so
    that changing it is a visible decision.
    """
    edges = np.quantile(funded_scores, np.linspace(0, 1, n_bands + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    funded_band = np.digitize(funded_scores, edges[1:-1])
    reject_band = np.digitize(reject_scores, edges[1:-1])

    inferred = np.zeros(len(reject_scores))
    for b in range(n_bands):
        m_f = funded_band == b
        rate = funded_y[m_f].mean() if m_f.any() else funded_y.mean()
        inferred[reject_band == b] = min(rate * bad_rate_multiplier, 0.999)
    return inferred


def augmentation_weights(funded_scores: np.ndarray, all_scores: np.ndarray,
                         n_bands: int = 10) -> np.ndarray:
    """Re-weight funded loans so their score distribution matches all applicants.

    Assumes MAR: that within a score band, funded and rejected applicants default
    at the same rate. That assumption is exactly what the approval policy
    violates, so treat augmentation as a lower bound on the correction.
    """
    edges = np.quantile(all_scores, np.linspace(0, 1, n_bands + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    f_band = np.digitize(funded_scores, edges[1:-1])
    a_band = np.digitize(all_scores, edges[1:-1])

    weights = np.ones(len(funded_scores))
    for b in range(n_bands):
        n_f = (f_band == b).sum()
        n_a = (a_band == b).sum()
        if n_f:
            weights[f_band == b] = (n_a / len(all_scores)) / (n_f / len(funded_scores))
    return weights


def fuzzy_augmentation(reject_features: pd.DataFrame, inferred_pd: np.ndarray):
    """Each reject becomes two weighted rows: a bad one and a good one.

    Returns (features, y, weights). Preserves the uncertainty instead of forcing
    a hard label onto an applicant whose outcome was never observed.
    """
    X = pd.concat([reject_features, reject_features], ignore_index=True)
    y = np.concatenate([np.ones(len(reject_features)),
                        np.zeros(len(reject_features))])
    w = np.concatenate([inferred_pd, 1 - inferred_pd])
    return X, y.astype(int), w


def score_against_counterfactual(inferred_pd: np.ndarray,
                                 actual_default: np.ndarray) -> dict:
    """Only possible because the generator produced counterfactual outcomes.

    A real lender cannot run this, which is the point: it shows how far the
    assumption is from the truth in a setting where the truth exists.
    """
    actual_rate = float(actual_default.mean())
    inferred_rate = float(inferred_pd.mean())
    return {
        "inferred_bad_rate": inferred_rate,
        "actual_bad_rate": actual_rate,
        "absolute_error": inferred_rate - actual_rate,
        "relative_error": (inferred_rate - actual_rate) / actual_rate
        if actual_rate else float("nan"),
        "direction": ("over-estimates reject risk" if inferred_rate > actual_rate
                      else "under-estimates reject risk"),
    }
