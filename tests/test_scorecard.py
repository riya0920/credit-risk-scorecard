"""WoE binning, scorecard scaling, adverse action, fair lending, reject inference."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import fair_lending as fl
from src import reject_inference as ri
from src.scorecard import Scorecard, ScorecardConfig
from src.woe import fit_binning, iv_band


@pytest.fixture(scope="module")
def toy():
    """A feature with a monotone relationship to default, by construction."""
    rng = np.random.default_rng(0)
    n = 6000
    x = rng.uniform(0, 100, n)
    p = 0.05 + 0.006 * x                       # risk rises with x
    y = (rng.random(n) < p).astype(int)
    return x, y


# ------------------------------------------------------------------- WoE
def test_binning_is_monotonic_in_bad_rate(toy):
    """A card where risk goes down, then up, then down cannot be explained to an
    applicant and is usually the model fitting noise in a thin bin."""
    x, y = toy
    b = fit_binning(x, y, "x")
    rates = [bn.bad_rate for bn in b.bins]
    assert b.monotonic
    assert rates == sorted(rates) or rates == sorted(rates, reverse=True)


def test_information_value_is_positive_for_a_predictive_feature(toy):
    x, y = toy
    assert fit_binning(x, y, "x").iv > 0.10


def test_noise_feature_has_negligible_iv(toy):
    _x, y = toy
    rng = np.random.default_rng(1)
    noise = rng.uniform(0, 100, len(y))
    assert fit_binning(noise, y, "noise").iv < 0.02


def test_iv_bands_flag_suspiciously_strong_features():
    """IV above 0.5 is usually leakage, not a great feature."""
    assert iv_band(0.01) == "useless"
    assert iv_band(0.2) == "medium"
    assert "SUSPICIOUS" in iv_band(0.8)


def test_woe_sign_convention_is_stable(toy):
    """Positive WoE = safer than the portfolio. Half of scorecard bugs are sign
    errors, so the convention gets a test rather than a comment."""
    x, y = toy
    b = fit_binning(x, y, "x")
    safest = min(b.bins, key=lambda bn: bn.bad_rate)
    riskiest = max(b.bins, key=lambda bn: bn.bad_rate)
    assert safest.woe > riskiest.woe


# ------------------------------------------------------------- scorecard
@pytest.fixture(scope="module")
def card(toy):
    x, y = toy
    rng = np.random.default_rng(2)
    x2 = np.clip(x + rng.normal(0, 20, len(x)), 0, 200)
    cols = {"x": x, "x2": x2}
    binnings = {k: fit_binning(v, y, k) for k, v in cols.items()}
    c = Scorecard(binnings, ScorecardConfig(base_score=600, base_odds=20, pdo=20))
    c.fit(cols, y)
    return c, cols, y


def test_pdo_scaling_math(card):
    """PDO 20 means 20 points doubles the good:bad odds. factor = PDO/ln(2)."""
    c, _cols, _y = card
    import math
    assert c.cfg.factor == pytest.approx(20 / math.log(2))
    assert c.cfg.offset == pytest.approx(600 - c.cfg.factor * math.log(20))


def test_higher_score_means_lower_risk(card):
    c, cols, y = card
    scores = c.scores(cols)
    lo, hi = np.quantile(scores, 0.2), np.quantile(scores, 0.8)
    assert y[scores <= lo].mean() > y[scores >= hi].mean(), \
        "the card ranks upside down -- check the WoE/coefficient sign"


def test_reason_codes_tie_to_the_printed_scorecard(card):
    """A reason code citing points nobody can find on the card is a finding
    waiting to happen."""
    c, cols, _y = card
    table = {(r["feature"], r["bin"]): r["points"] for r in c.points_table()}
    row = {k: float(v[0]) for k, v in cols.items()}
    for rc in c.reason_codes(row):
        binning = c.binnings[rc["feature"]]
        label = binning.bin_of(row[rc["feature"]]).label
        assert table[(rc["feature"], label)] == pytest.approx(
            rc["points_earned"], abs=0.05)


def test_reason_codes_are_ranked_by_points_lost(card):
    c, cols, _y = card
    row = {k: float(v[0]) for k, v in cols.items()}
    codes = c.reason_codes(row)
    lost = [rc["points_lost"] for rc in codes]
    assert lost == sorted(lost, reverse=True)
    assert all(rc["points_lost"] >= 0 for rc in codes)


# ---------------------------------------------------------- fair lending
def test_air_detects_a_planted_disparity():
    approved = np.array([1] * 50 + [0] * 50 + [1] * 90 + [0] * 10)
    group = np.array([1] * 100 + [0] * 100)
    r = fl.adverse_impact_ratio(approved, group)
    assert r["air"] == pytest.approx(50 / 90, rel=1e-6)
    assert r["flags"] is True
    assert r["disadvantaged"] == 1


def test_air_does_not_flag_parity():
    approved = np.array([1] * 80 + [0] * 20 + [1] * 80 + [0] * 20)
    group = np.array([1] * 100 + [0] * 100)
    assert fl.adverse_impact_ratio(approved, group)["flags"] is False


def test_air_confidence_interval_brackets_the_estimate():
    rng = np.random.default_rng(4)
    group = (rng.random(4000) < 0.4).astype(int)
    approved = (rng.random(4000) < np.where(group == 1, 0.7, 0.8)).astype(int)
    point = fl.adverse_impact_ratio(approved, group)["air"]
    lo, hi = fl.air_confidence_interval(approved, group)
    assert lo < point < hi


def test_proxy_reconstruction_detects_an_encoded_attribute():
    """The number that decides whether 'we do not use the attribute' means
    anything."""
    rng = np.random.default_rng(5)
    group = (rng.random(3000) < 0.4).astype(int)
    proxy = group + rng.normal(0, 0.3, 3000)          # strongly encodes group
    noise = rng.normal(0, 1, 3000)
    assert fl.proxy_reconstruction(np.c_[proxy, noise], group) > 0.85
    assert fl.proxy_reconstruction(np.c_[noise], group) < 0.60


def test_lda_search_separates_its_two_failure_modes():
    """'Did not help' and 'helped but cost too much' are different findings and
    collapsing them is how a real alternative gets dismissed."""
    blocked = fl.less_discriminatory_alternative_search({
        "champion": {"air": 0.80, "auc": 0.70},
        "alt": {"air": 0.90, "auc": 0.65},          # big AIR gain, big AUC cost
    })
    assert not blocked["qualifying_alternatives"]
    assert blocked["blocked_on_cost"]
    assert "tolerance" in blocked["conclusion"]

    useless = fl.less_discriminatory_alternative_search({
        "champion": {"air": 0.80, "auc": 0.70},
        "alt": {"air": 0.801, "auc": 0.699},
    })
    assert not useless["blocked_on_cost"]
    assert "did not" in useless["conclusion"] or "no candidate" in useless["conclusion"]

    good = fl.less_discriminatory_alternative_search({
        "champion": {"air": 0.80, "auc": 0.70},
        "alt": {"air": 0.85, "auc": 0.698},
    })
    assert good["qualifying_alternatives"]


# ------------------------------------------------------ reject inference
def test_parcelling_scales_with_the_multiplier():
    """The multiplier IS the assumption, so it must visibly move the answer."""
    rng = np.random.default_rng(6)
    funded_scores = rng.random(2000)
    funded_y = (rng.random(2000) < funded_scores * 0.4).astype(int)
    reject_scores = rng.random(500) * 0.5 + 0.5

    low = ri.parcelling(reject_scores, funded_scores, funded_y,
                        bad_rate_multiplier=1.0)
    high = ri.parcelling(reject_scores, funded_scores, funded_y,
                         bad_rate_multiplier=3.0)
    assert high.mean() > low.mean()
    assert high.max() <= 1.0


def test_fuzzy_augmentation_preserves_total_weight():
    import pandas as pd
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    pd_hat = np.array([0.2, 0.5, 0.9])
    Xa, y, w = ri.fuzzy_augmentation(X, pd_hat)
    assert len(Xa) == 6 and len(y) == 6
    assert w.sum() == pytest.approx(len(X))
    assert y[:3].sum() == 3 and y[3:].sum() == 0


def test_counterfactual_scoring_reports_direction():
    inferred = np.full(100, 0.6)
    actual = np.zeros(100, dtype=int)
    actual[:30] = 1
    r = ri.score_against_counterfactual(inferred, actual)
    assert r["absolute_error"] == pytest.approx(0.3)
    assert "over-estimates" in r["direction"]
