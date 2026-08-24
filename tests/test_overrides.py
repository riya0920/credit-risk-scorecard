"""Override tracking: the layer between the model and the lending."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.overrides import (Decision, discretion_disparity, effective_cutoff,
                           frame, override_performance, override_report,
                           override_value, underwriter_concentration,
                           undocumented)


def _d(app_id, score, approved, cutoff=620.0, **kw):
    return Decision(app_id=app_id, score=score, cutoff=cutoff,
                    approved=approved, **kw)


# ------------------------------------------------------------ vocabulary
def test_low_and_high_side_are_not_the_same_thing():
    """The two terms are easy to swap and the consequences are opposite: one
    adds risk the model priced as unacceptable, the other is an adverse action
    needing a documented reason."""
    assert _d("a", 600, True).override_type == "low_side"
    assert _d("b", 700, False).override_type == "high_side"


def test_agreeing_with_the_model_is_not_an_override():
    assert _d("a", 700, True).override_type == "none"
    assert _d("b", 600, False).override_type == "none"


def test_a_score_exactly_at_the_cutoff_is_an_approve():
    """Boundary convention stated once and tested, because ">= cutoff" and
    "> cutoff" differ by a whole band of applicants at the busiest score in the
    distribution."""
    assert _d("a", 620.0, True).override_type == "none"
    assert _d("b", 620.0, False).override_type == "high_side"


def test_the_recorded_score_is_not_recomputed_under_a_new_model():
    """Recomputing an old application under today's card would let a
    recalibration retroactively rewrite override history -- last quarter's
    overrides silently becoming non-overrides. The audit question is what the
    underwriter saw."""
    d = _d("a", 600, True, model_version="v1")
    assert d.override_type == "low_side"
    d.model_version = "v2"                    # the card was recalibrated
    assert d.override_type == "low_side", "history changed under a model bump"


# ----------------------------------------------------------------- rates
def test_override_rate_is_reported_against_a_declared_threshold():
    decisions = [_d(f"a{i}", 700, True) for i in range(95)]
    decisions += [_d(f"b{i}", 600, True) for i in range(5)]
    rep = override_report(frame(decisions))
    assert rep["low_side"] == 5
    assert rep["low_side_rate"] == pytest.approx(0.05)
    assert rep["low_side_exceeds_concern"] is True     # 5% > the 3% low-side line


def test_a_clean_book_trips_nothing():
    rep = override_report(frame([_d(f"a{i}", 700, True) for i in range(50)]))
    assert rep["override_rate"] == 0.0
    assert rep["exceeds_concern"] is False


def test_an_empty_book_does_not_divide_by_zero():
    assert override_report(frame([]))["decisions"] == 0


# ------------------------------------------------------ effective cutoff
def test_the_documented_cutoff_becomes_a_fiction_once_overrides_are_routine():
    """The finding this exists to surface: a model validated at 620 and operated
    at an effective 600 has been validated on a population it does not lend to.
    """
    decisions = []
    for i in range(200):                       # above cutoff, approved
        decisions.append(_d(f"hi{i}", 660, True))
    for i in range(200):                       # 600-620: mostly approved anyway
        decisions.append(_d(f"lo{i}", 605, i < 160))
    for i in range(200):                       # well below: genuinely declined
        decisions.append(_d(f"vlo{i}", 560, False))

    eff = effective_cutoff(frame(decisions))
    assert eff["policy_cutoff"] == 620.0
    assert eff["effective_cutoff"] < 620.0
    assert eff["drift"] < 0
    assert eff["approval_rate_below_cutoff"] > 0.3


def test_a_book_that_follows_its_policy_shows_no_drift():
    decisions = [_d(f"hi{i}", 660, True) for i in range(100)]
    decisions += [_d(f"lo{i}", 560, False) for i in range(100)]
    eff = effective_cutoff(frame(decisions))
    assert eff["effective_cutoff"] >= eff["policy_cutoff"]
    assert eff["approval_rate_below_cutoff"] == 0.0


# -------------------------------------------------------- did they know?
def test_the_two_populations_are_adjacent_by_construction_never_overlapping():
    """The design error this replaced, and the constraint that replaced it.

    A low-side override is below the cutoff by definition and a score-approved
    loan is at or above it by definition, so NO banding can ever put them in the
    same bucket. Bucketing on absolute score leaves them an arbitrary distance
    apart and the comparison silently reports "nothing to compare", forever.
    Bucketing on distance from cutoff makes them exactly ADJACENT -- which is the
    closest the two populations can ever come, and is why `override_value`
    compares across the cutoff boundary rather than within a band.
    """
    decisions = [_d(f"b{i}", 625, True, defaulted=0) for i in range(50)]
    decisions += [_d(f"o{i}", 615, True, defaulted=0) for i in range(50)]
    df = frame(decisions)

    perf = override_performance(df, width=20.0)
    lo = set(perf[perf.override_type == "low_side"].band)
    no = set(perf[perf.override_type == "none"].band)
    assert not (lo & no), "no banding can make these populations overlap"
    assert max(lo) + 20.0 == min(no), "they should be neighbouring bands"

    # And the comparison spans that boundary rather than failing on it.
    val = override_value(df, width=20.0)
    assert val["low_side_n"] == 50 and val["comparison_n"] == 50


def test_low_side_overrides_are_compared_against_accepts_just_above_the_cutoff():
    decisions = [_d(f"b{i}", 625, True, defaulted=1 if i < 15 else 0)
                 for i in range(100)]
    decisions += [_d(f"o{i}", 615, True, defaulted=1 if i < 8 else 0)
                  for i in range(50)]
    # Far-above-cutoff loans must NOT dilute the comparison cohort.
    decisions += [_d(f"hi{i}", 750, True, defaulted=0) for i in range(500)]

    val = override_value(frame(decisions), width=20.0)
    assert val["low_side_n"] == 50
    assert val["comparison_n"] == 100, "the far-above cohort leaked in"
    assert val["score_approved_bad_rate"] == pytest.approx(0.15, abs=0.01)


def test_the_comparison_reports_the_score_gap_that_biases_it():
    """It is not like-for-like: the override cohort scored lower, so the model's
    own prediction is that they default MORE. Reporting the gap is what stops the
    difference being read at face value."""
    decisions = [_d(f"b{i}", 630, True, defaulted=0) for i in range(50)]
    decisions += [_d(f"o{i}", 610, True, defaulted=0) for i in range(50)]
    val = override_value(frame(decisions), width=20.0)
    assert val["expected_gap"] == pytest.approx(20.0)


def test_overrides_outperforming_the_benchmark_is_reported_as_a_model_problem():
    """The counter-intuitive result and the reason the module exists.
    Underwriters being right means the model is missing a real predictor, and the
    response is to find it and retrain -- not to congratulate the credit team."""
    decisions = [_d(f"b{i}", 625, True, defaulted=1 if i < 20 else 0)
                 for i in range(100)]
    decisions += [_d(f"o{i}", 615, True, defaulted=1 if i < 2 else 0)
                  for i in range(100)]
    val = override_value(frame(decisions), width=20.0)
    assert val["lift"] < 0
    assert "retrain" in val["verdict"]


def test_overrides_merely_MATCHING_a_higher_scoring_cohort_is_already_a_signal():
    """The subtlest of the three verdicts. Equal bad rates look like a null
    result and are not one: the override cohort scored lower, so the model
    predicted they would default more, and they did not."""
    decisions = [_d(f"b{i}", 630, True, defaulted=1 if i < 10 else 0)
                 for i in range(100)]
    decisions += [_d(f"o{i}", 610, True, defaulted=1 if i < 10 else 0)
                  for i in range(100)]
    val = override_value(frame(decisions), width=20.0)
    assert val["lift"] == pytest.approx(0.0, abs=0.005)
    assert "should not happen" in val["verdict"]


def test_overrides_underperforming_the_benchmark_is_reported_as_destroying_value():
    decisions = [_d(f"b{i}", 625, True, defaulted=1 if i < 10 else 0)
                 for i in range(100)]
    decisions += [_d(f"o{i}", 615, True, defaulted=1 if i < 40 else 0)
                  for i in range(100)]
    val = override_value(frame(decisions), width=20.0)
    assert val["lift"] > 0
    assert "destroying value" in val["verdict"]


def test_a_recorded_predicted_pd_is_preferred_over_any_comparison_cohort():
    """The clean test: observed defaults against what the score itself predicted
    for the SAME applicants needs no comparison cohort, so none of the
    adjacent-band bias applies."""
    decisions = [_d(f"o{i}", 610, True, defaulted=1 if i < 5 else 0,
                    predicted_pd=0.25) for i in range(100)]
    val = override_value(frame(decisions), width=20.0)
    assert val["basis"].startswith("model's own predicted PD")
    assert val["predicted_bad_rate"] == pytest.approx(0.25)
    assert val["lift"] == pytest.approx(0.05 - 0.25)
    assert "retrain" in val["verdict"]


def test_no_comparable_cohort_returns_no_conclusion_rather_than_a_number():
    """A silent zero here would read as 'overrides perform in line', which is the
    opposite of 'there is nothing to compare against'."""
    decisions = [_d(f"b{i}", 700, True, defaulted=0) for i in range(50)]
    decisions += [_d(f"o{i}", 615, True, defaulted=1) for i in range(20)]
    val = override_value(frame(decisions), width=20.0)
    assert "lift" not in val
    assert "no conclusion" in val["note"]


def test_unseasoned_loans_are_excluded_rather_than_counted_as_good():
    """`defaulted=None` means the loan has not seasoned. Treating it as 0 makes
    every recent cohort look excellent, which is how a book looks best right
    before it does not."""
    decisions = [_d(f"b{i}", 625, True, defaulted=1 if i < 10 else 0)
                 for i in range(100)]
    decisions += [_d(f"new{i}", 615, True, defaulted=None) for i in range(500)]
    val = override_value(frame(decisions), width=20.0)
    assert val["low_side_n"] == 0, "unseasoned loans entered the comparison"
    assert "no conclusion" in val["note"]


# ------------------------------------------------------------- disparity
def test_the_override_layer_can_be_unfair_while_the_model_is_not():
    """The point of the whole file. The model here is a single number applied
    identically to everyone -- provably neutral -- and the discretionary layer
    above it is not."""
    decisions = []
    for i in range(200):
        decisions.append(_d(f"a{i}", 615, i < 60, group="A"))     # 30% waved in
    for i in range(200):
        decisions.append(_d(f"b{i}", 615, i < 6, group="B"))      # 3% waved in

    d = discretion_disparity(frame(decisions))
    a = d[d.group == "A"].iloc[0]
    b = d[d.group == "B"].iloc[0]
    assert a.low_side_rate == pytest.approx(0.30)
    assert b.low_side_rate == pytest.approx(0.03)
    assert b.low_side_ratio == pytest.approx(0.10, abs=0.01)
    assert b.low_side_ratio < 0.80, "an 80%-rule failure in the override layer"


def test_disparity_is_benchmarked_against_the_best_treated_group():
    decisions = [_d(f"a{i}", 615, i < 20, group="A") for i in range(100)]
    decisions += [_d(f"b{i}", 615, i < 10, group="B") for i in range(100)]
    d = discretion_disparity(frame(decisions))
    assert d.low_side_ratio.max() == pytest.approx(1.0)


def test_no_group_data_returns_empty_rather_than_a_fabricated_zero():
    decisions = [_d(f"a{i}", 615, True) for i in range(10)]
    assert discretion_disparity(frame(decisions)).empty


# --------------------------------------------------------- concentration
def test_override_authority_concentration_is_measured():
    decisions = [_d(f"a{i}", 615, True, underwriter="jo") for i in range(30)]
    decisions += [_d(f"b{i}", 615, True, underwriter="sam") for i in range(5)]
    decisions += [_d(f"c{i}", 700, True, underwriter="sam") for i in range(65)]

    g = underwriter_concentration(frame(decisions))
    top = g.iloc[0]
    assert top.underwriter == "jo"
    assert top.share_of_low_side == pytest.approx(30 / 35, abs=0.01)
    assert top.low_side_rate == pytest.approx(1.0)


def test_an_override_with_no_recorded_reason_is_flagged():
    """A high-side override with no reason is an adverse action without a
    documented basis -- an ECOA problem before it is a modelling one."""
    decisions = [_d("a", 700, False, reason=""),
                 _d("b", 700, False, reason="collateral condition"),
                 _d("c", 700, True, reason="")]        # not an override at all
    flagged = undocumented(frame(decisions))
    assert list(flagged.app_id) == ["a"]


# ------------------------------------- the real-data run, pinned
def test_a_group_can_show_more_disagreement_in_BOTH_directions():
    """The alternative explanation that the ratio columns hide, and the reason
    docs/OVERRIDES.md prints a `both` column before the ratios.

    A group with elevated low-side AND high-side rates is not obviously getting
    more discretion -- it is a group the model fits worse. Constructed here so
    the pattern is unambiguous: identical true approval behaviour, but group B's
    approvals are uncorrelated with the score.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    decisions = []
    for i in range(1000):                      # A: decisions track the score
        sc = float(rng.normal(620, 30))
        decisions.append(_d(f"a{i}", sc, sc >= 620, group="A"))
    for i in range(1000):                      # B: decisions ignore the score
        sc = float(rng.normal(620, 30))
        decisions.append(_d(f"b{i}", sc, bool(rng.random() < 0.5), group="B"))

    d = discretion_disparity(frame(decisions))
    a = d[d.group == "A"].iloc[0]
    b = d[d.group == "B"].iloc[0]
    assert b.low_side_rate > a.low_side_rate
    assert b.high_side_rate > a.high_side_rate, (
        "B should disagree in BOTH directions -- that is the signature of poor "
        "model fit rather than of one-sided discretion")


def test_calibrating_the_cutoff_to_the_approval_rate_makes_the_sides_equal():
    """Pinned because docs/OVERRIDES.md claims the symmetry carries no
    information, and that claim depends on this construction being exact."""
    import numpy as np
    rng = np.random.default_rng(1)
    scores = rng.normal(600, 40, size=2000)
    approved = rng.random(2000) < 0.7
    cutoff = float(np.quantile(scores, 1 - approved.mean()))
    decisions = [Decision(app_id=str(i), score=float(s), cutoff=cutoff,
                          approved=bool(a))
                 for i, (s, a) in enumerate(zip(scores, approved))]
    rep = override_report(frame(decisions))
    assert abs(rep["low_side"] - rep["high_side"]) <= 1


def test_the_real_hmda_run_produces_a_disagreement_rate_far_above_the_threshold():
    """A live check on the caveat rather than on the number. The measured
    disagreement rate on real HMDA is ~22%, four times the 5% governance
    threshold -- and docs/OVERRIDES.md must NOT read that as a governance
    finding, because it is a restatement of a weak AUC. If this ever falls below
    the threshold the caveat in section 3 has stopped being necessary.
    """
    from src.hmda import DATA
    if not DATA.exists():
        pytest.skip("HMDA extract not downloaded -- see src/hmda.py")

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from src.hmda import load, protected

    feats = ["dti", "ltv", "income_k", "loan_amount_k", "loan_to_income"]
    df = protected(load()).dropna(subset=feats).copy()
    pdb = LogisticRegression(max_iter=3000).fit(
        df[feats].to_numpy(float), df.denied.to_numpy()).predict_proba(
            df[feats].to_numpy(float))[:, 1]
    rate = float((df.denied == 0).mean())
    cutoff = float(np.quantile(-pdb, 1 - rate))

    decisions = [Decision(app_id=str(i), score=float(-p), cutoff=cutoff,
                          approved=bool(dn == 0))
                 for i, (p, dn) in enumerate(zip(pdb, df.denied))]
    rep = override_report(frame(decisions))
    assert rep["override_rate"] > 0.15, (
        "the disagreement rate has dropped; check whether section 3's warning "
        "that this is a model-fit statistic is still needed")
