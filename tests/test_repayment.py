"""A real repayment outcome, and the fairness question a decision target cannot pose."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.repayment import (COLUMNS, DATA, calibration_by_group,
                           calibration_error, load, outcome_rates,
                           undocumented_codes)

pytestmark = pytest.mark.skipif(
    not DATA.exists(),
    reason="repayment data not downloaded -- run fetch_repayment_data.py")


@pytest.fixture(scope="module")
def df():
    return load()


# ------------------------------------------------- the mapping was verified
def test_the_default_rate_matches_the_published_figure(df):
    """The check that confirms the whole column mapping.

    The file arrives with columns named x1..x23, y. The published dictionary
    gives a mapping; this asserts the consequence rather than the mapping,
    because the consequence is independently known: this study's default rate is
    22.12%. If the target column were mis-mapped this would not land on it.
    """
    assert df["default"].mean() == pytest.approx(0.2212, abs=0.0002)


def test_each_mapped_column_has_the_range_its_name_implies(df):
    """Mapping a column by position and hoping is how `age` ends up holding a
    bill amount. Each one is checked against what its name claims."""
    assert 10_000 <= df.limit_bal.min() and df.limit_bal.max() <= 1_000_000
    assert set(df.sex.unique()) == {1, 2}
    assert 18 <= df.age.min() and df.age.max() <= 100
    assert df.pay_0.min() < 0, "a repayment status should go negative (paid early)"
    assert df.bill_1.min() < 0, "a bill amount should go negative (credit balance)"
    assert df.pay_amt_1.min() >= 0, "a payment should not be negative"
    assert set(df["default"].unique()) == {0, 1}


def test_every_x_column_is_mapped(df):
    """A column left as `x14` is one nobody checked, and it will be used by
    something eventually."""
    assert not [c for c in df.columns if c.startswith("x")]
    assert len(COLUMNS) == 24


# ------------------------------------------------------ undocumented codes
def test_undocumented_codes_exist_and_are_reported(df):
    """The dictionary says EDUCATION takes 1-4 and MARRIAGE 1-3. The file
    disagrees, and that disagreement is a fact about the data-collection
    process."""
    und = undocumented_codes(df)
    assert len(und) >= 3
    assert set(und.column) <= {"education_label", "marriage_label"}
    assert (und.n > 0).all()


def test_undocumented_codes_are_not_folded_into_other(df):
    """Folding them away destroys the only evidence that they happened -- and
    their default rates differ, so they are not interchangeable with `other`
    even on the merits."""
    assert "undocumented_5" in set(df.education_label)
    assert "other" in set(df.education_label)
    und = undocumented_codes(df)
    other_rate = df.loc[df.education_label == "other", "default"].mean()
    assert not np.allclose(und[und.column == "education_label"].default_rate,
                           other_rate, atol=0.01)


def test_an_unmapped_code_is_labelled_rather_than_dropped(df):
    """`.map` on an unlisted code yields NaN, which silently drops the row
    later. Nothing should be NaN here."""
    assert df.education_label.notna().all()
    assert df.marriage_label.notna().all()
    assert df.sex_label.notna().all()


# --------------------------------------------------------- derived features
def test_utilisation_is_guarded_against_a_zero_limit(df):
    assert df.utilisation.notna().all() or (df.limit_bal == 0).any()


def test_delinquency_features_are_consistent(df):
    """months_delinquent counts the months in arrears; worst_delinquency is the
    deepest. A row with zero months cannot have a positive worst."""
    zero_months = df[df.months_delinquent == 0]
    assert (zero_months.worst_delinquency <= 0).all()


# ------------------------------------------------ the two different questions
def test_outcome_rates_produce_a_real_default_rate_per_group(df):
    """The thing HMDA structurally cannot produce."""
    r = outcome_rates(df, "sex_label")
    assert set(r.sex_label) == {"male", "female"}
    assert (r.default_rate.between(0, 1)).all()
    assert r.n.sum() == len(df)


def test_men_and_women_default_at_materially_different_rates(df):
    """The measured fact that makes the disparity-versus-calibration tension
    real rather than hypothetical: if the groups defaulted identically, a
    well-calibrated model would decline them identically and there would be
    nothing to discuss."""
    r = outcome_rates(df, "sex_label").set_index("sex_label")
    gap = abs(r.default_rate["male"] - r.default_rate["female"])
    assert gap > 0.02, "default rates are too close for the tension to arise"


def test_calibration_is_measured_per_group_and_per_band(df):
    rng = np.random.default_rng(0)
    p = np.clip(df["default"] * 0.4 + rng.random(len(df)) * 0.4, 0, 1)
    cal = calibration_by_group(df, p, "sex_label", bins=5)
    assert set(cal.sex_label) == {"male", "female"}
    assert cal.band.nunique() <= 5
    assert np.allclose(cal.gap, cal.observed - cal.predicted)


def test_a_perfectly_calibrated_prediction_has_almost_no_gap(df):
    """The control. If this does not come out near zero the calibration
    arithmetic is wrong and every finding built on it is noise."""
    rng = np.random.default_rng(1)
    p = rng.random(len(df))
    y = (rng.random(len(df)) < p).astype(int)
    work = df.copy()
    work["default"] = y
    err = calibration_error(calibration_by_group(work, p, "sex_label"),
                            "sex_label")
    assert (err.weighted_abs_gap < 0.03).all()


def test_a_biased_prediction_is_caught_for_the_group_it_is_biased_against(df):
    """A model can be well calibrated overall and badly calibrated for one
    group, which is exactly the failure a pooled calibration plot hides."""
    rng = np.random.default_rng(2)
    p = np.full(len(df), df["default"].mean())
    work = df.copy()
    # Make one group default far more than the flat prediction says.
    male = (work.sex_label == "male").to_numpy()
    y = (rng.random(len(work)) < np.where(male, 0.6, 0.1)).astype(int)
    work["default"] = y

    err = calibration_error(calibration_by_group(work, p, "sex_label"),
                            "sex_label").set_index("sex_label")
    assert err.loc["male", "signed_gap"] > 0.2
    assert err.loc["female", "signed_gap"] < 0.0


def test_calibration_error_is_population_weighted(df):
    """An unweighted average over bands lets a band holding forty accounts count
    as much as one holding four thousand."""
    cal = pd.DataFrame({
        "g": ["a", "a"], "band": [0, 1], "n": [10_000, 10],
        "predicted": [0.1, 0.1], "observed": [0.1, 0.9], "gap": [0.0, 0.8]})
    err = calibration_error(cal, "g")
    assert err.weighted_abs_gap.iloc[0] < 0.01, (
        "a 10-account band dominated a 10,000-account one")


def test_a_constant_prediction_produces_one_band_rather_than_nothing(df):
    """A real bug, not defensive padding.

    `pd.qcut` on a constant vector returns all-NaN even with duplicates="drop",
    the groupby then discards every row, and calibration came back as an EMPTY
    frame -- which reads as "nothing to report" when it means "the model outputs
    one number".

    A constant prediction is exactly what a broken model produces, so silence
    here was silence at the moment the check is most needed.
    """
    p = np.full(len(df), 0.5)
    cal = calibration_by_group(df, p, "sex_label")
    assert len(cal) > 0, "a constant prediction produced an empty calibration"
    assert cal.band.nunique() == 1
    err = calibration_error(cal, "sex_label")
    assert set(err.sex_label) == {"male", "female"}
    # And the single band still measures whether the one number is right.
    assert (err.signed_gap.abs() > 0.2).all(), (
        "predicting 0.5 against a 22% default rate should show a large gap")
