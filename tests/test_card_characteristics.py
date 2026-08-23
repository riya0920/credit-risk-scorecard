"""Categorical, missing, special-value and hybrid characteristics ON the card."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.categorical import (fit_categorical_binning, fit_hybrid_binning,
                             split_special_values)
from src.generate import generate
from src.scorecard import Scorecard, ScorecardConfig
from src.woe import fit_binning


@pytest.fixture(scope="module")
def df():
    return generate(n=20_000, seed=5)


# ------------------------------------------------------------- generator
def test_the_generator_emits_real_categoricals(df):
    for col in ("home_ownership", "loan_purpose", "employment_type"):
        assert df[col].nunique(dropna=False) >= 4


def test_missingness_is_not_at_random(df):
    """It concentrates in the self-employed, who default more. That is what
    makes imputing the mode wrong rather than merely lossy."""
    overall = df.employment_type.isna().mean()
    assert 0.05 < overall < 0.20
    missing_default = df.loc[df.employment_type.isna(), "defaulted"].mean()
    w2_default = df.loc[df.employment_type == "W2", "defaulted"].mean()
    assert missing_default > w2_default + 0.03


def test_vintages_carry_a_real_outcome_gradient(df):
    rates = df.groupby("vintage_month").defaulted.mean()
    assert rates.iloc[-1] > rates.iloc[0] + 0.15


def test_the_base_rate_is_held_fixed_by_the_solved_intercept(df):
    """The categoricals redistribute risk; they must not reprice the book, or
    every before/after comparison in the repo becomes meaningless."""
    assert df.defaulted.mean() == pytest.approx(0.41, abs=0.02)


# ----------------------------------------------------------- categorical
def test_missing_gets_its_own_level_and_its_own_woe(df):
    b = fit_categorical_binning(df.employment_type.to_numpy(object),
                                df.defaulted.to_numpy(), "employment_type")
    levels = {x.level: x for x in b.bins}
    assert "__MISSING__" in levels
    assert levels["__MISSING__"].woe != levels["W2"].woe


def test_a_rare_level_is_merged_rather_than_scored():
    x = np.array(["A"] * 1000 + ["B"] * 1000 + ["Z"] * 3, dtype=object)
    y = np.array([0] * 500 + [1] * 500 + [0] * 500 + [1] * 500 + [1, 1, 1])
    b = fit_categorical_binning(x, y, "f", min_share=0.02)
    assert "Z" not in {bb.level for bb in b.bins}
    assert any(bb.is_other for bb in b.bins)


def test_an_unseen_level_at_score_time_falls_to_other():
    """A level that did not exist at build time must not raise and must not be
    given the base score by accident."""
    x = np.array(["A"] * 500 + ["B"] * 500, dtype=object)
    y = np.array([0] * 400 + [1] * 100 + [0] * 250 + [1] * 250)
    b = fit_categorical_binning(x, y, "f")
    assert b.transform(np.array(["NEW"], dtype=object))[0] == b.other_woe


# --------------------------------------------------------------- hybrid
def test_a_special_code_does_not_sort_below_every_real_value(df):
    """-9 means "no bureau record". Binned as a number it lands in the
    youngest-file bin and inherits its risk, which is an artefact of the
    encoding rather than anything said about the applicant."""
    x = df.credit_age_reported.to_numpy(float)
    y = df.defaulted.to_numpy()
    hybrid = fit_hybrid_binning(x, y, "credit_age_reported", [-9])

    special = [b for b in hybrid.bins if getattr(b, "is_special", False)]
    assert len(special) == 1

    numeric_woes = [b.woe for b in hybrid.numeric.bins]
    assert min(numeric_woes) < special[0].woe < max(numeric_woes), (
        "the no-hit bin sits at an extreme, which is the bug this class exists "
        "to prevent")


def test_the_hybrid_beats_treating_the_whole_column_as_categorical(df):
    """Categorical binning collapses a mostly-continuous column into two bins
    with near-identical WoE -- a card row that cannot affect a decision."""
    x = df.credit_age_reported.to_numpy(float)
    y = df.defaulted.to_numpy()
    cat = fit_categorical_binning(np.asarray(x, dtype=object), y,
                                  "credit_age_reported", special_values=[-9.0])
    hybrid = fit_hybrid_binning(x, y, "credit_age_reported", [-9])
    assert hybrid.iv > cat.iv * 3
    assert len(hybrid.bins) > len(cat.bins)


def test_split_separates_specials_from_quantities():
    numeric_mask, specials = split_special_values(
        np.array([10.0, -9.0, 30.0, np.nan]), [-9])
    assert list(numeric_mask) == [True, False, True, False]
    assert specials.iloc[3] == "__MISSING__"


# ----------------------------------------------------------------- card
def test_one_card_carries_numeric_categorical_and_hybrid_rows(df):
    y = df.defaulted.to_numpy()
    binnings = {
        "dti": fit_binning(df.dti.to_numpy(float), y, "dti"),
        "home_ownership": fit_categorical_binning(
            df.home_ownership.to_numpy(object), y, "home_ownership"),
        "credit_age_reported": fit_hybrid_binning(
            df.credit_age_reported.to_numpy(float), y, "credit_age_reported", [-9]),
    }
    card = Scorecard(binnings, ScorecardConfig()).fit(
        {"dti": df.dti.to_numpy(float),
         "home_ownership": df.home_ownership.to_numpy(object),
         "credit_age_reported": df.credit_age_reported.to_numpy(float)}, y)

    table = card.points_table()
    kinds = {r["feature"] for r in table}
    assert kinds == set(binnings)
    assert all(isinstance(r["bin"], str) for r in table)


def test_the_printed_points_tie_to_the_scored_points(df):
    """The card an examiner holds and the number the system computed have to be
    the same arithmetic, to the decimal."""
    y = df.defaulted.to_numpy()
    binnings = {
        "dti": fit_binning(df.dti.to_numpy(float), y, "dti"),
        "home_ownership": fit_categorical_binning(
            df.home_ownership.to_numpy(object), y, "home_ownership"),
    }
    X = {"dti": df.dti.to_numpy(float),
         "home_ownership": df.home_ownership.to_numpy(object)}
    card = Scorecard(binnings, ScorecardConfig()).fit(X, y)

    row = {"dti": float(df.dti.iloc[7]), "home_ownership": df.home_ownership.iloc[7]}
    table = card.points_table()
    total = 0.0
    for feat in binnings:
        label = binnings[feat].bin_of(row[feat]).label
        total += next(r["points"] for r in table
                      if r["feature"] == feat and r["bin"] == label)
    assert total == pytest.approx(card.score(row), abs=0.15)
