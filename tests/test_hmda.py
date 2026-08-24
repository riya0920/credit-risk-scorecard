"""Real HMDA data: the loading decisions, each of which can silently skew a result."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hmda import DATA, approval_rates, load, protected

pytestmark = pytest.mark.skipif(
    not DATA.exists(),
    reason="HMDA extract not downloaded -- see src/hmda.py for the free URL")


@pytest.fixture(scope="module")
def df():
    return protected(load())


def test_only_originations_and_denials_are_kept(df):
    """Withdrawn (4) and incomplete (5) are NOT denials. Counting them as such
    is the most common way a HMDA analysis manufactures a disparity, because
    withdrawal rates differ by group for reasons that are not the lender's
    decision."""
    assert set(df.action_taken.unique()) <= {1, 3}
    assert set(df.denied.unique()) == {0, 1}


def test_banded_dti_values_are_not_silently_dropped(df):
    """HMDA reports DTI as a number OR a band in the same column -- values
    below 20% and above 60% are banded for privacy. Coercing straight to
    numeric drops exactly the tails, which is the opposite of a random subset.
    """
    assert df.dti.notna().mean() > 0.90
    # The banded extremes must survive into the numeric column.
    assert (df.dti < 20).any(), "the <20% band was lost"
    assert (df.dti > 60).any(), "the >60% band was lost"


def test_non_reporting_is_flagged_rather_than_dropped(df):
    """Reporting rates differ by channel and by group, so dropping those rows
    silently is a choice with a direction."""
    assert "race_reported" in df
    assert 0 < df.race_reported.mean() < 1
    unreported = df[~df.race_reported]
    assert len(unreported) > 0
    assert unreported.derived_race.str.contains(
        "Not Available|Free Form", na=False).all()


def test_the_protected_attributes_are_real_categories(df):
    """The point of the whole file: these came from applicants, not a generator."""
    races = set(df.derived_race.unique())
    assert "White" in races
    assert "Black or African American" in races
    assert df.is_minority.sum() > 1000


def test_approval_rates_compare_against_the_best_group(df):
    """The 80% rule is formulated against the best-performing group -- the only
    reference that needs no defence, because it is the treatment the market
    demonstrably can give."""
    rates = approval_rates(df[df.race_reported], "race_group")
    big = rates[rates.applications >= 100]
    assert np.isclose(big.air_vs_best.max(), 1.0)
    assert (big.air_vs_best <= 1.0 + 1e-9).all()


def test_thin_groups_do_not_set_the_benchmark(df):
    """A 12-applicant group at 100% approval would otherwise become the
    reference and fail everybody else against noise."""
    rates = approval_rates(df[df.race_reported], "race_group")
    best_row = rates.loc[rates.air_vs_best.idxmax()]
    assert best_row.applications >= 100


def test_the_aggregate_can_pass_while_a_group_fails(df):
    """The finding this dataset exists to make: rolling every non-White
    applicant into one bucket averages the disparity away.

    If this ever stops holding, the headline claim in
    docs/HMDA_FAIR_LENDING.md has stopped being true and needs rewriting.
    """
    from src.fair_lending import adverse_impact_ratio

    reported = df.race_reported.to_numpy()
    approved = (df.denied == 0).to_numpy()[reported]
    minority = df.is_minority.to_numpy().astype(int)[reported]

    aggregate = adverse_impact_ratio(approved, minority)["air"]
    rates = approval_rates(df[df.race_reported], "race_group")
    failing = rates[(rates.applications >= 50) & (~rates.passes_80pc)]

    assert aggregate >= 0.80, "aggregate no longer passes"
    assert len(failing) >= 1, "no individual group fails any more"


def test_income_and_ltv_are_bounded_not_left_raw(df):
    """HMDA carries sentinel-looking extremes; an unbounded LTV of 9999 drags
    every model fitted on it."""
    assert df.ltv.dropna().max() <= 200
    assert df.income_k.dropna().max() <= 5000
