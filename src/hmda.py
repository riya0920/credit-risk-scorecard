"""Real mortgage applications, with REAL protected attributes.

Every fair-lending number in this project until now carried the same caveat: the
protected attribute was **fabricated by the generator**, correlated with
`region_risk` by construction, so the AIR measured a property of my own code.
The machinery was demonstrated; the finding described nothing.

HMDA fixes that, and it is free and needs no account. Under the Home Mortgage
Disclosure Act every covered lender files loan-level records, and the CFPB
publishes them: **real applications, real decisions, and real
race/ethnicity/sex**. That is the exact input a fair-lending review runs on, and
it is the input this project had been simulating.

    https://ffiec.cfpb.gov/data-browser/

WHAT CHANGES, AND WHAT THAT COSTS. A real AIR on a real population is a claim
about a real market, so it has to be handled more carefully than a synthetic
one -- and several caveats now apply that did not apply to fabricated data:

  THIS IS NOT A CREDIT MODEL'S OUTPUT. HMDA records the lender's DECISION, not
  the score behind it. So an AIR computed here measures the disparity in
  *observed outcomes across all lenders in a state*, which is a market
  statistic, not an audit of one model. Presenting it as the latter would be the
  central error of the whole exercise.

  THE OMITTED-VARIABLE PROBLEM IS REAL AND UNFIXABLE HERE. HMDA has no credit
  score. Since 2018 it carries DTI, LTV and property value, which is far more
  than the pre-2018 data, and it still does not have the single most predictive
  underwriting variable. Any disparity measured here is therefore an upper bound
  on unexplained disparity: some of it is credit risk nobody can see in this
  file. Every serious HMDA study says this, and the ones that do not are the
  ones to distrust.

  "RACE NOT AVAILABLE" IS NOT MISSING AT RANDOM. Applicants may decline to
  report, and reporting rates differ by channel and by group. Dropping those
  rows is a choice with a direction, so the counts are reported rather than
  quietly filtered.

WHAT THIS IS GOOD FOR. Measuring a real disparity, on a real population, with
the actual arithmetic a regulator uses -- and then being precise about which
questions that arithmetic can and cannot answer. That is a better exhibit than a
clean number on data I made up.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "hmda_de_2023.csv"

# HMDA action codes: 1 = originated, 3 = denied. Restricting to these two makes
# the outcome binary and unambiguous. Withdrawn (4) and incomplete (5) are
# NOT denials -- counting them as such is the most common way a HMDA analysis
# manufactures a disparity, because withdrawal rates differ by group for
# reasons that have nothing to do with the lender's decision.
ORIGINATED, DENIED = 1, 3

DTI_MIDPOINTS = {
    "<20%": 15.0, "20%-<30%": 25.0, "30%-<36%": 33.0,
    "50%-60%": 55.0, ">60%": 65.0,
}

AGE_MIDPOINTS = {
    "<25": 22.0, "25-34": 30.0, "35-44": 40.0, "45-54": 50.0,
    "55-64": 60.0, "65-74": 70.0, ">74": 80.0,
}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _dti(series: pd.Series) -> pd.Series:
    """HMDA reports DTI as a number OR a band, in the same column.

    Values below 20 and above 60 are banded to protect privacy, so the column
    is genuinely mixed-type. Coercing straight to numeric silently drops every
    banded row -- which is not a random subset, it is the tails.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    banded = series.astype(str).str.strip().map(DTI_MIDPOINTS)
    return numeric.fillna(banded)


def load(path: Path | str = DATA) -> pd.DataFrame:
    """Loan-level applications with a binary outcome and real demographics."""
    df = pd.read_csv(path, low_memory=False)
    df = df[df.action_taken.isin([ORIGINATED, DENIED])].copy()

    # `denied` is the modelled event, so a higher score means higher risk of
    # denial -- the same orientation as `defaulted` in the synthetic build.
    df["denied"] = (df.action_taken == DENIED).astype(int)

    df["dti"] = _dti(df.debt_to_income_ratio)
    df["ltv"] = _numeric(df.loan_to_value_ratio).clip(0, 200)
    df["income_k"] = _numeric(df.income).clip(0, 5000)
    df["loan_amount_k"] = _numeric(df.loan_amount) / 1000.0
    df["property_value_k"] = _numeric(df.property_value) / 1000.0
    df["age_mid"] = df.applicant_age.astype(str).str.strip().map(AGE_MIDPOINTS)
    df["loan_to_income"] = df.loan_amount_k / df.income_k.replace(0, np.nan)

    for col in ("loan_purpose", "loan_type", "occupancy_type", "lien_status"):
        df[col] = df[col].astype(str)

    return df


def protected(df: pd.DataFrame) -> pd.DataFrame:
    """Real protected attributes, with the reporting gap made explicit.

    `*_reported` marks rows where the applicant actually disclosed. Non-reporting
    is not missing at random -- rates differ by channel and by group -- so the
    analysis reports how many rows it is dropping rather than dropping them
    quietly.
    """
    out = df.copy()
    race = out.derived_race.astype(str)
    eth = out.derived_ethnicity.astype(str)
    sex = out.derived_sex.astype(str)

    out["race_reported"] = ~race.str.contains("Not Available|Free Form", na=False)
    out["eth_reported"] = ~eth.str.contains("Not Available|Free Form", na=False)
    out["sex_reported"] = ~sex.str.contains("Not Available", na=False)

    # Binary cuts, each stated rather than implied. "White vs everyone else" is
    # the crudest possible split and it is the one the 80% rule is usually run
    # on first; the per-group table below is what a real review actually reads.
    out["is_minority"] = (~race.eq("White")) & out.race_reported
    out["is_hispanic"] = eth.eq("Hispanic or Latino")
    out["is_female"] = sex.eq("Female")
    out["race_group"] = race
    return out


def approval_rates(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Approval rate per group, with counts, because a rate without an n is not
    a finding."""
    g = df.groupby(by).agg(
        applications=("denied", "size"),
        denials=("denied", "sum"),
    )
    g["approval_rate"] = 1 - g.denials / g.applications
    best = g.loc[g.applications >= 100, "approval_rate"].max()
    g["air_vs_best"] = g.approval_rate / best
    g["passes_80pc"] = g.air_vs_best >= 0.80
    return g.sort_values("applications", ascending=False)
