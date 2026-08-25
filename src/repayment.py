"""A real repayment outcome, and the fairness question HMDA could not answer.

Every scorecard in this project has been fitted against `denied` -- what an
underwriter concluded -- because HMDA records decisions and never records
whether the loan was repaid. Those are different targets:

    denied    what an underwriter concluded about an applicant
    default   what the applicant subsequently did

A card fitted on the first learns to imitate the underwriter, including their
mistakes. A card fitted on the second learns credit risk.

THE QUESTION THIS UNLOCKS, and it is the one that matters. With a decision
target you can measure whether groups are treated differently. You cannot
measure whether they are treated CORRECTLY, because there is nothing to be
correct about -- the target IS the treatment. Given a real outcome, the two come
apart:

    DISPARITY      does the model decline one group more?
    CALIBRATION    when it assigns a group a probability of default, is that
                   probability right for them?

A model can fail the first and pass the second, and that combination is the
whole difficulty of fair lending. If a group genuinely defaults more, a
well-calibrated model declines them more, and the AIR falls below 0.80 without
anything in the model being wrong about them. Equalising the AIR then requires
lending to the higher-risk group at a price that does not cover their risk --
which is a policy choice somebody should make deliberately, not a bug to be
patched out of a model.

`docs/HMDA_FAIR_LENDING.md` could state that tension and could not measure it.
This can.

THE UNDOCUMENTED CODES. The published data dictionary says EDUCATION takes 1-4
and MARRIAGE 1-3. The file contains EDUCATION in {0..6} and MARRIAGE in {0..3}.
Codes 0, 5 and 6 are undocumented, and this is exactly the case
`src/categorical.py` was written for: they get their own levels rather than
being folded into "others", because a code nobody documented is a fact about the
data-collection process and folding it away destroys the only evidence of it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "credit_default.parquet"

# Verified against the data rather than taken on trust from the documentation:
# x1 spans 10,000-1,000,000 across 81 values (a credit limit), x2 is binary,
# x5 runs 21-79 (an age), x6 runs -2..8 (a repayment status), x12 goes negative
# (a bill amount), and the resulting default rate is 22.12% -- which matches the
# published figure for this study exactly, and is what confirms the mapping.
COLUMNS = {
    "x1": "limit_bal", "x2": "sex", "x3": "education", "x4": "marriage",
    "x5": "age",
    "x6": "pay_0", "x7": "pay_2", "x8": "pay_3", "x9": "pay_4",
    "x10": "pay_5", "x11": "pay_6",
    "x12": "bill_1", "x13": "bill_2", "x14": "bill_3", "x15": "bill_4",
    "x16": "bill_5", "x17": "bill_6",
    "x18": "pay_amt_1", "x19": "pay_amt_2", "x20": "pay_amt_3",
    "x21": "pay_amt_4", "x22": "pay_amt_5", "x23": "pay_amt_6",
    "y": "default",
}

SEX = {1: "male", 2: "female"}
EDUCATION = {1: "graduate", 2: "university", 3: "high_school", 4: "other",
             0: "undocumented_0", 5: "undocumented_5", 6: "undocumented_6"}
MARRIAGE = {1: "married", 2: "single", 3: "other", 0: "undocumented_0"}


def load(path: Path | str = DATA) -> pd.DataFrame:
    df = pd.read_parquet(path).rename(columns=COLUMNS)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["sex_label"] = df.sex.map(SEX)
    # `.map` on an unlisted code yields NaN, which would silently drop the row
    # later. The undocumented codes are mapped explicitly and anything still
    # unmapped is labelled rather than lost.
    df["education_label"] = df.education.map(EDUCATION).fillna("unmapped")
    df["marriage_label"] = df.marriage.map(MARRIAGE).fillna("unmapped")

    # Utilisation, which is the feature this data actually has and mortgage
    # data does not. Guarded because limit_bal is never zero here but a divide
    # that only works on today's file is a divide waiting to fail.
    df["utilisation"] = np.where(df.limit_bal > 0,
                                 df.bill_1 / df.limit_bal, np.nan)
    df["worst_delinquency"] = df[["pay_0", "pay_2", "pay_3", "pay_4",
                                  "pay_5", "pay_6"]].max(axis=1)
    df["months_delinquent"] = (df[["pay_0", "pay_2", "pay_3", "pay_4",
                                   "pay_5", "pay_6"]] > 0).sum(axis=1)
    return df


def undocumented_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Levels the published data dictionary does not list.

    Reported rather than merged. A code nobody documented is a fact about the
    data-collection process, and folding it into "other" destroys the only
    evidence that it happened.
    """
    rows = []
    for col, documented in (("education_label", {"graduate", "university",
                                                 "high_school", "other"}),
                            ("marriage_label", {"married", "single", "other"})):
        counts = df[col].value_counts()
        for level, n in counts.items():
            if level not in documented:
                rows.append({"column": col, "level": level, "n": int(n),
                             "share": n / len(df),
                             "default_rate": float(
                                 df.loc[df[col] == level, "default"].mean())})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def outcome_rates(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Actual default rate by group. The thing HMDA cannot produce."""
    g = df.groupby(by).agg(n=("default", "size"),
                           defaults=("default", "sum"),
                           default_rate=("default", "mean"))
    return g.reset_index()


def calibration_by_group(df: pd.DataFrame, predicted: np.ndarray, by: str,
                         bins: int = 10) -> pd.DataFrame:
    """Predicted probability against realised default, per group.

    The question a decision target cannot pose. Disparity asks whether a group
    is declined more; calibration asks whether the probability assigned to them
    is RIGHT for them. A model can fail the first and pass the second, and that
    combination is the whole difficulty.
    """
    work = df.copy()
    work["predicted"] = predicted

    # A DEGENERATE PREDICTION MUST NOT PRODUCE AN EMPTY ANSWER, and this was a
    # real bug rather than defensive padding. `pd.qcut` on a constant vector
    # returns all-NaN even with duplicates="drop", the groupby then discards
    # every row, and calibration comes back as an empty frame -- which reads as
    # "nothing to report" when it means "the model outputs one number".
    #
    # A constant prediction is exactly what a broken model produces, so silence
    # here is silence at the moment the check is most needed. One band is the
    # honest answer: it says the model does not discriminate between accounts,
    # and the gap in that single band still measures whether its one number is
    # right.
    if work.predicted.nunique() < 2:
        work["band"] = 0
    else:
        work["band"] = pd.qcut(work.predicted, bins, labels=False,
                               duplicates="drop")
        if work.band.isna().all():
            work["band"] = 0
    work = work[work.band.notna()]

    out = work.groupby([by, "band"]).agg(
        n=("default", "size"),
        predicted=("predicted", "mean"),
        observed=("default", "mean")).reset_index()
    out["gap"] = out.observed - out.predicted
    return out


def calibration_error(cal: pd.DataFrame, by: str) -> pd.DataFrame:
    """Population-weighted mean absolute gap, per group.

    Weighted, because an unweighted average over bands lets a band holding
    forty accounts count as much as one holding four thousand.
    """
    rows = []
    for g, sub in cal.groupby(by):
        w = sub.n / sub.n.sum()
        rows.append({by: g, "n": int(sub.n.sum()),
                     "weighted_abs_gap": float((w * sub.gap.abs()).sum()),
                     "signed_gap": float((w * sub.gap).sum())})
    return pd.DataFrame(rows)
