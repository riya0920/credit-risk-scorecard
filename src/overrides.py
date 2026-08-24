"""Override tracking: where a scorecard's real-world performance goes first.

Every validation number in this project -- AUC, KS, PSI, the AIR -- describes
the MODEL. None of them describes the lending. Between the two sits a human
underwriter who can approve below the cutoff and decline above it, and that
layer is unmodelled, undocumented and usually unmeasured.

The vocabulary, because the two terms are easy to swap and the consequences are
opposite:

  LOW-SIDE OVERRIDE    score says decline, human approves. Adds risk the model
                       priced as unacceptable. This is the one that shows up in
                       losses.
  HIGH-SIDE OVERRIDE   score says approve, human declines. Removes business the
                       model wanted. This is the one that shows up in fair
                       lending, because a decline needs an adverse-action reason
                       and "the underwriter didn't like it" is not one.

FOUR THINGS THIS MEASURES THAT A VALIDATION REPORT DOES NOT:

  1. THE EFFECTIVE CUTOFF. The documented cutoff is policy. The cutoff the book
     actually operates at is policy plus overrides, and once the override rate
     passes a few percent the documented number is a fiction. A model validated
     at cutoff 620 and operated at an effective 608 has been validated on a
     population it does not lend to.

  2. WHETHER OVERRIDES ADD OR DESTROY VALUE. A low-side override is a claim that
     the underwriter knows something the model does not. That claim is testable
     once the loans season: compare the override cohort's bad rate against what
     the score predicted for their band. Underwriters being right is not good
     news -- it means the model is missing a real feature and should be
     retrained on it.

  3. DISCRETION DISPARITY. Discretion is where disparate treatment lives. A
     model can be provably fair and the override layer discriminatory, because
     the model is an equation and the override is a person. The override rate by
     protected group is a fair-lending exposure in its own right, and it is
     invisible to every model-level fairness metric in `src/fair_lending.py`.

  4. CONCENTRATION BY UNDERWRITER. Override authority is delegated, and it
     concentrates. One underwriter accounting for a third of low-side overrides
     is a control finding regardless of how those loans perform.

WHY THE SCORE IS STORED AND NOT RECOMPUTED. Each decision records the score, the
cutoff and the model version AS OF THE DECISION. Recomputing an old application
under today's model would let every model change retroactively rewrite override
history -- last quarter's overrides would silently become non-overrides because
the card was recalibrated. The audit question is what the underwriter saw, not
what the current model would have said.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Regulatory guidance does not set a universal number; supervisory exam manuals
# and industry practice treat sustained override rates above ~5% as the point
# where the policy stops describing the book. Declared here rather than buried.
OVERRIDE_RATE_CONCERN = 0.05
LOW_SIDE_CONCERN = 0.03


@dataclass
class Decision:
    """One credit decision, as recorded at the time it was made."""
    app_id: str
    score: float
    cutoff: float
    approved: bool                      # the FINAL decision
    model_version: str = "v1"
    reason: str = ""
    underwriter: str = ""
    group: str = ""                     # protected-class attribute, if collected
    defaulted: int | None = None        # None until the loan seasons
    predicted_pd: float | None = None   # the model's PD at decision time, if kept

    @property
    def model_says_approve(self) -> bool:
        return self.score >= self.cutoff

    @property
    def override_type(self) -> str:
        if self.approved and not self.model_says_approve:
            return "low_side"
        if not self.approved and self.model_says_approve:
            return "high_side"
        return "none"


def frame(decisions: list) -> pd.DataFrame:
    rows = []
    for d in decisions:
        rows.append({
            "app_id": d.app_id, "score": d.score, "cutoff": d.cutoff,
            "approved": d.approved, "model_version": d.model_version,
            "reason": d.reason, "underwriter": d.underwriter, "group": d.group,
            "defaulted": d.defaulted, "predicted_pd": d.predicted_pd,
            "model_approve": d.model_says_approve,
            "override_type": d.override_type,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- rates
def override_report(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"decisions": 0}
    low = int((df.override_type == "low_side").sum())
    high = int((df.override_type == "high_side").sum())
    return {
        "decisions": n,
        "low_side": low,
        "high_side": high,
        "low_side_rate": low / n,
        "high_side_rate": high / n,
        "override_rate": (low + high) / n,
        "exceeds_concern": (low + high) / n > OVERRIDE_RATE_CONCERN,
        "low_side_exceeds_concern": low / n > LOW_SIDE_CONCERN,
    }


def effective_cutoff(df: pd.DataFrame) -> dict:
    """The score at which the book actually approves, versus the documented one.

    Defined as the lowest score at which the approval rate is at least 50%, which
    is the closest thing to "the line" once the line is fuzzy. Compared against
    the policy cutoff -- a gap here means the model was validated on a population
    the lender does not lend to.
    """
    if df.empty:
        return {}
    policy = float(df.cutoff.mode().iloc[0])
    bands = df.assign(band=(df.score // 10) * 10).groupby("band").approved.mean()
    approving = bands[bands >= 0.5]
    effective = float(approving.index.min()) if len(approving) else float("nan")
    return {
        "policy_cutoff": policy,
        "effective_cutoff": effective,
        "drift": effective - policy,
        "approval_rate_below_cutoff": float(
            df[~df.model_approve].approved.mean()) if (~df.model_approve).any() else 0.0,
        "decline_rate_above_cutoff": float(
            (~df[df.model_approve].approved).mean()) if df.model_approve.any() else 0.0,
    }


# --------------------------------------------------------- performance
def override_performance(df: pd.DataFrame, width: float = 20.0) -> pd.DataFrame:
    """Seasoned approvals bucketed by DISTANCE FROM CUTOFF, not by absolute score.

    The banding matters and the obvious choice is wrong. A low-side override is
    below the cutoff by definition and a score-approved loan is at or above it by
    definition, so the two can never share an absolute score band -- the bands
    align to the cutoff and the populations sit on opposite sides of it. Bucketing
    by `score // 20 * 20` produces a comparison with zero overlap and reports
    "nothing to compare", forever.

    Distance from cutoff is the axis that makes them adjacent: an override five
    points below the line sits next to an accept five points above it.
    """
    seasoned = df[df.defaulted.notna() & df.approved].copy()
    if seasoned.empty:
        return pd.DataFrame()
    seasoned["distance"] = seasoned.score - seasoned.cutoff
    seasoned["band"] = (seasoned.distance // width) * width
    return seasoned.groupby(["band", "override_type"]).agg(
        n=("defaulted", "size"), bad_rate=("defaulted", "mean"),
        mean_distance=("distance", "mean")).reset_index()


def override_value(df: pd.DataFrame, width: float = 20.0) -> dict:
    """Did the underwriters know something the model did not?

    Compares low-side overrides just BELOW the cutoff against score-approved
    loans just ABOVE it.

    THIS IS NOT A LIKE-FOR-LIKE COMPARISON AND THE BIAS HAS A KNOWN DIRECTION.
    The override cohort scored lower, so the model's own prediction is that they
    default MORE. That is the null. `expected_gap` reports how many points of
    score separate the two cohorts so the reader can see the tilt rather than
    take the difference at face value.

    Which makes the interpretation counter-intuitive, and it is the reason this
    function exists. Overrides merely MATCHING the higher-scoring cohort already
    means the underwriters are using a real predictor the model does not have.
    The correct response is to find that feature and retrain -- not to
    congratulate the credit team, and not to widen the override policy.

    When `predicted_pd` is recorded, that is used instead: comparing observed
    defaults against what the score itself predicted for the same applicants is
    the clean test, because it needs no comparison cohort at all.
    """
    seasoned = df[df.defaulted.notna() & df.approved].copy()
    if seasoned.empty:
        return {"low_side_n": 0,
                "note": "no seasoned loans at all -- every approval has "
                        "defaulted=None, so nothing here can be evaluated and no "
                        "conclusion is available"}

    low = seasoned[seasoned.override_type == "low_side"]
    if low.empty:
        return {"low_side_n": 0,
                "note": "no seasoned low-side overrides, so there is nothing to "
                        "evaluate and no conclusion is available"}

    out = {"low_side_n": int(len(low)),
           "low_side_bad_rate": float(low.defaulted.mean())}

    # The clean test, when the model's own prediction was recorded.
    if "predicted_pd" in seasoned and low.predicted_pd.notna().all():
        predicted = float(low.predicted_pd.mean())
        out.update(predicted_bad_rate=predicted,
                   lift=out["low_side_bad_rate"] - predicted,
                   basis="model's own predicted PD for these applicants")
    else:
        # Adjacent-band comparison: overrides within `width` below the cutoff
        # against score-approved loans within `width` above it.
        near_low = low[low.score - low.cutoff >= -width]
        above = seasoned[(seasoned.override_type == "none") &
                         (seasoned.score - seasoned.cutoff >= 0) &
                         (seasoned.score - seasoned.cutoff < width)]
        if near_low.empty or above.empty:
            return {"low_side_n": int(len(low)),
                    "note": "no score-approved loans within {:.0f} points above "
                            "the cutoff to compare against, so there is nothing "
                            "to compare and no conclusion is available".format(width)}
        out.update(
            low_side_n=int(len(near_low)),
            low_side_bad_rate=float(near_low.defaulted.mean()),
            comparison_n=int(len(above)),
            score_approved_bad_rate=float(above.defaulted.mean()),
            expected_gap=float((above.score - above.cutoff).mean() -
                               (near_low.score - near_low.cutoff).mean()),
            basis="score-approved loans within {:.0f} points above the "
                  "cutoff".format(width))
        out["lift"] = out["low_side_bad_rate"] - out["score_approved_bad_rate"]

    benchmark = out.get("predicted_bad_rate", out.get("score_approved_bad_rate"))
    if out["lift"] > 0.005:
        out["verdict"] = ("overrides underperform the benchmark -- discretion is "
                          "destroying value")
    elif out["lift"] < -0.005:
        out["verdict"] = ("overrides outperform the benchmark despite scoring "
                          "lower -- the underwriters have a predictor the model "
                          "does not, and the response is to find it and retrain")
    else:
        out["verdict"] = ("overrides match a HIGHER-SCORING cohort, which the "
                          "model predicts should not happen -- weak evidence "
                          "the underwriters have information the card lacks")
    out["benchmark"] = benchmark
    return out


# ----------------------------------------------------------- disparity
def discretion_disparity(df: pd.DataFrame) -> pd.DataFrame:
    """Override rates by protected group.

    This is invisible to every model-level fairness metric, because the model is
    an equation and the override is a person. A card can pass the 80% rule on
    every group and the discretionary layer above it can still be where the
    disparity is.
    """
    if df.empty or not df.group.astype(bool).any():
        return pd.DataFrame()
    # A blank group means the applicant did not report, which is a category with
    # a direction rather than an absence -- reporting rates differ by channel and
    # by group. Labelled rather than dropped, and rather than printed as a blank
    # row that reads like a formatting bug.
    df = df.assign(group=df.group.replace("", "not reported"))
    g = df.groupby("group").agg(
        decisions=("app_id", "size"),
        low_side=("override_type", lambda s: (s == "low_side").sum()),
        high_side=("override_type", lambda s: (s == "high_side").sum()))
    g["low_side_rate"] = g.low_side / g.decisions
    g["high_side_rate"] = g.high_side / g.decisions

    # Benchmarked against the BEST-TREATED group, matching src/fair_lending.py:
    # most favourable = most low-side (approved despite the score), least
    # high-side (declined despite the score).
    best_low = g.low_side_rate.max()
    best_high = g.high_side_rate.min()
    g["low_side_ratio"] = g.low_side_rate / best_low if best_low else np.nan
    g["high_side_ratio"] = (best_high / g.high_side_rate).where(
        g.high_side_rate > 0, 1.0)
    return g.reset_index()


def underwriter_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """Who is using the authority. Delegated discretion concentrates, and one
    underwriter accounting for a third of low-side overrides is a control finding
    regardless of how those loans perform."""
    if df.empty or not df.underwriter.astype(bool).any():
        return pd.DataFrame()
    g = df.groupby("underwriter").agg(
        decisions=("app_id", "size"),
        low_side=("override_type", lambda s: (s == "low_side").sum()),
        high_side=("override_type", lambda s: (s == "high_side").sum()))
    g["low_side_rate"] = g.low_side / g.decisions
    total_low = g.low_side.sum()
    g["share_of_low_side"] = g.low_side / total_low if total_low else 0.0
    return g.sort_values("share_of_low_side", ascending=False).reset_index()


def undocumented(df: pd.DataFrame) -> pd.DataFrame:
    """Overrides with no reason recorded.

    A high-side override without a reason is an adverse action without a
    documented basis, which is an ECOA problem before it is a modelling one. This
    is the cheapest control in the file and the one most often missing.
    """
    o = df[df.override_type != "none"]
    return o[o.reason.fillna("").str.strip() == ""]
