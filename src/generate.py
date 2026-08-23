"""Synthetic loan-application data with a documented selection bias.

The population generated here is FUNDED loans only -- the same bias every public
credit dataset (Lending Club included) carries. Applications that were declined
have no repayment outcome, so they are absent from training, which means the
model is fit on a population the scoring population does not match. That is
reject inference, and it cannot be fully fixed; see docs/README 'reject
inference'. Naming it is the point.

To make the bias inspectable rather than theoretical, this generator produces
BOTH the funded population and the rejected one (with their counterfactual
outcomes, which a lender never observes). Training uses only the funded rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate(n: int = 60_000, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    income = np.exp(rng.normal(10.9, 0.55, n)).clip(12_000, 400_000)
    employment_years = rng.gamma(2.2, 2.6, n).clip(0, 40)
    loan_amount = (income * rng.uniform(0.08, 0.65, n)).clip(1_000, 60_000)
    dti = (loan_amount / income * 100 + rng.normal(12, 6, n)).clip(1, 75)
    utilization = rng.beta(2.2, 3.0, n) * 100
    delinq_2y = rng.poisson(0.28, n).clip(0, 8)
    inquiries_6m = rng.poisson(0.9, n).clip(0, 12)
    credit_age_months = rng.gamma(5.0, 26.0, n).clip(6, 500)

    # A geography feature that is NOT a protected attribute but correlates with
    # one -- the proxy problem, made concrete.
    region_risk = rng.beta(2, 5, n)

    # ---------------------------------------------------------------- vintages
    # Twelve monthly booking cohorts with a REAL population shift across them,
    # not slices of one homogeneous draw. Two things move and they move for
    # different reasons, which is what makes the stability report worth reading:
    #
    #   MIX SHIFT      later vintages carry more thin-file, higher-utilisation
    #                  applicants, because the lender loosened marketing. PSI on
    #                  the characteristics moves and the model is still correct.
    #   OUTCOME SHIFT  later vintages default more at the SAME characteristics,
    #                  because the economy turned. PSI does not move and the
    #                  model is now wrong.
    #
    # A vintage report that cannot separate those two tells a credit officer to
    # rebuild the scorecard when the right answer is to reprice, or the reverse.
    vintage = rng.integers(0, 12, n)
    drift = vintage / 11.0

    utilization = (utilization + 14.0 * drift).clip(0, 100)
    credit_age_months = (credit_age_months * (1.0 - 0.30 * drift)).clip(6, 500)
    macro_shock = 0.85 * drift          # outcome shift, invisible to PSI

    # -------------------------------------------------------- categoricals
    # Real categorical characteristics, so the card actually carries them:
    # a nominal one with a rare level, one with genuine missingness, and one
    # that arrives with a special value a naive pipeline would average.
    home = rng.choice(["OWN", "MORTGAGE", "RENT", "OTHER"], n,
                      p=[0.17, 0.42, 0.38, 0.03])
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "medical",
         "small_business", "wedding"], n,
        p=[0.42, 0.25, 0.14, 0.09, 0.07, 0.03])
    employment_type = rng.choice(["W2", "SELF_EMPLOYED", "CONTRACT", "RETIRED"],
                                 n, p=[0.63, 0.18, 0.13, 0.06])
    # Missingness that is NOT at random: self-employed applicants are far more
    # likely to have no verified employment record, and self-employment is
    # itself a risk factor. Imputing the mode here imports the W2 default rate
    # onto the riskiest slice of the book.
    missing_p = np.where(employment_type == "SELF_EMPLOYED", 0.34, 0.04)
    employment_type = np.where(rng.random(n) < missing_p, None, employment_type)

    home_effect = np.select(
        [home == "OWN", home == "MORTGAGE", home == "RENT"],
        [-0.34, -0.12, 0.22], default=0.30)
    purpose_effect = np.select(
        [purpose == "small_business", purpose == "medical",
         purpose == "debt_consolidation", purpose == "wedding"],
        [0.55, 0.30, 0.10, 0.20], default=-0.08)
    # The missing-employment slice carries its own risk, above and beyond the
    # self-employment it mostly stands for. `__MISSING__` is a level, not a hole.
    emp_effect = np.where(
        pd.isna(employment_type), 0.42,
        np.select([employment_type == "SELF_EMPLOYED",
                   employment_type == "CONTRACT"],
                  [0.28, 0.15], default=0.0))

    logit = (-2.9
             + macro_shock
             + home_effect
             + purpose_effect
             + emp_effect
             + 0.034 * dti
             + 0.021 * utilization
             + 0.38 * delinq_2y
             + 0.12 * inquiries_6m
             - 0.0028 * credit_age_months
             - 0.055 * employment_years
             - 0.55 * (np.log(income) - 10.9)
             + 1.5 * region_risk
             + rng.normal(0, 0.35, n))
    # Hold the book's base rate fixed while the new characteristics are added.
    # Without this the categoricals and the macro shock raise the overall
    # default rate from 41% to 57%, and every downstream number -- the points
    # card, the swap set, the reject-inference gap -- moves for a reason that
    # has nothing to do with the features being added. A generator change that
    # silently reprices the whole book makes the before/after incomparable.
    #
    # The intercept is solved for, not tuned by hand, and the target is the rate
    # the previous generator produced.
    TARGET_BASE_RATE = 0.41
    lo, hi = -12.0, 12.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logit + mid)))).mean() > TARGET_BASE_RATE:
            hi = mid
        else:
            lo = mid
    logit = logit + (lo + hi) / 2

    p_default = 1 / (1 + np.exp(-logit))
    defaulted = (rng.random(n) < p_default).astype(int)

    # Protected attribute, synthetic, correlated with region_risk. Documented as
    # SYNTHETIC everywhere it is used: no conclusion drawn from it describes the
    # real world.
    p_group = 0.25 + 0.45 * region_risk
    group_a = (rng.random(n) < p_group).astype(int)

    # The incumbent underwriting policy that decided who got funded. This is the
    # source of the selection bias.
    approved = ((dti < 45) & (delinq_2y <= 2) & (utilization < 85)
                & (income > 20_000)).astype(int)

    # A special value, not a measurement: -9 is the bureau's "no hit" code and
    # a pipeline that averages it produces a negative credit age.
    credit_age_reported = np.where(rng.random(n) < 0.035, -9.0,
                                   credit_age_months.round(0))

    return pd.DataFrame({
        "vintage_month": vintage,
        "home_ownership": home,
        "loan_purpose": purpose,
        "employment_type": employment_type,
        "credit_age_reported": credit_age_reported,
        "income": income.round(0),
        "dti": dti.round(2),
        "utilization": utilization.round(2),
        "delinq_2y": delinq_2y,
        "inquiries_6m": inquiries_6m,
        "credit_age_months": credit_age_months.round(0),
        "employment_years": employment_years.round(1),
        "loan_amount": loan_amount.round(0),
        "region_risk": region_risk.round(4),
        "protected_group_synthetic": group_a,
        "approved_by_incumbent": approved,
        "defaulted": defaulted,
    })


if __name__ == "__main__":
    import pathlib
    out = pathlib.Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    df = generate()
    df.to_parquet(out / "applications.parquet")
    print("rows {:,}  funded {:,}  overall default {:.3%}  funded default {:.3%}".format(
        len(df), int(df.approved_by_incumbent.sum()), df.defaulted.mean(),
        df[df.approved_by_incumbent == 1].defaulted.mean()))
    print()
    print("vintage  n      util mean  credit age  default rate")
    for v, g in df.groupby("vintage_month"):
        print("{:>7}  {:<6} {:>9.1f} {:>11.0f} {:>13.3%}".format(
            v, len(g), g.utilization.mean(), g.credit_age_months.mean(),
            g.defaulted.mean()))
    print()
    print("employment_type missing: {:.1%} overall, {:.1%} among the self-employed"
          .format(df.employment_type.isna().mean(),
                  df.loc[df.employment_type.isna()].shape[0]
                  / max((df.employment_type.isna()
                         | (df.employment_type == "SELF_EMPLOYED")).sum(), 1)))
    print("credit_age_reported == -9 (bureau no-hit): {:.1%}".format(
        (df.credit_age_reported == -9).mean()))
