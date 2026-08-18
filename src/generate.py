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

    logit = (-2.9
             + 0.034 * dti
             + 0.021 * utilization
             + 0.38 * delinq_2y
             + 0.12 * inquiries_6m
             - 0.0028 * credit_age_months
             - 0.055 * employment_years
             - 0.55 * (np.log(income) - 10.9)
             + 1.5 * region_risk
             + rng.normal(0, 0.35, n))
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

    return pd.DataFrame({
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
