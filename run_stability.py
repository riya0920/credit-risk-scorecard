"""Vintage stability, confidence intervals, and the recommendation re-examined.

run_scorecard.py recommended keeping the scorecard on a +0.009 Gini difference
and a +1.5pp swap-in gap, and its README said in words that those were probably
inside the noise band. This puts intervals on both, which either supports the
recommendation or retires it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_scorecard import APPROVAL_RATE, FEATURES, TARGET, fit_binning
from src import stability
from src.generate import generate
from src.scorecard import Scorecard, ScorecardConfig


def fit_card(X_tr, y_tr, names):
    cols = {n: X_tr[:, i] for i, n in enumerate(names)}
    binnings = {n: fit_binning(cols[n], y_tr, n) for n in names}
    card = Scorecard(binnings, ScorecardConfig())
    card.fit(cols, y_tr)
    return card


def main() -> int:
    df = generate()
    funded = df[df.approved_by_incumbent == 1].reset_index(drop=True)
    split = int(len(funded) * 0.7)
    tr, te = funded.iloc[:split], funded.iloc[split:]

    X_tr, y_tr = tr[FEATURES].to_numpy(float), tr[TARGET].to_numpy()
    X_te, y_te = te[FEATURES].to_numpy(float), te[TARGET].to_numpy()

    card = fit_card(X_tr, y_tr, FEATURES)
    s_card = card.predict_proba({n: X_te[:, i] for i, n in enumerate(FEATURES)})

    import lightgbm as lgb
    mono = {"dti": 1, "utilization": 1, "delinq_2y": 1, "inquiries_6m": 1,
            "credit_age_months": -1, "employment_years": -1, "income": -1}
    gbm = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.06, num_leaves=15,
        min_child_samples=100, reg_lambda=1.0, random_state=0, verbose=-1,
        n_jobs=1, monotone_constraints=[mono[f] for f in FEATURES])
    gbm.fit(X_tr, y_tr)
    s_gbm = gbm.predict_proba(X_te)[:, 1]

    print("=" * 78)
    print("1. DISCRIMINATION, WITH INTERVALS")
    print("-" * 78)
    for name, s in (("scorecard", s_card), ("LightGBM challenger", s_gbm)):
        auc, lo, hi = stability.auc_ci(y_te, s)
        g, glo, ghi = stability.gini_ci(y_te, s)
        print("{:<22} AUC {:.4f} [{:.4f}, {:.4f}]   Gini {:.4f} [{:.4f}, {:.4f}]"
              .format(name, auc, lo, hi, g, glo, ghi))

    print("\n" + "-" * 78)
    diff = stability.paired_auc_difference(y_te, s_card, s_gbm)
    print("PAIRED difference (challenger - champion): {:+.4f} [{:+.4f}, {:+.4f}]"
          .format(diff["difference"], diff["lo"], diff["hi"]))
    print("verdict: {}".format(diff["verdict"]))
    print("\nPaired, not two independent intervals. Two overlapping CIs can still")
    print("hide a significant paired difference, because both models make")
    print("correlated errors on the same applicants -- resampling them together")
    print("is the only comparison that answers 'is B better on this population'.")

    print("\n" + "=" * 78)
    print("2. SWAP SET, WITH AN INTERVAL")
    print("-" * 78)
    swap = stability.swap_set_ci(y_te, s_card, s_gbm, APPROVAL_RATE)
    print("swap-in minus swap-out default rate: {:+.4f} [{:+.4f}, {:+.4f}]".format(
        swap["gap"], swap["lo"], swap["hi"]))
    print("verdict: {}".format(swap["verdict"]))
    print("\nThe swap set is re-derived inside every resample, because both")
    print("thresholds and therefore both group memberships depend on the sample.")
    print("Fixing the groups once and resampling within them would hold constant")
    print("the very thing whose variability is being measured.")

    print("\n" + "=" * 78)
    print("3. THE RECOMMENDATION, RE-EXAMINED")
    print("-" * 78)
    if not diff["significant"] and not swap["significant"]:
        print("Neither the discrimination difference nor the swap-set gap excludes")
        print("zero. The earlier recommendation -- keep the scorecard -- was stated")
        print("with the caveat that the numbers were probably inside the noise")
        print("band, and these intervals confirm it.")
        print("\nThat does NOT make the recommendation wrong. When two models are")
        print("statistically indistinguishable, the tie-break is everything else:")
        print("monotonicity a credit officer can check, additive points an agent")
        print("can read out, and adverse-action reasons that fall out of the")
        print("arithmetic instead of needing a second model to explain the first.")
        print("The scorecard wins on those, and 'we could not tell them apart on")
        print("risk, so we shipped the explainable one' is a defensible sentence")
        print("in a model review. 'The GBM had +0.009 Gini' is not.")
    else:
        print("At least one comparison excludes zero:")
        if diff["significant"]:
            print("  discrimination: {}  ({:+.4f} [{:+.4f}, {:+.4f}])".format(
                diff["verdict"], diff["difference"], diff["lo"], diff["hi"]))
        if swap["significant"]:
            print("  swap set      : {}".format(swap["verdict"]))
        else:
            print("  swap set      : {}".format(swap["verdict"]))

        print("\nThis retires an earlier claim in this repo. run_scorecard.py and")
        print("the README said the two models were indistinguishable and that the")
        print("scorecard should ship because a tie breaks on explainability. The")
        print("paired interval says otherwise: the challenger IS reliably better")
        print("at rank-ordering. Note how easily that was missed -- the two")
        print("individual AUC intervals overlap heavily, and reading those alone")
        print("would have confirmed the comfortable answer.")
        print("\nThe recommendation does not automatically flip, but its")
        print("justification has to change, and the honest version is:")
        print("  * the challenger is better by {:+.4f} AUC, reliably but marginally".format(
            diff["difference"]))
        print("  * the risk it actually trades into the book is indistinguishable")
        print("    ({:+.4f} [{:+.4f}, {:+.4f}] on the swap set)".format(
            swap["gap"], swap["lo"], swap["hi"]))
        print("  * so the question is whether {:.4f} AUC of rank-ordering is worth".format(
            diff["difference"]))
        print("    monotonicity a credit officer can check, additive points an")
        print("    agent can read out, and adverse-action reasons that fall out of")
        print("    the arithmetic rather than needing a second model")
        print("\nThat is a pricing decision for the credit committee, not a")
        print("modelling one -- and it is a much better question than the one the")
        print("earlier point estimates supported.")

    # ---- vintages ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("4. VINTAGE STABILITY")
    print("-" * 78)
    ref = card.predict_proba({n: X_tr[:, i] for i, n in enumerate(FEATURES)})
    n_v = 4
    chunks = np.array_split(np.arange(len(te)), n_v)
    vintages = {"V{}".format(i + 1): s_card[c] for i, c in enumerate(chunks)}

    print("{:<10}{:>8}{:>10}  {}".format("vintage", "n", "score PSI", "band"))
    for row in stability.vintage_report(vintages, ref):
        print("{:<10}{:>8,}{:>10.4f}  {}".format(
            row["vintage"], row["n"], row["psi"], row["band"]))

    print("\n{:<22}{:>10}  {}".format("characteristic", "PSI", "band"))
    for row in stability.characteristic_stability(X_tr, X_te, FEATURES):
        print("{:<22}{:>10.4f}  {}".format(
            row["characteristic"], row["psi"], row["band"]))

    print("\nBands: <0.10 stable | 0.10-0.25 monitor | >0.25 investigate. The")
    print("vintages here are slices of one generated population, so stability is")
    print("the expected result and proves the machinery rather than the model --")
    print("a real vintage split spans quarters of genuinely different applicants.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
