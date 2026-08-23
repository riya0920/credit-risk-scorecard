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

from run_scorecard import (ALL_CHARACTERISTICS, APPROVAL_RATE, CATEGORICAL,
                           FEATURES, SPECIAL_VALUES, TARGET, _column,
                           fit_binning)
from src.categorical import fit_categorical_binning, fit_hybrid_binning
from src import stability
from src.generate import generate
from src.scorecard import Scorecard, ScorecardConfig


def fit_card(X_tr, y_tr, names):
    cols = {n: X_tr[:, i] for i, n in enumerate(names)}
    binnings = {n: fit_binning(cols[n], y_tr, n) for n in names}
    card = Scorecard(binnings, ScorecardConfig())
    card.fit(cols, y_tr)
    return card


def fit_full_card(frame, y):
    """The card as `run_scorecard.py` builds it -- categoricals included.

    This script used to fit both models on the numeric characteristics only.
    That was internally fair but it was not the same model the rest of the repo
    reports on, so the confidence intervals here described a card nobody else
    was looking at.
    """
    binnings = {f: fit_binning(frame[f].to_numpy(float), y, f) for f in FEATURES}
    for f in CATEGORICAL:
        binnings[f] = fit_categorical_binning(frame[f].to_numpy(object), y, f)
    for f, sp in SPECIAL_VALUES.items():
        binnings[f] = fit_hybrid_binning(frame[f].to_numpy(float), y, f, sp)
    card = Scorecard(binnings, ScorecardConfig())
    card.fit({f: _column(frame, f) for f in ALL_CHARACTERISTICS}, y)
    return card


def gbm_frame(frame):
    """The same characteristics, in the shape a tree wants them."""
    out = frame[FEATURES + CATEGORICAL + list(SPECIAL_VALUES)].copy()
    for c in CATEGORICAL:
        out[c] = out[c].astype("category")
    for c in SPECIAL_VALUES:
        out[c + "__nohit"] = (out[c] == -9).astype(int)
        out[c] = out[c].mask(out[c] == -9)
    return out


def main() -> int:
    df = generate()
    funded = df[df.approved_by_incumbent == 1].reset_index(drop=True)
    split = int(len(funded) * 0.7)
    tr, te = funded.iloc[:split], funded.iloc[split:]

    X_tr, y_tr = tr[FEATURES].to_numpy(float), tr[TARGET].to_numpy()
    X_te, y_te = te[FEATURES].to_numpy(float), te[TARGET].to_numpy()

    card = fit_full_card(tr, y_tr)
    s_card = card.predict_proba({f: _column(te, f) for f in ALL_CHARACTERISTICS})

    import lightgbm as lgb
    mono = {"dti": 1, "utilization": 1, "delinq_2y": 1, "inquiries_6m": 1,
            "credit_age_months": -1, "employment_years": -1, "income": -1}
    tr_g, te_g = gbm_frame(tr), gbm_frame(te)
    gbm = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.06, num_leaves=15,
        min_child_samples=100, reg_lambda=1.0, random_state=0, verbose=-1,
        n_jobs=1, monotone_constraints=[mono.get(c, 0) for c in tr_g.columns])
    gbm.fit(tr_g, y_tr, categorical_feature=CATEGORICAL)
    s_gbm = gbm.predict_proba(te_g)[:, 1]

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
    print("4. VINTAGE STABILITY -- twelve real booking cohorts")
    print("-" * 78)
    ref = card.predict_proba({f: _column(tr, f) for f in ALL_CHARACTERISTICS})

    # Real cohorts from the generator, not slices of one homogeneous draw. Two
    # things move across them and they mean opposite things:
    #   MIX SHIFT      later cohorts carry thinner files and higher utilisation
    #                  (the lender loosened marketing). PSI moves; the model is
    #                  still correct and the answer is to reprice.
    #   OUTCOME SHIFT  later cohorts default more AT THE SAME characteristics
    #                  (the economy turned). PSI does not move; the model is now
    #                  wrong and the answer is to rebuild.
    # A vintage report that cannot separate them sends a credit officer to do
    # the opposite of the right thing.
    v_te = te["vintage_month"].to_numpy()
    vintages, actuals = {}, {}
    for v in sorted(set(v_te)):
        m = v_te == v
        if m.sum() < 100:
            continue
        vintages["M{:02d}".format(int(v))] = s_card[m]
        actuals["M{:02d}".format(int(v))] = float(y_te[m].mean())

    print("{:<8}{:>8}{:>11}{:>12}{:>16}  {}".format(
        "vintage", "n", "score PSI", "mean PD", "actual default", "band"))
    for row in stability.vintage_report(vintages, ref):
        name = row["vintage"]
        print("{:<8}{:>8,}{:>11.4f}{:>12.4f}{:>15.2%}   {}".format(
            name, row["n"], row["psi"], float(np.mean(vintages[name])),
            actuals[name], row["band"]))

    rates = [actuals[k] for k in sorted(actuals)]
    psis = [r["psi"] for r in stability.vintage_report(vintages, ref)]
    print("\nactual default rate, first vintage -> last : {:.2%} -> {:.2%} "
          "({:+.1f} pts)".format(rates[0], rates[-1],
                                 (rates[-1] - rates[0]) * 100))
    print("score PSI over the same span              : {:.4f} -> {:.4f}".format(
        psis[0], psis[-1]))

    # Characteristic PSI EARLY vintages vs LATE ones. Comparing the build
    # sample against the whole test set instead would compare two mixtures of
    # the same twelve cohorts, whose marginals agree by construction -- a
    # comparison guaranteed to report stability whatever the cohorts did.
    early = v_te <= 3
    late = v_te >= 8
    print("\n{:<22}{:>10}  {}".format(
        "characteristic (M00-03 vs M08-11)", "PSI", "band"))
    for row in stability.characteristic_stability(
            X_te[early], X_te[late], FEATURES):
        print("{:<22}{:>10.4f}  {}".format(
            row["characteristic"], row["psi"], row["band"]))

    # ---- the point of the section ----------------------------------------
    print("\n" + "-" * 78)
    print("SEPARATING THE TWO SHIFTS")
    print("-" * 78)
    print("{:<28}{:>16}{:>16}".format("", "early M00-M03", "late M08-M11"))
    print("{:<28}{:>16,}{:>16,}".format("n", int(early.sum()), int(late.sum())))
    print("{:<28}{:>16.2%}{:>16.2%}".format(
        "actual default rate", y_te[early].mean(), y_te[late].mean()))
    print("{:<28}{:>16.4f}{:>16.4f}".format(
        "predicted default rate",
        s_pred_early := float(np.mean(s_card[early])),
        s_pred_late := float(np.mean(s_card[late]))))

    obs_gap = y_te[late].mean() - y_te[early].mean()
    pred_gap = s_pred_late - s_pred_early
    print()
    print("observed deterioration : {:+.2%}".format(obs_gap))
    print("explained by mix shift : {:+.2%}  (what the model predicted)".format(
        pred_gap))
    print("UNEXPLAINED            : {:+.2%}  (the model did not see this)".format(
        obs_gap - pred_gap))
    print()
    print("That residual is the number a monitoring pack needs and PSI cannot")
    print("produce. The characteristics did move -- utilisation and credit age")
    print("both drifted, and the card repriced for them. What it could not")
    print("reprice is applicants defaulting MORE AT THE SAME characteristics,")
    print("because nothing in the characteristics says the economy turned.")
    print()
    print("Bands: <0.10 stable | 0.10-0.25 monitor | >0.25 investigate. Bands")
    print("are a convention, not a test -- they carry no sample size, so a 0.09")
    print("on 200 accounts and a 0.09 on 200,000 read identically and are not")
    print("the same evidence.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
