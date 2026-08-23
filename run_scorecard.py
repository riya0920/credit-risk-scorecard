"""Champion scorecard vs GBM challenger, with the swap-set analysis.

The swap set is the artifact a credit team actually argues over: not "which model
has higher AUC" but "who specifically does the new model approve that the old one
declined, and are those people better or worse risk than the ones it stops
approving." A challenger with +2 Gini that swaps in a worse-performing population
loses that meeting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.generate import generate
from src.scorecard import Scorecard, ScorecardConfig
from src.categorical import fit_categorical_binning, fit_hybrid_binning
from src.woe import fit_binning, iv_band


def _column(df, feature):
    """Numeric characteristics as floats, categorical ones as objects.

    `to_numpy(float)` on a column holding None produces nan and silently
    collapses the missing LEVEL into a numeric hole, which is the bug this
    project exists to avoid making.
    """
    if feature in CATEGORICAL:
        return df[feature].to_numpy(object)
    return df[feature].to_numpy(float)

FEATURES = ["dti", "utilization", "delinq_2y", "inquiries_6m",
            "credit_age_months", "employment_years", "income"]

# Categorical characteristics carried by the SAME card, not a parallel model.
# `home_ownership` is ordinary; `loan_purpose` has a rare level that has to be
# merged rather than given its own points; `employment_type` is missing for
# 9.4% of applicants and that missingness is NOT at random -- it concentrates in
# the self-employed, who default more. `__MISSING__` is therefore a LEVEL with
# its own WoE, not a hole to impute, and imputing the mode here would import the
# W2 default rate onto the riskiest slice of the book.
CATEGORICAL = ["home_ownership", "loan_purpose", "employment_type"]

# A special value, not a measurement: -9 is the bureau's "no hit" code. Binned
# on its own so it cannot be averaged into a credit age.
SPECIAL_VALUES = {"credit_age_reported": [-9]}
ALL_CHARACTERISTICS = FEATURES + CATEGORICAL + list(SPECIAL_VALUES)
TARGET = "defaulted"
APPROVAL_RATE = 0.80          # policy: approve the best 80% of applicants


def ks(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr - fpr))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def calibration(y, p, bins=8):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    idx = np.digitize(p, q[1:-1])
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            rows.append((int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return rows


def adverse_impact_ratio(approved: np.ndarray, group: np.ndarray) -> dict:
    """Selection rate of the lower-rate group / the higher-rate group.

    The '80% rule' (EEOC Uniform Guidelines, borrowed into fair-lending practice)
    treats a ratio below 0.80 as a screen worth investigating -- NOT as proof of
    discrimination and NOT as a safe harbour above it. Disparate impact is about
    effect; disparate treatment is about the use of the attribute itself. This
    computes the first and says nothing about the second.
    """
    r1 = float(approved[group == 1].mean())
    r0 = float(approved[group == 0].mean())
    lo, hi = min(r1, r0), max(r1, r0)
    return {"rate_group_1": r1, "rate_group_0": r0,
            "adverse_impact_ratio": lo / hi if hi else float("nan"),
            "flags_80_pct_rule": (lo / hi if hi else 1.0) < 0.80}


def main() -> None:
    df = generate()
    funded = df[df.approved_by_incumbent == 1].reset_index(drop=True)
    split = int(len(funded) * 0.7)
    tr, te = funded.iloc[:split], funded.iloc[split:]

    print("=" * 76)
    print("POPULATION")
    print("-" * 76)
    print("applications generated : {:,}".format(len(df)))
    print("funded (trainable)     : {:,} ({:.1%})".format(
        len(funded), len(funded) / len(df)))
    print("default rate, funded   : {:.3%}".format(funded[TARGET].mean()))
    print("default rate, ALL      : {:.3%}  <- unobservable to a real lender".format(
        df[TARGET].mean()))
    print("\nThe model is fit on funded loans only. The scoring population is all")
    print("applicants. That gap is reject inference; it is not fixed here, and any")
    print("number below inherits it.")

    y_tr = tr[TARGET].to_numpy()
    y_te = te[TARGET].to_numpy()

    # ---- champion: WoE scorecard ------------------------------------------
    binnings = {f: fit_binning(tr[f].to_numpy(float), y_tr, f) for f in FEATURES}
    for f in CATEGORICAL:
        binnings[f] = fit_categorical_binning(tr[f].to_numpy(object), y_tr, f)
    for f, specials in SPECIAL_VALUES.items():
        binnings[f] = fit_hybrid_binning(
            tr[f].to_numpy(float), y_tr, f, special_values=specials)
    print("\n" + "=" * 76)
    print("CHAMPION: WoE SCORECARD -- information value")
    print("-" * 76)
    print("{:<22}{:>12}{:>8}{:>8}{:>18}".format(
        "characteristic", "kind", "IV", "bins", "band"))
    for f in ALL_CHARACTERISTICS:
        b = binnings[f]
        kind = ("numeric" if f in FEATURES
                else "categorical" if f in CATEGORICAL else "hybrid")
        note = iv_band(b.iv)
        if f in FEATURES and not b.monotonic:
            note += " NON-MONOTONIC"
        print("{:<22}{:>12}{:>8.4f}{:>8}{:>18}".format(
            f, kind, b.iv, len(b.bins), note))

    card = Scorecard(binnings, ScorecardConfig(base_score=600, base_odds=20, pdo=20))
    card.fit({f: _column(tr, f) for f in ALL_CHARACTERISTICS}, y_tr)

    X_te = {f: _column(te, f) for f in ALL_CHARACTERISTICS}
    p_card = card.predict_proba(X_te)
    s_card = card.scores(X_te)

    print("\nscaling: base {} at {}:1 odds, PDO {}  ->  factor {:.3f}, offset {:.3f}"
          .format(card.cfg.base_score, card.cfg.base_odds, card.cfg.pdo,
                  card.cfg.factor, card.cfg.offset))
    print("score range on test: {:.0f} to {:.0f}, mean {:.0f}".format(
        s_card.min(), s_card.max(), s_card.mean()))

    print("\n" + "-" * 76)
    print("SCORECARD (points per attribute)")
    print("-" * 76)
    print("{:<20}{:<24}{:>8}{:>10}{:>9}".format("feature", "bin", "n", "bad_rate", "points"))
    for r in card.points_table():
        print("{:<20}{:<24}{:>8}{:>10.4f}{:>9.1f}".format(
            r["feature"], r["bin"], r["n"], r["bad_rate"], r["points"]))

    # ---- challenger: GBM ---------------------------------------------------
    mono = {"dti": 1, "utilization": 1, "delinq_2y": 1, "inquiries_6m": 1,
            "credit_age_months": -1, "employment_years": -1, "income": -1}
    # The challenger gets the SAME characteristics as the card, categoricals
    # included. It did not, at first, and the card came out ahead by 0.013 AUC
    # -- a result that would have read as "the interpretable model wins" when
    # what actually happened is that the challenger was handicapped by four
    # characteristics it was never shown. A bakeoff where the two models see
    # different data measures the feature list, not the model.
    #
    # HistGradientBoosting takes categoricals natively as a pandas category
    # dtype, which also means missing stays missing rather than becoming a
    # one-hot column of zeros indistinguishable from "not this level".
    gbm_cols = FEATURES + CATEGORICAL + list(SPECIAL_VALUES)

    def _gbm_frame(frame):
        out = frame[gbm_cols].copy()
        for c in CATEGORICAL:
            out[c] = out[c].astype("category")
        # -9 is a code, not a quantity: hand the tree the indicator separately
        # and leave the numeric column missing, so it cannot split on -9 as an
        # age.
        for c in SPECIAL_VALUES:
            out[c + "__nohit"] = (out[c] == -9).astype(int)
            out[c] = out[c].mask(out[c] == -9)
        return out

    tr_g, te_g = _gbm_frame(tr), _gbm_frame(te)
    mono_vec = [mono.get(c, 0) for c in tr_g.columns]
    gbm = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=100,
        l2_regularization=1.0, random_state=0,
        categorical_features="from_dtype",
        monotonic_cst=mono_vec)
    gbm.fit(tr_g, y_tr)
    p_gbm = gbm.predict_proba(te_g)[:, 1]

    print("\n" + "=" * 76)
    print("CHAMPION vs CHALLENGER (out-of-sample)")
    print("-" * 76)
    print("{:<26}{:>10}{:>10}{:>10}{:>12}".format("model", "AUC", "Gini", "KS", "Brier"))
    for name, p in [("scorecard (WoE+logit)", p_card), ("GBM (monotonic)", p_gbm)]:
        auc = roc_auc_score(y_te, p)
        print("{:<26}{:>10.4f}{:>10.4f}{:>10.4f}{:>12.5f}".format(
            name, auc, 2 * auc - 1, ks(y_te, p), brier(y_te, p)))

    print("\nBoth models see the same {} characteristics.".format(len(gbm_cols)))
    print("Monotonic constraints on the GBM: {}".format(
        ", ".join("{} {}".format(f, "+" if mono[f] > 0 else "-") for f in FEATURES)))
    print("The categoricals are unconstrained -- there is no domain direction to")
    print("impose on a nominal characteristic, and inventing one would be a")
    print("constraint with no argument behind it.")
    print("Domain says risk rises with DTI/utilisation/delinquency and falls with")
    print("credit age/tenure/income. An unconstrained GBM that disagrees in a thin")
    print("region is fitting noise, and it produces reason codes that read as absurd.")

    print("\n" + "-" * 76)
    print("CALIBRATION (a lender prices with these numbers)")
    print("-" * 76)
    print("{:>6}{:>10}{:>16}{:>16}{:>16}".format(
        "decile", "n", "card predicted", "gbm predicted", "observed"))
    cal_c = calibration(y_te, p_card)
    cal_g = calibration(y_te, p_gbm)
    for i, ((n, pc, obs), (_, pg, obs_g)) in enumerate(zip(cal_c, cal_g)):
        print("{:>6}{:>10,}{:>16.4f}{:>16.4f}{:>16.4f}".format(i, n, pc, pg, obs))

    # ---- swap set ----------------------------------------------------------
    thr_c = float(np.quantile(p_card, APPROVAL_RATE))
    thr_g = float(np.quantile(p_gbm, APPROVAL_RATE))
    app_c = p_card <= thr_c
    app_g = p_gbm <= thr_g
    swap_in = app_g & ~app_c            # challenger approves, champion declines
    swap_out = app_c & ~app_g           # champion approves, challenger declines
    both = app_c & app_g

    print("\n" + "=" * 76)
    print("SWAP-SET ANALYSIS at a {:.0%} approval rate".format(APPROVAL_RATE))
    print("-" * 76)
    print("{:<34}{:>10}{:>16}".format("population", "n", "default rate"))
    for label, m in [("approved by both", both),
                     ("SWAP-IN  (GBM yes, card no)", swap_in),
                     ("SWAP-OUT (card yes, GBM no)", swap_out)]:
        n = int(m.sum())
        rate = float(y_te[m].mean()) if n else float("nan")
        print("{:<34}{:>10,}{:>16.4f}".format(label, n, rate))

    if swap_in.sum() and swap_out.sum():
        delta = float(y_te[swap_in].mean() - y_te[swap_out].mean())
        print("\nswap-in default rate minus swap-out: {:+.4f}".format(delta))
        print("Reading: the challenger is {} risk into the book at the same".format(
            "trading BETTER" if delta < 0 else "trading WORSE"))
        print("approval rate. This, not the Gini delta, is what the credit committee")
        print("votes on.")

    # ---- fairness screen (the only fairness piece built) -------------------
    grp = te.protected_group_synthetic.to_numpy()
    print("\n" + "=" * 76)
    print("ADVERSE IMPACT RATIO -- SYNTHETIC attribute, screen only")
    print("-" * 76)
    for name, m in [("scorecard", app_c), ("GBM", app_g)]:
        air = adverse_impact_ratio(m, grp)
        print("{:<12} group1 {:.4f}  group0 {:.4f}  AIR {:.4f}  {}".format(
            name, air["rate_group_1"], air["rate_group_0"],
            air["adverse_impact_ratio"],
            "FLAGS 80% RULE" if air["flags_80_pct_rule"] else "above 0.80"))
    print("\nThe protected attribute is SYNTHETIC and correlated with region_risk by")
    print("construction. These ratios describe the generator, not any real")
    print("population, and no conclusion about real-world fairness follows from")
    print("them. What the screen demonstrates is the mechanic: disparate IMPACT is")
    print("measured on outcomes; disparate TREATMENT is about using the attribute,")
    print("and neither model here uses it as an input.")

    # ---- adverse action ----------------------------------------------------
    declined = np.argsort(-p_card)[:2]
    print("\n" + "=" * 76)
    print("ADVERSE ACTION -- points-lost reasons (the artifact handed to a regulator)")
    print("-" * 76)
    for i in declined:
        row = {f: (float(te.iloc[i][f]) if f in FEATURES else te.iloc[i][f])
               for f in ALL_CHARACTERISTICS}
        print("\napplicant #{}  score {:.0f}  PD {:.4f}  -> DECLINE".format(
            int(te.index[i]), card.score(row), float(p_card[i])))
        for rc in card.reason_codes(row):
            print("   {}. {:<20} bin {:<22} lost {:>6.1f} pts (earned {:.1f} of {:.1f})"
                  .format(rc["rank"], rc["feature"], rc["applicant_bin"],
                          rc["points_lost"], rc["points_earned"], rc["points_available"]))
    print("=" * 76)


if __name__ == "__main__":
    main()
