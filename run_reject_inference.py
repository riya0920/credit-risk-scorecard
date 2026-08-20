"""Reject inference, scored against counterfactual truth.

A real lender cannot run the last section of this script. The generator produced
outcomes for rejected applicants that a lender would never observe, so the
inferred bad rates can be checked against what would actually have happened --
which turns "reject inference is approximate" from a caveat into a number.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_scorecard import APPROVAL_RATE, FEATURES, TARGET
from src import reject_inference as ri
from src.generate import generate
from src.scorecard import Scorecard, ScorecardConfig
from src.woe import fit_binning

MULTIPLIERS = (1.0, 1.5, 2.0, 3.0, 4.0)


def fit_card(X_tr, y_tr, names):
    cols = {n: X_tr[:, i] for i, n in enumerate(names)}
    binnings = {n: fit_binning(cols[n], y_tr, n) for n in names}
    card = Scorecard(binnings, ScorecardConfig())
    card.fit(cols, y_tr)
    return card


def main() -> int:
    df = generate()
    funded = df[df.approved_by_incumbent == 1].reset_index(drop=True)
    rejected = df[df.approved_by_incumbent == 0].reset_index(drop=True)

    X_f = funded[FEATURES].to_numpy(float)
    y_f = funded[TARGET].to_numpy()
    X_r = rejected[FEATURES].to_numpy(float)
    y_r_counterfactual = rejected[TARGET].to_numpy()   # NEVER observable in reality

    print("=" * 78)
    print("THE SELECTION PROBLEM")
    print("-" * 78)
    print("applications        : {:,}".format(len(df)))
    print("funded (observable) : {:,}  bad rate {:.3%}".format(
        len(funded), y_f.mean()))
    print("rejected            : {:,}  bad rate {:.3%}  <- NOT OBSERVABLE".format(
        len(rejected), y_r_counterfactual.mean()))
    print("all applicants      : {:,}  bad rate {:.3%}".format(
        len(df), df[TARGET].mean()))
    print("\nThe model is fit on P(default | approved) and deployed to answer")
    print("P(default | applied). Those differ whenever approval was informative,")
    print("which is always -- the incumbent policy was not random. The rejected")
    print("population is not missing at random; it is missing because it looked bad.")

    card = fit_card(X_f, y_f, FEATURES)
    s_f = card.predict_proba({n: X_f[:, i] for i, n in enumerate(FEATURES)})
    s_r = card.predict_proba({n: X_r[:, i] for i, n in enumerate(FEATURES)})

    # ---- how wrong is the funded-only model on the rejects? ---------------
    print("\n" + "=" * 78)
    print("FUNDED-ONLY MODEL, APPLIED TO REJECTS")
    print("-" * 78)
    print("mean predicted PD on funded   : {:.4f}  (actual {:.4f})".format(
        s_f.mean(), y_f.mean()))
    print("mean predicted PD on rejects  : {:.4f}  (actual {:.4f})".format(
        s_r.mean(), y_r_counterfactual.mean()))
    print("\nThe model already ranks rejects as riskier, but it UNDERSTATES how")
    print("much riskier, because it never saw a bad loan from that region of the")
    print("feature space. That understatement is what reject inference tries to")
    print("correct -- and it corrects it with an assumption, not with evidence.")

    # ---- parcelling across multipliers ------------------------------------
    print("\n" + "=" * 78)
    print("PARCELLING: the multiplier IS the assumption")
    print("-" * 78)
    print("{:>12}{:>22}{:>20}{:>16}".format(
        "multiplier", "inferred bad rate", "actual (hidden)", "error"))
    best_k, best_err = None, float("inf")
    for k in MULTIPLIERS:
        inferred = ri.parcelling(s_r, s_f, y_f, bad_rate_multiplier=k)
        res = ri.score_against_counterfactual(inferred, y_r_counterfactual)
        if abs(res["absolute_error"]) < best_err:
            best_k, best_err = k, abs(res["absolute_error"])
        print("{:>12.1f}{:>22.4f}{:>20.4f}{:>16.4f}".format(
            k, res["inferred_bad_rate"], res["actual_bad_rate"],
            res["absolute_error"]))

    print("\nThe multiplier closest to truth on THIS generator is {:.1f}.".format(best_k))
    print("That number is not transferable and must not be read as a recommendation:")
    print("it was found by peeking at outcomes a lender cannot see. In production")
    print("the multiplier is chosen from judgement and challenged in a model")
    print("review, and this table is the argument for why 2x vs 4x is a decision")
    print("worth arguing about -- the inferred bad rate roughly doubles across it.")

    # ---- does inference actually improve the model? -----------------------
    print("\n" + "=" * 78)
    print("DOES IT IMPROVE THE MODEL? (scored on ALL applicants)")
    print("-" * 78)

    X_all = df[FEATURES].to_numpy(float)
    y_all = df[TARGET].to_numpy()
    s_all_funded_only = card.predict_proba(
        {n: X_all[:, i] for i, n in enumerate(FEATURES)})
    auc_funded_only = roc_auc_score(y_all, s_all_funded_only)

    inferred = ri.parcelling(s_r, s_f, y_f, bad_rate_multiplier=2.0)
    rng = np.random.default_rng(7)
    y_r_hard = (rng.random(len(inferred)) < inferred).astype(int)
    X_aug = np.vstack([X_f, X_r])
    y_aug = np.concatenate([y_f, y_r_hard])
    card_aug = fit_card(X_aug, y_aug, FEATURES)
    s_all_aug = card_aug.predict_proba(
        {n: X_all[:, i] for i, n in enumerate(FEATURES)})
    auc_aug = roc_auc_score(y_all, s_all_aug)

    print("{:<34}{:>12}".format("model", "AUC on all applicants"))
    print("{:<34}{:>12.4f}".format("funded-only", auc_funded_only))
    print("{:<34}{:>12.4f}".format("with parcelled rejects (2x)", auc_aug))
    print("{:<34}{:>12.4f}".format("delta", auc_aug - auc_funded_only))

    print("\nRead this carefully. Reject inference cannot create information that")
    print("was never observed -- it propagates an assumption. When the assumption")
    print("is roughly right the model improves on the applicant population; when")
    print("it is wrong the model gets confidently worse, and nothing in the")
    print("training data will tell you which happened.")
    print("\nThe only real fix is a randomised approval holdout: approve a small")
    print("random slice of marginal applicants and observe outcomes. That costs")
    print("money, which is why it is rare -- and why the honest answer is 'we")
    print("bound the bias and buy ground truth where we can afford to'.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
