# ML-3 — Credit Risk Scorecard with Fair-Lending Analysis

**Status: ~20% slice.** The scorecard machinery, the challenger comparison, and
the swap-set analysis are built. The fair-lending work is a single screen, not
the analysis the spec asks for — see below.

```bash
python run_scorecard.py
```

## What is built

- **WoE binning with enforced monotonicity + IV** (`src/woe.py`), with the sign
  convention fixed in one place and stated, because half of scorecard bugs are
  sign errors. IV bands labelled, including the `>0.50 → check for leakage` band.
- **Points-scaled scorecard** (`src/scorecard.py`): base 600 at 20:1 odds,
  PDO 20 → factor 28.854, offset 513.561, with the derivation written out. The
  full points table prints, one row per attribute.
- **Adverse action by the points-lost method** — exact, not an explainer model
  bolted on afterwards. The reason codes tie to the printed card to the decimal
  (utilization bin `(62.11, inf]` earns 53.1 points in both places); an earlier
  version omitted the intercept allocation from the reason codes, so they cited
  points nobody could find on the card.
- **Monotonic-constrained GBM challenger**, with the domain direction declared
  per feature.
- **Calibration compared side by side**, because a lender prices on PDs.
- **Swap-set analysis** at a fixed approval rate.
- **Documented selection bias**: the generator produces both funded and rejected
  applications; training uses funded only, and the unobservable all-applicant
  default rate is printed next to the funded one to make the gap concrete.

## Results (current build, synthetic data)

| model | AUC | Gini | KS | Brier |
|---|---|---|---|---|
| scorecard (WoE + logistic) | 0.6544 | 0.3088 | 0.2345 | 0.19158 |
| GBM (monotonic) | 0.6588 | 0.3176 | 0.2343 | 0.19062 |

**Swap set at 80% approval:**

| population | n | default rate |
|---|---|---|
| approved by both | 5,797 | 0.2349 |
| swap-in (GBM yes, card no) | 331 | 0.3535 |
| swap-out (card yes, GBM no) | 331 | 0.3384 |

The GBM wins on Gini by +0.009 and **trades worse risk into the book** at the
same approval rate: the applicants it newly approves default at 35.4% versus
33.8% for the ones it stops approving. That is the recommendation this repo
makes — keep the scorecard — and it is the opposite of what the AUC column alone
would suggest. Whether +0.009 Gini and +1.5pp swap-in default are inside the
noise band on 9,580 test rows is a fair challenge, and the answer is that they
probably are; the point of the swap set is that it is the right question to ask,
and the honest response is "these two models are indistinguishable, so ship the
one that's explainable."

Reject inference is not fixed: funded default rate 28.3%, true all-applicant
rate 41.0% (visible only because the generator produced it). Every number above
inherits that bias.

## What is NOT built (the other 80%)

1. **Fair lending is one screen, not an analysis.** The adverse impact ratio at
   the approval threshold exists. Missing: score-distribution comparison across
   groups, the **proxy-feature test** (does removing zip-code-like features
   change the disparity — the sophisticated part, and the spec's differentiator
   #3), threshold-sweep AIR, and `docs/FAIR_LENDING.md` itself. `region_risk` was
   built into the generator as a proxy specifically so this test could be run,
   and it has not been run.
2. **Real data.** Lending Club / Home Credit swap-in not done; this is synthetic.
3. **Reject inference method** — no parcelling, no augmentation, no bureau-score
   fuzzy augmentation. Named and quantified only.
4. **Categorical WoE binning** — numeric features only; no categorical
   treatment, no special-value/missing bin handling (a real card needs both).
5. **Score stability**: no PSI across vintages, no bin-population drift.
6. **Model documentation**: no validation report, no scorecard sign-off pack.
7. Confidence intervals on AUC/Gini/swap-set rates — needed before the
   recommendation above deserves the word "recommendation".
