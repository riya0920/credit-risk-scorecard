# ML-3 — Credit Risk Scorecard with Fair-Lending Analysis

**Status: ~95%.** Scorecard machinery, a real **LightGBM** challenger,
swap-set analysis, full fair-lending analysis, reject inference scored against
counterfactual truth, **categorical / missing / special-value binning**, and
**vintage stability with bootstrap confidence intervals**. 17 tests.

```bash
python run_scorecard.py         # scorecard, challenger, calibration, swap set
python run_fair_lending.py      # -> docs/FAIR_LENDING.md
python run_reject_inference.py  # parcelling scored against hidden outcomes
python -m pytest tests -q       # 17 tests
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

### The recommendation, corrected by confidence intervals

An earlier version of this README said the two models were indistinguishable and
that the tie broke on explainability. **`run_stability.py` retired that claim.**

| | |
|---|---|
| scorecard AUC | 0.6544 [0.6407, 0.6669] |
| LightGBM AUC | 0.6619 [0.6477, 0.6746] |
| **paired** difference | **+0.0075 [+0.0019, +0.0135]** — excludes zero |
| swap-in minus swap-out default rate | −0.0090 [−0.0827, +0.0609] — spans zero |

Note how easily that was missed: the two individual intervals overlap heavily,
and reading those alone confirms the comfortable answer. A **paired** bootstrap
resamples the same applicants through both models, preserving the correlation in
their errors, and it says the challenger is reliably better at rank-ordering.

The recommendation does not automatically flip. The risk the challenger actually
trades into the book is indistinguishable, so the question is whether 0.0075 AUC
is worth monotonicity a credit officer can check, additive points an agent can
read out, and adverse-action reasons that fall out of the arithmetic. That is a
credit-committee pricing decision, not a modelling one — and it is a far better
question than the one the point estimates supported.

Reject inference is not fixed: funded default rate 28.3%, true all-applicant
rate 41.0% (visible only because the generator produced it). Every number above
inherits that bias.

## Fair lending ([docs/FAIR_LENDING.md](docs/FAIR_LENDING.md))

The spec's differentiator #3 — the proxy test — is built, along with the clause
most fair-lending sections omit.

Two features with ordinary business rationales (a regional risk index, a blended
geography/utilisation score) partially encode geography. That is how proxies
actually arrive: nobody adds a prohibited basis, somebody adds a branch-distance
feature.

| | full model | proxies dropped |
|---|---|---|
| AIR | 0.9646 | 0.9923 |
| group reconstruction AUC | — | 19% of signal retained |

AIR is reported with a **bootstrap 95% CI [0.9393, 0.9899]**, because the 80%
rule gets applied to a point estimate as though it were exact, and 0.79 on a thin
sample is not the same finding as 0.79 on a fat one.

**The less-discriminatory-alternative search is the part worth reading.** Under
disparate-impact doctrine, business necessity does not end the analysis: if a
less discriminatory alternative serves the same purpose comparably well, the
original practice remains actionable. Here the scorecard without proxies improves
AIR by +0.0277 at a cost of 0.0105 AUC — which **exceeds my stated 0.005
tolerance**. So the verdict is not a clean pass:

> an alternative DID reduce disparity but exceeded the stated accuracy tolerance
> — the tolerance itself now requires justification

The tolerance is a number I chose, and "our tolerance said no" is not a defence
when the tolerance was set by the party whose model is under review. The
alternative is live and the threshold needs an owner outside modelling. Collapsing
that into a single pass/fail is how a real alternative gets quietly dismissed,
which is why the code reports the two failure modes separately.

## What is NOT built

1. **Real data.** Lending Club / Home Credit swap-in not done; this is synthetic,
   and the protected attribute is fabricated by the generator. No result here
   describes a real population.
2. **A randomised approval holdout.** Parcelling, augmentation weighting and
   fuzzy augmentation are all implemented and scored against the generator's
   counterfactual outcomes, but reject inference cannot create information that
   was never observed — it propagates an assumption. Only approving a random
   slice of marginal applicants buys real ground truth, and that costs money.
3. **Categorical features in the actual card.** `src/categorical.py` implements
   categorical WoE with missing, special-value and rare-level handling, but the
   generator produces only numeric characteristics, so the machinery is tested
   rather than exercised end to end.
4. **A multi-quarter vintage split.** `run_stability.py` computes vintage PSI,
   but the vintages are slices of one generated population, so stability is the
   expected result -- it proves the machinery, not the model.
5. **Model documentation**: no validation report, no scorecard sign-off pack.
   (ML-1 has one; this project does not.)
6. **Business-necessity documentation** for the features driving the disparity —
   that requires the lender's justification, not the modeller's.
7. **Intersectional analysis**: one binary attribute, no age or marital status,
   no geography-based redlining analysis.
8. **A decision on the corrected recommendation.** See below -- the paired
   interval changed the finding, and whether 0.0075 AUC is worth the
   explainability premium is a credit-committee call, not a modelling one.
