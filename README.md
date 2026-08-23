# ML-3 — Credit Risk Scorecard with Fair-Lending Analysis

**Status: ~98%.** Scorecard machinery, a real **LightGBM** challenger on the
same characteristics, swap-set analysis, full fair-lending analysis, reject
inference scored against counterfactual truth, **categorical / missing /
special-value / hybrid binning carried on the card itself**, **twelve real
booking vintages** with the mix-shift-versus-outcome-shift decomposition, and a
**generated validation report**. **29 tests.**

```bash
python src/generate.py            # 12 vintages, categoricals, special codes
python run_scorecard.py           # scorecard, challenger, calibration, swap set
python run_stability.py           # intervals, vintages, the two-shift split
python run_fair_lending.py        # -> docs/FAIR_LENDING.md
python run_reject_inference.py    # parcelling scored against hidden outcomes
python run_validation_report.py   # -> docs/SCORECARD_VALIDATION.md
python -m pytest tests -q         # 29 tests
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

Both models now see the **same eleven characteristics** — seven numeric, three
categorical, one hybrid.

| model | AUC | Gini | KS | Brier |
|---|---|---|---|---|
| scorecard (WoE + logistic) | 0.6710 | 0.3420 | 0.2456 | 0.18689 |
| GBM (monotonic, same characteristics) | 0.6728 | 0.3456 | 0.2596 | 0.18705 |

**The challenger was handicapped and I nearly published the result.** When the
categoricals were first added they went onto the card only; the GBM was still
fitted on the seven numeric columns. The scorecard came out ahead by 0.0128 AUC,
which reads as *the interpretable model wins* — and what had actually happened
is that the challenger was never shown four characteristics. A bakeoff where the
two models see different data measures the feature list, not the model. Fixed,
and the gap collapsed to 0.0018.

## The characteristics that are not numbers

| characteristic | kind | treatment |
|---|---|---|
| `home_ownership` | categorical | ordinary nominal, WoE per level |
| `loan_purpose` | categorical | rare level merged into `__OTHER__` rather than scored |
| `employment_type` | categorical | **`__MISSING__` is a level with its own points** |
| `credit_age_reported` | **hybrid** | bureau's `-9` no-hit code binned apart from the quantity |

Three treatments worth defending:

**Missing is a level, not a hole.** `employment_type` is absent for 9.4% of
applicants, and that absence is *not at random* — it concentrates in the
self-employed, who default more. On the card the missing level earns **34.6
points against W2's 47.1**. Impute the mode and you import the W2 default rate
onto the riskiest slice of the book.

**A special code is not a quantity.** `-9` means "no bureau record". Binned as a
number it sorts below every real credit age, lands in the youngest-file bin and
inherits its risk — an artefact of the encoding, not a statement anyone made
about the applicant.

**But binning the whole column as a category is just as wrong**, and that is a
mistake this build actually made. Every distinct age became its own level, the
rare ones merged into `__OTHER__`, and the characteristic collapsed to two bins
with **identical points (44.3 and 44.3)** — a row on the card that could not
affect a decision. `HybridBinning` bins the special codes apart and the
remainder as the quantity it is:

```
credit_age_reported (-inf, 63]     2808   0.3251   41.7
credit_age_reported (63, 84]       2870   0.3115   42.6
credit_age_reported (84, 126]      5552   0.2983   43.4
credit_age_reported (126, 158]     2728   0.2526   46.4
credit_age_reported (158, inf]     2755   0.2203   48.8
credit_age_reported -9.0            588   0.2704   45.2   <- no-hit, mid-range
```

Its IV goes from 0.0000 to 0.0329, and a test asserts the no-hit bin sits
*between* the numeric extremes rather than at one of them.

## Vintages: PSI found the smaller half

The generator now produces **twelve monthly booking cohorts** with two
independent shifts, because they mean opposite things:

- **mix shift** — later cohorts carry thinner files and higher utilisation. PSI
  moves; the model is still correct and the answer is to reprice.
- **outcome shift** — later cohorts default more *at the same characteristics*.
  PSI does not move; the model is wrong and the answer is to rebuild.

Actual default rate runs 18.15% (M00) to 40.52% (M11). Characteristic PSI, early
cohorts against late:

| characteristic | PSI | band |
|---|---|---|
| utilization | 0.2939 | INVESTIGATE |
| credit_age_months | 0.2872 | INVESTIGATE |
| income | 0.0130 | stable |
| dti | 0.0070 | stable |

PSI fires. And then the decomposition:

```
observed deterioration : +15.77%
explained by mix shift : +5.12%   (what the model predicted)
UNEXPLAINED            : +10.66%  (the model did not see this)
```

**PSI found the shift it is designed to find, and that shift is the smaller
half.** Two-thirds of the deterioration is applicants defaulting more at
unchanged characteristics, which is invisible to any distribution statistic
computed on those characteristics — by construction, not by bad luck. The
monitoring conclusion is that PSI is a supporting signal and the primary one is
the vintage actual-versus-expected gap, which needs outcomes and is therefore
months late. There is no early label-free signal for outcome shift, and treating
PSI as one is the specific error this section exists to prevent.

## The recommendation, and how it moved twice

| | |
|---|---|
| scorecard AUC | 0.6710 [0.6586, 0.6838] |
| LightGBM AUC | 0.6682 [0.6548, 0.6815] |
| **paired** difference | **−0.0029 [−0.0091, +0.0046]** — spans zero |
| swap-in minus swap-out default rate | +0.0087 [−0.0707, +0.0768] — spans zero |

An earlier version of this README said the two models were indistinguishable.
The paired bootstrap then retired that claim: on seven numeric characteristics
it measured **+0.0075 [+0.0019, +0.0135]**, excluding zero, and the challenger
was reliably better at rank-ordering. With the four new characteristics on both
sides, the paired difference is back inside the noise band.

Both readings were correct on their own data, which is the uncomfortable part:
**the conclusion is a function of the feature list, and the feature list is a
choice.** A hypothesis for why, flagged as a hypothesis and not a result: the
generator's categorical effects are additive on the log-odds, which is exactly
the scorecard's functional form, so handing them to both models closes a gap
that came from the tree capturing structure the card could not express. That is
testable — a generator with interacting categoricals should reopen it — and it
has not been tested here.

The recommendation itself does not flip. When two models are statistically
indistinguishable the tie-break is everything else: monotonicity a credit officer
can check, additive points an agent can read out, and adverse-action reasons that
fall out of the arithmetic rather than needing a second model to explain the
first.

Reject inference is not fixed: funded default rate 28.28%, true all-applicant
rate 41.02% (visible only because the generator produced it). Every number above
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
| AIR | 0.9518 | 0.9880 |
| group reconstruction AUC | 0.5935 | 18% of signal retained |

AIR is reported with a **bootstrap 95% CI [0.9291, 0.9740]**, because the 80%
rule gets applied to a point estimate as though it were exact, and 0.79 on a thin
sample is not the same finding as 0.79 on a fat one.

**The less-discriminatory-alternative search is the part worth reading.** Under
disparate-impact doctrine, business necessity does not end the analysis: if a
less discriminatory alternative serves the same purpose comparably well, the
original practice remains actionable. Here the scorecard without proxies improves
AIR by +0.0362 at a cost of 0.0141 AUC — which **exceeds my stated 0.005
tolerance by nearly 3x**. So the verdict is not a clean pass:

> an alternative DID reduce disparity but exceeded the stated accuracy tolerance
> — the tolerance itself now requires justification

The tolerance is a number I chose, and "our tolerance said no" is not a defence
when the tolerance was set by the party whose model is under review. The
alternative is live and the threshold needs an owner outside modelling. Collapsing
that into a single pass/fail is how a real alternative gets quietly dismissed,
which is why the code reports the two failure modes separately.

## Validation report

`run_validation_report.py` generates `docs/SCORECARD_VALIDATION.md` — model
identification and scaling, the limitation that governs every number (§2, second
rather than last on purpose), discrimination with a bootstrap interval,
characteristic-level review, the stability section above, use limitations, and a
sign-off table whose validator row reads **UNSIGNED**.

It is generated rather than written for the same reason ML-1's RESULTS.md is: a
validation pack whose numbers were typed in by the developer is a document about
the developer's memory.

## What is NOT built

1. **Real data.** Lending Club / Home Credit swap-in not done; this is synthetic,
   and the protected attribute is fabricated by the generator. No result here
   describes a real population, and that is the one gap in this project no
   amount of further code closes.
2. **A randomised approval holdout.** Parcelling, augmentation weighting and
   fuzzy augmentation are all implemented and scored against the generator's
   counterfactual outcomes, but reject inference cannot create information that
   was never observed — it propagates an assumption. Only approving a random
   slice of marginal applicants buys real ground truth, and that costs money.
3. **Interacting categoricals.** The generator's categorical effects are additive
   on the log-odds, which is the scorecard's own functional form and is the
   likeliest explanation for the challenger's edge disappearing. A generator with
   genuine interactions would test that, and this one does not have them.
4. **An independent validator.** `docs/SCORECARD_VALIDATION.md` has the sign-off
   section and both rows read UNSIGNED. Developer and validator are the same
   party, which under SR 11-7 blocks production use by itself.
5. **Intersectional analysis.** One binary attribute, no age or marital status,
   no geography-based redlining analysis. (ML-1 implements the intersectional
   cross; this project does not.)
6. **Business-necessity documentation** for the features driving the disparity —
   that requires the lender's justification, not the modeller's.
7. **Override tracking.** No manual-underwrite path and no low-side/high-side
   override monitoring, which is where a scorecard's real-world performance
   usually goes wrong first.
