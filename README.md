# ML-3 — Credit Risk Scorecard with Fair-Lending Analysis

**Status: ~99%.** Scorecard machinery, a real **LightGBM** challenger on the
same characteristics, swap-set analysis, full fair-lending analysis, reject
inference scored against counterfactual truth, **categorical / missing /
special-value / hybrid binning carried on the card itself**, **twelve real
booking vintages** with the mix-shift-versus-outcome-shift decomposition, and a
**generated validation report**. **40 tests.**

```bash
python src/generate.py            # 12 vintages, categoricals, special codes
python run_scorecard.py           # scorecard, challenger, calibration, swap set
python run_stability.py           # intervals, vintages, the two-shift split
python run_fair_lending.py        # -> docs/FAIR_LENDING.md
python run_reject_inference.py    # parcelling scored against hidden outcomes
python run_validation_report.py   # -> docs/SCORECARD_VALIDATION.md
python run_hmda_fair_lending.py   # REAL applicants, REAL protected attributes
python run_redlining.py           # geographic disparity on real census tracts
python -m pytest tests -q         # 40 tests
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

## Fair lending on REAL data — and the finding that changes the argument

Every fair-lending number above runs on a generator whose protected attribute I
fabricated. `run_hmda_fair_lending.py` runs the same arithmetic on **HMDA**: real
mortgage applications, real decisions, and **real race, ethnicity and sex**,
published by the CFPB, free, no account. 2023 Delaware: **29,929 applications,
21,889 originated, 8,040 denied.**

| group | applications | approval rate | AIR vs best | 80% rule |
|---|---|---|---|---|
| Joint | 606 | 79.37% | 1.0000 | pass |
| White | 17,355 | 77.48% | 0.9762 | pass |
| Asian | 1,384 | 76.88% | 0.9686 | pass |
| **Black or African American** | **5,394** | **62.27%** | **0.7846** | **FAIL** |
| 2 or more minority races | 120 | 56.67% | 0.7139 | **FAIL** |
| Native Hawaiian / Pacific Islander | 62 | 50.00% | 0.6299 | **FAIL** |
| American Indian / Alaska Native | 123 | 39.02% | 0.4917 | **FAIL** |

### The headline: the aggregate passes while four groups fail

The binary White-vs-everyone-else cut gives **AIR 0.8478 [0.8333, 0.8625] —
passes.** Four individual groups fail, one of them with 5,394 applications.

Rolling every non-White applicant into one bucket averages a 39% approval rate
together with a 79% one and reports the mean. **The disparity did not go
anywhere; it was averaged away, and the aggregate number would have closed the
file.** ML-1 demonstrates that masking effect on constructed data with its
intersectional cross — this is the same effect on 29,929 real applications, and
it is the strongest argument in this repository for reporting group-by-group
before reporting any summary. A test pins it, so if it ever stops holding the
claim gets rewritten rather than quietly surviving.

### What survives the underwriting controls

| model | features | AUC |
|---|---|---|
| DTI, LTV, income, loan amount, loan-to-income | 5 | 0.7612 |
| + group indicator | 6 | 0.7695 |

Group coefficient **+0.5300** — an odds ratio of **1.699** on denial, after
controlling for every underwriting variable HMDA carries.

**That is a screen, not a verdict.** HMDA has no credit score, and the missing
variable sits exactly where this residual lives. Any factor correlated with both
group and risk shows up here whether or not any lender did anything wrong. It is
grounds to investigate — which is what a regulator's screen is for.

Note also that AUC barely moves (+0.0083). A variable can be statistically
significant and add almost nothing to prediction; reading the AUC delta as "group
does not matter" would be reading the wrong number.

### Three loading decisions that could have skewed all of it

- **Withdrawn is not denied.** Only action codes 1 and 3 are kept. Counting
  withdrawals as denials is the commonest way a HMDA analysis manufactures a
  disparity, because withdrawal rates differ by group for reasons that are not
  the lender's decision.
- **DTI is banded at the extremes.** Below 20% and above 60% it is a string, not
  a number, so `to_numeric` drops precisely the tails. Both bands are mapped
  back, and a test asserts they survive.
- **The best-performing group must have ≥100 applications.** Otherwise a
  12-applicant group at 100% becomes the benchmark and everyone fails against
  noise.

## Redlining screen — a different question from the AIR

The AIR asks whether individual applicants of one group fare worse. Redlining
asks whether **places** do, and the two come apart: a lender can treat every
applicant identically and still decline a neighbourhood by not lending in it.
HMDA carries `census_tract`, so this is measurable on the data already here.

| tract minority share | tracts | applications | denial rate | median income |
|---|---|---|---|---|
| <20% | 114 | 12,808 | 22.14% | $103k |
| 20–50% | 115 | 14,775 | 29.84% | $84k |
| 50–80% | 19 | 1,940 | 32.11% | $70k |

A clean 10-point gradient — **and look at the income column before reading
anything into it.** Tract minority share and tract income move opposite ways
(a test asserts the correlation is negative), so a raw geographic gradient is
partly an income gradient. **A disparity surviving no controls is not evidence
of redlining**; it is evidence that poorer areas get declined more, which
everyone already knows and which is not by itself unlawful.

Controlling for underwriting **and** the applicant's own group:

| term | coefficient | odds ratio |
|---|---|---|
| applicant is minority | +0.3459 | 1.413 |
| tract minority share | +0.3306 | **1.392** |

**The applicant's own group goes in the model first, on purpose.** Without it
the tract term absorbs individual-level disparity and gets reported as geography
— the ecological fallacy with a regression in front of it. A test asserts that
adding the individual term *shrinks* the tract coefficient, which is the
mechanism that warning describes.

The tract term survives anyway. That is a screen, not a finding: no credit
score, no property condition, and this is every lender in the state pooled
rather than one institution's conduct.

Thin tracts (under 30 applications) are excluded and **the exclusion is
reported** — a tract with four applications and one denial has a 25% denial rate
and belongs nowhere near the top of a report.

## What is NOT built

1. ~~**Real data for the SCORECARD.**~~ **DONE** — `src/repayment.py` fits a
   card on 30,000 real credit accounts with a real default outcome AND real
   protected attributes in the same file, which is the only configuration in
   which fairness can be measured against *repayment* rather than against a past
   decision. Freddie Mac was tried first and needs account registration; it also
   holds only originated loans, so it has an outcome and no protected
   attributes. Note the result that inverted: the HMDA decision model scores
   AUC 0.7550 and the real-outcome model 0.7482 — predicting a decision is the
   easier problem, so a high AUC on `denied` is evidence the target was easy.
   See `docs/REPAYMENT.md`.
2. **A randomised approval holdout.** Parcelling, augmentation weighting and
   fuzzy augmentation are all implemented and scored against the generator's
   counterfactual outcomes, but reject inference cannot create information that
   was never observed — it propagates an assumption. Only approving a random
   slice of marginal applicants buys real ground truth, and that costs money.
3. ~~**Interacting categoricals.**~~ **DONE** — `src/interactions.py` detects
   cells where the grid disagrees with the additive model, and screens each
   interaction cell for protected-group concentration, which a per-feature
   fairness check structurally cannot do. The screen produced a **spurious
   finding on real data** — a 3.6x concentration built on *two applicants* — and
   the fix (a group-count floor, not just a cell-size floor) is the recorded
   lesson. After the fix the honest result on HMDA is null.
   See `docs/INTERACTIONS.md`.
4. **An independent validator.** `docs/SCORECARD_VALIDATION.md` has the sign-off
   section and both rows read UNSIGNED. Developer and validator are the same
   party, which under SR 11-7 blocks production use by itself.
5. **The strongest redlining evidence.** It is not the denial rate in a tract
   -- it is the near-absence of APPLICATIONS from it, which means measuring
   marketing and branch presence against tract population. HMDA alone cannot
   see that, and neither can this. Peer-group comparison against similarly
   situated lenders is also absent, as is pricing disparity (`interest_rate`
   is on 73% of rows, originations only).
6. **Business-necessity documentation** for the features driving the disparity —
   that requires the lender's justification, not the modeller's.
7. **Override tracking.** No manual-underwrite path and no low-side/high-side
   override monitoring, which is where a scorecard's real-world performance
   usually goes wrong first.
