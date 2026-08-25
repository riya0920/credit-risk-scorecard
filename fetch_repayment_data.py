"""Real repayment outcomes, which HMDA structurally cannot provide.

    python fetch_repayment_data.py

THE GAP THIS FILLS, stated first because it is the whole reason for a second
dataset. `src/hmda.py` gave this project real applications, real decisions and
real protected attributes -- and HMDA records the lender's DECISION, never
whether the loan was repaid. So every scorecard in this repo has been fitted
against `denied` as a stand-in for default, which is a different target with a
different meaning:

    denied    what an underwriter concluded about an applicant
    default   what the applicant subsequently did

A card fitted on the first learns to imitate the underwriter, including their
mistakes and their biases. A card fitted on the second learns credit risk. Every
AUC, KS and PSI figure computed on HMDA in this project is a statement about
agreement with past decisions, and `docs/HMDA_FAIR_LENDING.md` says so.

`run_overrides.py` put the same gap the other way round: "Freddie Mac's
Single-Family Loan-Level dataset is free and does carry repayment outcomes, but
it contains only originated loans -- no denials -- so it can never identify an
override either. The two halves of the question live in two datasets and no
public source joins them."

FREDDIE MAC WAS TRIED FIRST AND IS NOT USABLE HERE. The loan-level dataset
requires registering an account, and creating one is not something this should
do on somebody's behalf. What is used instead is the UCI/OpenML
default-of-credit-card-clients study: 30,000 Taiwanese credit accounts, six
months of repayment history, and a binary default outcome for the following
month.

WHAT MAKES IT THE RIGHT SUBSTITUTE rather than merely an available one: it
carries SEX, EDUCATION and MARRIAGE alongside the outcome. HMDA has protected
attributes and no outcome; Freddie has an outcome and no protected attributes.
This has both in the same file, which is the only configuration in which a
fairness metric can be computed against REPAYMENT rather than against a
decision.

WHAT IT IS NOT. Taiwanese revolving credit in 2005, not US mortgages in 2023.
Nothing measured here transfers to the HMDA population, and the two are kept in
separate documents for that reason. Its value is that the target means what the
word means.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "credit_default.parquet"
URL = "https://data.openml.org/datasets/0004/42477/dataset_42477.pq"


def main() -> int:
    if CACHE.exists() and "--force" not in sys.argv:
        import pandas as pd
        df = pd.read_parquet(CACHE)
        print("cached: {:,} rows x {} columns".format(*df.shape))
        print("columns: {}".format(", ".join(df.columns[:8])))
        return 0

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".part")

    # Resumable, because a truncated parquet reads as a corrupt file rather
    # than as a short one -- and this project has already lost a download to a
    # host that ignored Range and returned 200 instead of 206.
    have = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    if have:
        req.add_header("Range", "bytes={}-".format(have))

    with urllib.request.urlopen(req, timeout=120) as resp:
        if have and resp.status != 206:
            print("host ignored Range (status {}); restarting".format(resp.status))
            have = 0
            tmp.unlink(missing_ok=True)
        mode = "ab" if have else "wb"
        total = int(resp.headers.get("Content-Length", 0)) + have
        with tmp.open(mode) as fh:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
    print("downloaded {:,} bytes (expected {:,})".format(tmp.stat().st_size, total))

    import pandas as pd
    df = pd.read_parquet(tmp)          # verify it parses BEFORE committing it
    tmp.replace(CACHE)

    print("\n{:,} accounts x {} columns".format(*df.shape))
    print("columns: {}".format(", ".join(df.columns)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
