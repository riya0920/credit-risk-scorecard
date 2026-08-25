"""Interactions between categoricals, and the proxy they can smuggle in.

A WoE scorecard is ADDITIVE BY CONSTRUCTION. Each feature is replaced by its
weight of evidence and the pieces are summed in log-odds space, so the card can
say "self-employed is worse" and "a thin file is worse" and it cannot say "a
thin file is worse IF you are self-employed". An interaction is precisely the
statement the functional form cannot make.

That is usually fine and occasionally the whole risk. Where it is not fine, the
standard fix is to cross the two features into one categorical whose levels are
the cells of the grid, and bin that. Which brings three problems, in order of
how quietly they bite.

CELL COUNT. Crossing a 6-level feature with an 8-level one gives 48 cells, and
the population does not grow to match. Most of the new levels are thin, their
bad rates are noise, and `fit_categorical_binning`'s rare-level floor will merge
them into OTHER -- so the interaction that motivated the cross mostly disappears
into a bucket. Crossing is only worth it where the cells are populated.

WHICH INTERACTION, AND WHY. Adding every cross and keeping what improves fit is
how a card gets ten spurious interactions and no generalisation. `detect` scores
each candidate cell by the gap between its observed bad rate and what the
ADDITIVE model predicts for it, weighted by population -- so a cell is only
interesting if it is both surprising and real.

AND THE ONE THAT MATTERS MOST: AN INTERACTION IS WHERE A SCORECARD BECOMES A
PROXY. Two features can each be individually uncorrelated with a protected
class and their INTERSECTION can be almost entirely one group. "Applicant in
tract type X with loan purpose Y" is a cell, and cells are small -- small enough
to identify a population that a regulator would call a class.

A single-feature fairness screen cannot see this. `src/fair_lending.py` measures
disparity per feature and per group; it does not enumerate the intersections,
because before the cross those intersections were not features. `proxy_screen`
below checks each interaction cell for protected-group concentration, and it is
the reason this module exists rather than a `pd.crosstab` at a call site.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# A cell below this share of the population is not evidence of anything, so it
# is not a candidate interaction however surprising its bad rate.
MIN_CELL_SHARE = 0.01

# A cell whose protected-group mix is this far from the population's is flagged.
# Expressed as a RATIO rather than a difference: a group at 5% of the book
# appearing at 20% of a cell is a 4x concentration and matters, while a group at
# 40% appearing at 55% is a 1.4x and mostly does not.
PROXY_CONCENTRATION_RATIO = 3.0

# And a cell must contain at least this many members OF THE GROUP before its
# concentration means anything.
#
# THIS WAS ADDED AFTER THE SCREEN PRODUCED A SPURIOUS FINDING ON REAL DATA, and
# the mechanism is worth stating because flooring the cell size looks sufficient
# and is not. On HMDA Delaware the screen flagged a 233-applicant cell as
# concentrating a group at 3.6x. The group is 0.2% of the book, so its expected
# count in that cell is HALF A PERSON -- and the observed count was two. Two
# applicants is not a proxy; it is a rounding error with a ratio attached.
#
# A ratio is unstable exactly where its denominator is small, and a per-cell
# population floor does not constrain the denominator. Both floors are needed.
MIN_GROUP_COUNT = 30


@dataclass
class Cell:
    level_a: str
    level_b: str
    n: int
    share: float
    bad_rate: float
    additive_bad_rate: float
    surprise: float               # observed minus additive, in bad-rate points
    weighted_surprise: float      # surprise x share -- the ranking quantity


@dataclass
class ProxyFinding:
    level_a: str
    level_b: str
    n: int
    group: str
    group_count_in_cell: int
    group_share_in_cell: float
    group_share_in_population: float
    concentration: float


def _rate(y: np.ndarray) -> float:
    return float(np.mean(y)) if len(y) else float("nan")


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return float(np.log(p / (1 - p)))


def _inv_logit(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


def additive_prediction(a: pd.Series, b: pd.Series, y: np.ndarray) -> pd.DataFrame:
    """What an additive model predicts for each cell of the a-by-b grid.

    Built from the MARGINALS in log-odds space, which is exactly what a WoE
    card does: base log-odds plus the deviation for level a plus the deviation
    for level b. Anything the grid does that this cannot reproduce is the
    interaction, by definition.
    """
    y = np.asarray(y)
    base = _logit(_rate(y))
    da = {lv: _logit(_rate(y[a.to_numpy() == lv])) - base for lv in a.unique()}
    db = {lv: _logit(_rate(y[b.to_numpy() == lv])) - base for lv in b.unique()}

    rows = []
    for la in a.unique():
        for lb in b.unique():
            mask = (a.to_numpy() == la) & (b.to_numpy() == lb)
            n = int(mask.sum())
            rows.append({
                "level_a": str(la), "level_b": str(lb), "n": n,
                "observed": _rate(y[mask]) if n else float("nan"),
                "additive": _inv_logit(base + da[la] + db[lb]),
            })
    return pd.DataFrame(rows)


def detect(a: pd.Series, b: pd.Series, y: np.ndarray,
           min_cell_share: float = MIN_CELL_SHARE) -> list:
    """Cells where the grid disagrees with the additive model.

    Ranked by surprise WEIGHTED BY POPULATION. An unweighted ranking puts the
    thinnest cells at the top, because a cell of forty applicants can be
    surprising by pure sampling -- and the top of a report is where a reader
    looks.
    """
    grid = additive_prediction(a, b, y)
    total = len(a)
    out = []
    for _, r in grid.iterrows():
        share = r["n"] / total if total else 0.0
        if share < min_cell_share or not np.isfinite(r["observed"]):
            continue
        surprise = r["observed"] - r["additive"]
        out.append(Cell(r["level_a"], r["level_b"], int(r["n"]), share,
                        r["observed"], r["additive"], surprise,
                        abs(surprise) * share))
    return sorted(out, key=lambda c: -c.weighted_surprise)


def cross(a: pd.Series, b: pd.Series, sep: str = " x ") -> pd.Series:
    """The crossed feature. Fed to `fit_categorical_binning` like any other."""
    return a.astype(str) + sep + b.astype(str)


def cell_counts(a: pd.Series, b: pd.Series,
                min_cell_share: float = MIN_CELL_SHARE) -> dict:
    """How much of the cross survives a population floor.

    The number to look at before deciding a cross is worth it: if most cells are
    below the floor they will merge into OTHER, and the interaction that
    motivated the cross disappears into a bucket.
    """
    counts = cross(a, b).value_counts()
    total = counts.sum()
    keep = counts[counts / total >= min_cell_share]
    return {
        "cells": int(len(counts)),
        "cells_above_floor": int(len(keep)),
        "population_above_floor": float(keep.sum() / total) if total else 0.0,
        "population_merged_to_other": float(1 - keep.sum() / total)
        if total else 0.0,
    }


def proxy_screen(a: pd.Series, b: pd.Series, group: pd.Series,
                 min_cell_share: float = MIN_CELL_SHARE,
                 concentration_ratio: float = PROXY_CONCENTRATION_RATIO,
                 min_group_count: int = MIN_GROUP_COUNT) -> list:
    """Interaction cells that are disproportionately one protected group.

    THE CHECK A PER-FEATURE FAIRNESS SCREEN CANNOT PERFORM. Before the cross,
    those intersections were not features, so nothing enumerated them. Two
    individually-neutral inputs can meet in a cell that is almost entirely one
    group, and that cell is then a variable the model can use.

    Reported as a CONCENTRATION RATIO against the population mix rather than as
    a raw share: a group at 5% of the book appearing at 20% of a cell is a 4x
    concentration and matters, while a group at 40% appearing at 55% is 1.4x and
    mostly does not.

    A LIMITATION THIS HAS, recorded rather than tuned away, because it was found
    by a test that failed for the right reason. The reference is the book's own
    mix, which makes it SELF-REFERENTIAL: when a cell is a large fraction of the
    population, filling it with a group raises the book's rate too and the ratio
    falls back toward 1. On a 2x2 cross each cell is a quarter of the book, and
    loading one of them 75% with a group that starts at 10% takes the population
    to ~27% and the cell to ~77% -- a concentration of 2.9, under the threshold,
    on a cell that is three-quarters one group.

    So the screen is most sensitive where cells are small, which is where
    proxies usually live, and least sensitive on coarse crosses, where the
    "proxy" is arguably just a large segment. Fixing it properly needs an
    external reference population rather than the book itself, which this data
    does not have.
    """
    total = len(a)
    crossed = cross(a, b)
    base = group.value_counts(normalize=True)

    out = []
    for cell, n in crossed.value_counts().items():
        if n / total < min_cell_share:
            continue
        in_cell = group[crossed == cell]
        counts = in_cell.value_counts()
        mix = in_cell.value_counts(normalize=True)
        for g, share in mix.items():
            pop = base.get(g, 0.0)
            if pop <= 0:
                continue
            # BOTH floors. The cell must be big enough AND the group must
            # actually be present in it -- a ratio is unstable exactly where its
            # denominator is small, and a cell-size floor does not constrain the
            # denominator.
            if int(counts[g]) < min_group_count:
                continue
            conc = share / pop
            if conc >= concentration_ratio:
                la, _, lb = str(cell).partition(" x ")
                out.append(ProxyFinding(la, lb, int(n), str(g),
                                        int(counts[g]), float(share),
                                        float(pop), float(conc)))
    return sorted(out, key=lambda f: -f.concentration)


def render_cells(cells: list, limit: int = 10) -> str:
    L = ["{:<22}{:<22}{:>8}{:>9}{:>11}{:>11}".format(
        "level a", "level b", "n", "share", "observed", "additive")]
    for c in cells[:limit]:
        L.append("{:<22}{:<22}{:>8,}{:>8.1%}{:>10.2%}{:>11.2%}".format(
            c.level_a[:21], c.level_b[:21], c.n, c.share, c.bad_rate,
            c.additive_bad_rate))
    return "\n".join(L)
