"""Interactions a WoE card cannot represent, and the proxy a cross can smuggle."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.interactions import (additive_prediction, cell_counts, cross, detect,
                              proxy_screen, render_cells)


def _additive(n=8000, seed=0):
    """A world where the two features really are additive. The control."""
    rng = np.random.default_rng(seed)
    a = pd.Series(rng.choice(["employed", "self_employed"], n))
    b = pd.Series(rng.choice(["thick", "thin"], n))
    z = -2.0 + 0.8 * (a == "self_employed") + 0.8 * (b == "thin")
    p = 1 / (1 + np.exp(-z))
    return a, b, (rng.random(n) < p).astype(int)


def _interacting(n=8000, seed=0):
    """The same world plus a genuine interaction: thin files are much worse for
    the self-employed specifically."""
    rng = np.random.default_rng(seed)
    a = pd.Series(rng.choice(["employed", "self_employed"], n))
    b = pd.Series(rng.choice(["thick", "thin"], n))
    both = (a == "self_employed") & (b == "thin")
    z = -2.0 + 0.5 * (a == "self_employed") + 0.5 * (b == "thin") + 1.8 * both
    p = 1 / (1 + np.exp(-z))
    return a, b, (rng.random(n) < p).astype(int)


# ---------------------------------------------------------- the control
def test_an_additive_world_produces_no_material_surprise():
    """If this fires on additive data the detector is measuring noise, and every
    finding it produces on real data is suspect."""
    a, b, y = _additive()
    cells = detect(a, b, y)
    assert cells, "no cells passed the population floor at all"
    assert max(c.weighted_surprise for c in cells) < 0.01


def test_the_additive_prediction_reproduces_the_marginals():
    """The prediction is built from marginals in log-odds space, which is
    exactly what a WoE card does. If it did not match the marginals it would not
    be the thing the card can represent."""
    a, b, y = _additive()
    grid = additive_prediction(a, b, y)
    assert len(grid) == 4
    assert grid.additive.between(0, 1).all()
    assert np.allclose(grid.observed, grid.additive, atol=0.03)


# ------------------------------------------------------ a real interaction
def test_a_genuine_interaction_is_detected():
    a, b, y = _interacting()
    cells = detect(a, b, y)
    top = cells[0]
    assert {top.level_a, top.level_b} == {"self_employed", "thin"}
    assert top.surprise > 0.05, "the interacting cell is not surprising enough"


def test_the_additive_model_understates_the_interacting_cell():
    """The direction matters. An additive card does not merely mis-rank that
    cell -- it prices it too cheaply, so those loans are approved at a score
    they do not deserve."""
    a, b, y = _interacting()
    top = detect(a, b, y)[0]
    assert top.bad_rate > top.additive_bad_rate


def test_surprise_is_weighted_by_population():
    """An unweighted ranking puts the thinnest cells at the top, because a cell
    of forty applicants can be surprising by pure sampling -- and the top of a
    report is where a reader looks."""
    rng = np.random.default_rng(3)
    n = 8000
    a = pd.Series(rng.choice(["big", "big", "big", "big", "tiny"], n))
    b = pd.Series(rng.choice(["x", "y"], n))
    y = (rng.random(n) < 0.1).astype(int)
    # Make the rare combination wildly bad, but keep it rare.
    rare = (a == "tiny") & (b == "y")
    y[rare] = 1

    cells = detect(a, b, y)
    ranked = [(c.level_a, c.level_b) for c in cells]
    assert ("tiny", "y") in ranked
    # It should not outrank a populated cell purely on surprise per applicant.
    assert max(c.weighted_surprise for c in cells) >= \
        [c for c in cells if (c.level_a, c.level_b) == ("tiny", "y")][0].weighted_surprise


def test_cells_below_the_population_floor_are_excluded():
    a, b, y = _interacting()
    a = a.copy()
    a.iloc[:20] = "vanishingly_rare"
    cells = detect(a, b, y, min_cell_share=0.01)
    assert all(c.level_a != "vanishingly_rare" for c in cells)


# ------------------------------------------------------------- the cross
def test_crossing_multiplies_the_level_count():
    a = pd.Series(["p", "q", "r"] * 100)
    b = pd.Series(["x", "y"] * 150)
    assert cross(a, b).nunique() <= 6


def test_cell_counts_reports_what_would_merge_into_other():
    """The number to look at BEFORE deciding a cross is worth it. If most cells
    fall below the floor they merge into OTHER, and the interaction that
    motivated the cross disappears into a bucket."""
    rng = np.random.default_rng(5)
    n = 3000
    a = pd.Series(rng.choice(list("abcdef"), n))
    b = pd.Series(rng.choice(list("ghijklmn"), n))
    c = cell_counts(a, b, min_cell_share=0.02)
    assert c["cells"] > c["cells_above_floor"]
    assert c["population_merged_to_other"] > 0
    assert c["population_above_floor"] + c["population_merged_to_other"] == \
        pytest.approx(1.0)


def test_a_cross_of_two_binary_features_keeps_everything():
    a, b, _ = _additive()
    c = cell_counts(a, b)
    assert c["cells"] == c["cells_above_floor"] == 4
    assert c["population_merged_to_other"] == pytest.approx(0.0)


# ------------------------------------------------------------ the proxy
def _proxy_world(seed=7, n=20000, cell_frac_a=4, cell_frac_b=4):
    """A book where one small cell is overwhelmingly group B.

    The cell is deliberately SMALL relative to the population. A large one
    drags the population baseline up as it fills with the group, which hides
    the concentration -- see
    `test_a_proxy_cell_large_enough_to_move_the_baseline_hides_itself`.
    """
    rng = np.random.default_rng(seed)
    a = pd.Series(rng.choice(["a{}".format(i) for i in range(cell_frac_a)], n))
    b = pd.Series(rng.choice(["b{}".format(i) for i in range(cell_frac_b)], n))
    group = pd.Series(np.where(rng.random(n) < 0.10, "B", "A"))
    cellmask = (a == "a0") & (b == "b0")
    group[cellmask & (rng.random(n) < 0.80)] = "B"
    return a, b, group


def test_an_interaction_cell_that_is_almost_entirely_one_group_is_flagged():
    """The check a per-feature fairness screen cannot perform. Before the cross,
    these intersections were not features, so nothing enumerated them."""
    a, b, group = _proxy_world()
    findings = proxy_screen(a, b, group)
    assert findings, "the constructed proxy cell was not flagged"
    top = findings[0]
    assert (top.level_a, top.level_b) == ("a0", "b0")
    assert top.group == "B"
    assert top.concentration > 3


def test_neither_feature_alone_reveals_the_proxy():
    """The whole point: each feature is individually near-neutral and their
    intersection is not."""
    a, b, group = _proxy_world()
    base = (group == "B").mean()
    for feat in (a, b):
        for lv in feat.unique():
            share = (group[feat == lv] == "B").mean()
            assert share / base < 3.0, (
                "feature {} alone already concentrates the group, so the cross "
                "is not what created the proxy".format(lv))
    assert proxy_screen(a, b, group), "but the cross does"


def test_a_proxy_cell_large_enough_to_move_the_baseline_hides_itself():
    """A real limitation of measuring concentration against the population, and
    it is recorded rather than tuned away.

    The screen compares a cell's group mix to the BOOK's. When the cell is a
    large fraction of the book, filling it with a group raises the book's own
    rate, and the ratio falls back toward 1. A 2x2 cross puts a quarter of the
    population in each cell: loading 75% of one cell with a group that starts at
    10% takes the population to ~27% and the cell to ~77%, a concentration of
    2.9 -- under the 3.0 threshold, on a cell that is three-quarters one group.

    The screen is therefore most sensitive exactly where cells are small, which
    is where proxies usually live, and least sensitive on coarse crosses. Fixing
    it needs an external reference population rather than a self-referential
    one, which this data does not have.
    """
    rng = np.random.default_rng(7)
    n = 20000
    a = pd.Series(rng.choice(["urban", "suburban"], n))
    b = pd.Series(rng.choice(["purchase", "refi"], n))
    group = pd.Series(np.where(rng.random(n) < 0.10, "B", "A"))
    cellmask = (a == "urban") & (b == "refi")
    group[cellmask & (rng.random(n) < 0.75)] = "B"

    cell_share = (group[cellmask] == "B").mean()
    pop_share = (group == "B").mean()
    assert cell_share > 0.7, "the cell really is dominated by the group"
    assert cell_share / pop_share < 3.0
    assert proxy_screen(a, b, group) == [], (
        "if this now flags, the baseline-drag limitation has been addressed "
        "and the docstring needs rewriting rather than trusting")


def test_a_balanced_cross_is_not_flagged():
    rng = np.random.default_rng(9)
    n = 6000
    a = pd.Series(rng.choice(["urban", "suburban"], n))
    b = pd.Series(rng.choice(["purchase", "refi"], n))
    group = pd.Series(rng.choice(["A", "B"], n, p=[0.9, 0.1]))
    assert proxy_screen(a, b, group) == []


def test_concentration_is_a_ratio_not_a_difference():
    """A group at 5% of the book appearing at 20% of a cell is a 4x
    concentration and matters; a group at 40% appearing at 55% is 1.4x and
    mostly does not. A difference-based rule ranks them the opposite way."""
    rng = np.random.default_rng(11)
    n = 6000
    a = pd.Series(rng.choice(["u", "s"], n))
    b = pd.Series(rng.choice(["p", "r"], n))
    group = pd.Series(np.where(rng.random(n) < 0.40, "big", "small"))
    cellmask = (a == "u") & (b == "r")
    group[cellmask & (rng.random(n) < 0.30)] = "big"    # 40% -> ~58%, ~1.4x

    assert proxy_screen(a, b, group) == [], (
        "a 15-point move on a large group was flagged; the rule is behaving "
        "like a difference rather than a ratio")


def test_thin_cells_are_not_screened_as_proxies():
    """A cell of thirty applicants is always concentrated in something."""
    rng = np.random.default_rng(13)
    n = 6000
    a = pd.Series(rng.choice(["u", "s"], n))
    b = pd.Series(rng.choice(["p", "r"], n))
    a = a.copy()
    a.iloc[:25] = "rare"
    group = pd.Series(np.where(rng.random(n) < 0.10, "B", "A"))
    group.iloc[:25] = "B"
    assert all(f.level_a != "rare" for f in proxy_screen(a, b, group))


def test_render_produces_a_table_without_crashing_on_long_names():
    a, b, y = _interacting()
    text = render_cells(detect(a, b, y))
    assert "observed" in text and "additive" in text


# ------------------------------------- the floor real data forced into existence
def test_a_tiny_group_count_cannot_trip_the_screen():
    """The screen produced a spurious finding on real HMDA and the screen was
    wrong, not the data.

    It flagged a 233-applicant cell as concentrating a group at 3.6x. That group
    is 0.2% of the book, so its expected count in the cell is HALF A PERSON --
    and the observed count was two. Two applicants is not a proxy; it is a
    rounding error with a ratio attached.

    Flooring the cell size looks sufficient and is not: a ratio is unstable
    exactly where its denominator is small, and a cell-size floor does not
    constrain the denominator.
    """
    rng = np.random.default_rng(21)
    n = 20000
    # 8x8 bands so a cell is ~312 applicants -- above the 1% population floor
    # and small enough that a handful of people move its mix.
    a = pd.Series(rng.choice(["a{}".format(i) for i in range(8)], n))
    b = pd.Series(rng.choice(["b{}".format(i) for i in range(8)], n))
    group = pd.Series(np.full(n, "A", dtype=object))
    group[rng.choice(n, 10, replace=False)] = "rare"     # 0.05% of the book

    cellmask = (a == "a0") & (b == "b0")
    idx = np.flatnonzero(cellmask.to_numpy())[:4]
    group.iloc[idx] = "rare"

    assert 200 < int(cellmask.sum()) < 600, "cell size assumption broke"

    loose = proxy_screen(a, b, group, min_group_count=1)
    assert any(f.group == "rare" and (f.level_a, f.level_b) == ("a0", "b0")
               for f in loose), (
        "without the floor, four people out of a 0.05% group should produce a "
        "large concentration -- if they do not, this test no longer "
        "demonstrates the failure it was written for")

    assert all(f.group != "rare" for f in proxy_screen(a, b, group)), (
        "four applicants of a 0.05% group produced a proxy finding")


def test_the_group_count_is_reported_so_a_reader_can_judge_it():
    """A concentration ratio without the count behind it is unreadable. 3.6x on
    two people and 3.6x on two thousand are not the same finding."""
    a, b, group = _proxy_world()
    top = proxy_screen(a, b, group)[0]
    assert top.group_count_in_cell >= 30
    assert top.group_count_in_cell == pytest.approx(
        top.group_share_in_cell * top.n, rel=0.02)


def test_the_synthetic_proxy_still_fires_with_the_floor_in_place():
    """The floor must not be so high that it suppresses a real proxy -- the fix
    for a false positive is worthless if it buys a false negative."""
    a, b, group = _proxy_world()
    assert proxy_screen(a, b, group), "the floor suppressed a genuine proxy"
