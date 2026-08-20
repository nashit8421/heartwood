"""The split scan — the one routine every tree node calls, on every candidate.

`scan_threshold` is a vectorised cumulative-sum trick standing in for an O(n²)
search.  It is checked against that O(n²) search directly, on adversarial inputs:
duplicate feature values, missing values, degenerate hessians, zero-gain cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import brute_scan, noisy_feature
from heartwood.splits import SplitSpec, sample_interval, sample_shapelet, scan_threshold


def test_matches_brute_force_over_many_random_problems(rng):
    """The single most important test in the suite."""
    checked = 0
    for _ in range(400):
        n = int(rng.integers(6, 45))
        f = noisy_feature(rng, n, nan_frac=0.25 if rng.random() < 0.6 else 0.0)
        g = rng.normal(size=n)
        h = rng.uniform(0.05, 1.5, size=n)
        params = dict(
            reg_lambda=float(rng.choice([0.0, 1.0, 5.0])),
            gamma=float(rng.choice([0.0, 0.05])),
            min_child_weight=float(rng.choice([0.0, 0.1, 1.0])),
            min_samples_leaf=int(rng.integers(1, 5)),
        )
        got = scan_threshold(f, g, h, **params)
        want = brute_scan(f, g, h, **params)

        assert (got is None) == (want is None), f"disagreed on whether a split exists\nf={f}"
        if got is not None:
            assert np.isclose(got[0], want[0], rtol=1e-9, atol=1e-12), f"gain differs\nf={f}"
            checked += 1
    assert checked > 100, "the random problems were too easy to be informative"


def test_reported_gain_is_the_gain_of_the_reported_split(rng):
    """Guards against returning the right gain with the wrong threshold."""
    for _ in range(200):
        n = int(rng.integers(10, 40))
        f = noisy_feature(rng, n, nan_frac=0.3)
        g = rng.normal(size=n)
        h = rng.uniform(0.1, 1.0, size=n)
        lam = 1.0
        found = scan_threshold(f, g, h, lam, 0.0, 1e-3, 2)
        if found is None:
            continue
        gain, threshold, missing_left = found

        left = np.where(np.isfinite(f), f <= threshold, missing_left)
        GL, HL = g[left].sum(), h[left].sum()
        GR, HR = g[~left].sum(), h[~left].sum()
        G, H = g.sum(), h.sum()
        expected = 0.5 * (
            GL**2 / (HL + lam) + GR**2 / (HR + lam) - G**2 / (H + lam)
        )
        assert np.isclose(gain, expected, rtol=1e-9, atol=1e-12)


def test_missing_rows_are_counted_in_the_parent(rng):
    """Parent and children totals must both include the missing group."""
    f = np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.nan])
    g = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -0.5])
    h = np.ones(6)
    gain, threshold, missing_left = scan_threshold(f, g, h, 1.0, 0.0, 0.0, 1)

    left = np.where(np.isfinite(f), f <= threshold, missing_left)
    assert left.sum() + (~left).sum() == 6
    assert np.isfinite(gain)


def test_missing_direction_is_chosen_not_assumed():
    """Missing rows carry signal here, so they must be sent to the matching side."""
    f = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, np.nan, np.nan, np.nan])
    # missing rows behave like the f==1 group, so they belong on the right
    g = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    h = np.ones(9)
    gain, _, missing_left = scan_threshold(f, g, h, 1.0, 0.0, 0.0, 1)
    assert missing_left is False

    g_flipped = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    _, _, missing_left = scan_threshold(f, g_flipped, h, 1.0, 0.0, 0.0, 1)
    assert missing_left is True


@pytest.mark.parametrize(
    "feature",
    [
        np.ones(20),  # constant
        np.full(20, np.nan),  # entirely missing
        np.array([np.nan] * 19 + [1.0]),  # a single observation
        np.array([]),  # empty
    ],
)
def test_unsplittable_features_return_none(feature, rng):
    n = len(feature)
    result = scan_threshold(
        feature, rng.normal(size=n), np.ones(n), 1.0, 0.0, 1e-3, 1
    )
    assert result is None


def test_never_splits_between_equal_values():
    """A threshold falling on a tie would route identical rows differently."""
    f = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    g = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    _, threshold, _ = scan_threshold(f, g, np.ones(8), 1.0, 0.0, 0.0, 1)
    assert 1.0 <= threshold < 2.0


def test_min_samples_leaf_is_respected(rng):
    for _ in range(100):
        n = 30
        f = noisy_feature(rng, n, nan_frac=0.2)
        g, h = rng.normal(size=n), np.ones(n)
        found = scan_threshold(f, g, h, 1.0, 0.0, 0.0, min_samples_leaf=8)
        if found is None:
            continue
        _, threshold, missing_left = found
        left = np.where(np.isfinite(f), f <= threshold, missing_left)
        assert left.sum() >= 8 and (~left).sum() >= 8


def test_min_child_weight_is_respected(rng):
    n = 40
    f = noisy_feature(rng, n)
    g = rng.normal(size=n)
    h = rng.uniform(0.01, 0.05, size=n)
    found = scan_threshold(f, g, h, 1.0, 0.0, min_child_weight=0.5, min_samples_leaf=1)
    if found is not None:
        _, threshold, missing_left = found
        left = np.where(np.isfinite(f), f <= threshold, missing_left)
        assert h[left].sum() >= 0.5 and h[~left].sum() >= 0.5


def test_gamma_suppresses_marginal_splits(rng):
    n = 30
    f = np.arange(n, dtype=float)
    g = rng.normal(scale=0.01, size=n)
    h = np.ones(n)
    assert scan_threshold(f, g, h, 1.0, 0.0, 0.0, 1) is not None
    assert scan_threshold(f, g, h, 1.0, 1e6, 0.0, 1) is None


def test_zero_gradient_means_nothing_to_learn():
    f = np.arange(20, dtype=float)
    assert scan_threshold(f, np.zeros(20), np.ones(20), 1.0, 0.0, 0.0, 1) is None


# --------------------------------------------------------------- SplitSpec


def test_describe_and_family_are_readable():
    spec = SplitSpec(kind="interval", channel=1, start=12, end=40, stat="slope",
                     threshold=0.31, gain=4.0, missing_left=True)
    assert spec.describe().startswith("series[ch=1].slope[t=12:40] <= 0.31")
    assert spec.family() == "interval(ch=1, slope)"

    spec = SplitSpec(kind="static", col=3, threshold=0.5)
    assert "static[3]" in spec.describe()
    assert spec.family() == "static[3]"

    spec = SplitSpec(kind="shapelet_pos", channel=0, shapelet=np.zeros(9), threshold=0.2)
    assert "shapelet_pos(len=9)" in spec.describe()
    assert spec.family() == "shapelet_pos(ch=0)"


def test_describe_shows_where_missing_values_go():
    left = SplitSpec(kind="static", col=0, threshold=1.0, missing_left=True).describe()
    right = SplitSpec(kind="static", col=0, threshold=1.0, missing_left=False).describe()
    assert "missing->left" in left and "missing->right" in right


# ------------------------------------------------------------- the samplers


def test_sample_interval_stays_in_bounds(rng):
    T = 60
    for _ in range(500):
        start, end = sample_interval(T, rng, min_len=3, full_prob=0.25)
        assert 0 <= start < end <= T
        assert end - start >= 3


def test_sample_interval_can_propose_the_whole_series(rng):
    """The global aggregate must stay inside the hypothesis space."""
    assert sample_interval(50, rng, 3, full_prob=1.0) == (0, 50)
    windows = {sample_interval(50, rng, 3, 0.25) for _ in range(400)}
    assert (0, 50) in windows
    assert len(windows) > 20, "sampling should explore, not collapse onto one window"


def test_sample_interval_handles_series_shorter_than_the_minimum(rng):
    assert sample_interval(2, rng, min_len=3, full_prob=0.0) == (0, 2)


def test_sample_shapelet_returns_usable_templates(rng):
    X = rng.normal(size=(30, 2, 40))
    rows = np.arange(30)
    for _ in range(100):
        drawn = sample_shapelet(X, rows, rng, min_len=3, max_frac=0.5)
        assert drawn is not None
        channel, shp = drawn
        assert 0 <= channel < 2
        assert np.isfinite(shp).all(), "a template with NaN would match nothing"
        assert shp.std() > 1e-12, "a constant template has no shape to match"
        assert 3 <= len(shp) <= 40


def test_sample_shapelet_copies_so_later_training_cannot_mutate_it(rng):
    X = rng.normal(size=(5, 1, 20))
    channel, shp = sample_shapelet(X, np.arange(5), rng, 3, 0.5)
    before = shp.copy()
    X[:] = 0.0
    assert np.array_equal(shp, before)


def test_sample_shapelet_gives_up_gracefully(rng):
    """Series too short, or with nothing but NaN and constants, yield None."""
    assert sample_shapelet(rng.normal(size=(4, 1, 3)), np.arange(4), rng, 3, 0.5) is None
    assert sample_shapelet(np.full((4, 1, 30), np.nan), np.arange(4), rng, 3, 0.5) is None
    assert sample_shapelet(np.zeros((4, 1, 30)), np.arange(4), rng, 3, 0.5) is None
    assert sample_shapelet(rng.normal(size=(4, 1, 30)), np.array([], dtype=int), rng, 3, 0.5) is None
