"""The linear base and the cross-channel areas — Phase C (PLAN.md §11).

The leave-one-out test here is not a formality.  A first implementation of this
module passed every other check while quietly leaking: with more features than
rows the ridge interpolated, every leverage went to 1, and the leave-one-out
formula divided a rounding error by another rounding error — producing margins
that still carried the sign of the label.  Training accuracy was 1.00 and test
accuracy was 0.47.  Nothing raised.  The two tests that catch that are
``test_loo_margins_match_an_explicit_refit`` and ``test_no_leak_on_random_labels``.
"""

from __future__ import annotations

import numpy as np
import pytest

from heartwood import HeartwoodClassifier, HeartwoodRegressor
from heartwood.datasets import make_lead_lag, make_shape_amplitude_regression
from heartwood.features import STAT_NAMES, interval_stat
from heartwood.dense import (
    _platt,
    DenseBase,
    dyadic_windows,
    levy_area_columns,
)


# ------------------------------------------------------------- the bank




def test_dense_bank_contains_the_global_aggregate(rng):
    """The linear layer must be at least as expressive as the baseline it beats."""
    X = rng.normal(size=(20, 1, 64))
    bank = dense_bank(X, stats=("mean",))
    whole_series_mean = X[:, 0, :].mean(axis=1)
    assert any(
        np.allclose(bank[:, j], whole_series_mean) for j in range(bank.shape[1])
    ), "the whole-series mean should appear as one of the columns"



def test_loo_margins_match_an_explicit_refit(rng):
    """The closed form must equal actually refitting without each row.

    Preprocessing (imputation, standardisation, target centring) is treated as
    fixed and given, which is what the closed form assumes; the check is on the
    ridge solve itself.
    """
    n, p = 30, 12
    X_raw = rng.normal(size=(n, p))
    y = X_raw[:, 0] * 1.5 - X_raw[:, 3] + rng.normal(scale=0.3, size=n)

    base = DenseBase("regression", 1)
    loo = base.fit(X_raw, y)[:, 0]

    # rebuild exactly the matrix the ridge saw
    X = base._prepare(X_raw, fitting=False)
    lam = base.lambda_
    centre = float(base.target_center_[0])
    yc = y - centre

    for i in range(n):
        keep = np.arange(n) != i
        Xi, yi = X[keep], yc[keep]
        weights = np.linalg.solve(Xi.T @ Xi + lam * np.eye(p), Xi.T @ yi)
        explicit = float(X[i] @ weights) + centre
        assert np.isclose(loo[i], explicit, rtol=1e-6, atol=1e-8), (
            f"row {i}: closed form {loo[i]:.6f} vs explicit refit {explicit:.6f}"
        )


def test_loo_is_not_the_in_sample_fit(rng):
    """If the two coincided, the whole exercise would be pointless.

    Needs a design with real signal: on pure noise the base now declines
    outright, which is a different (and also correct) behaviour.
    """
    n, p = 60, 25
    X = rng.normal(size=(n, p))
    y = X[:, :5] @ np.array([2.0, -1.5, 1.0, -0.8, 0.6]) + 0.3 * rng.normal(size=n)
    base = DenseBase("regression", 1)
    margins = base.fit(X, y)
    assert margins is not None, "a design with real signal must not be declined"
    loo = margins[:, 0]
    fitted = base.transform(X)[:, 0]
    assert not np.allclose(loo, fitted, atol=1e-6)
    # and leave-one-out must be the more pessimistic of the two on noise
    assert np.mean((loo - y) ** 2) > np.mean((fitted - y) ** 2)


@pytest.mark.parametrize("n", [80, 200])
def test_no_leak_on_random_labels(n, rng):
    """With nothing to learn, the base must decline rather than invent a signal.

    A leaky implementation looks near-perfect here while being worthless out of
    sample, which is precisely how this failure hides.  The base now says so
    explicitly by returning ``None``; if it ever does return margins on noise,
    they must still be at chance.
    """
    X_series = rng.normal(size=(n, 1, 100))
    y = rng.integers(0, 2, size=n).astype(np.float64)

    base = DenseBase("classification", 1)
    margins = base.fit(dense_bank(X_series), y)
    if margins is None:
        assert base.degenerate_ and base.loo_r2_ <= 0.0
        return
    accuracy = ((margins[:, 0] > 0).astype(int) == y).mean()
    assert accuracy < 0.70, f"leave-one-out margins scored {accuracy:.3f} on noise"


def test_an_uninformative_base_is_declined_at_predict_time_too(rng):
    """Whatever ``fit`` decided, ``transform`` must decide the same thing.

    Otherwise a row is scored one way in training and another way in inference,
    which is the shape of the bug this guard was added for.
    """
    X_series = rng.normal(size=(120, 1, 100))
    y = rng.integers(0, 2, size=120).astype(np.float64)
    base = DenseBase("classification", 1)
    fitted = base.fit(dense_bank(X_series), y)
    transformed = base.transform(dense_bank(rng.normal(size=(9, 1, 100))))
    assert (fitted is None) == (transformed is None)


def test_calibration_never_flips_the_margin(rng):
    """Platt rescales a score; it does not get to reverse it.

    A negative slope turned a null ridge's anti-correlated leave-one-out margins
    into confident backwards predictions on unseen rows -- 0.422 balanced
    accuracy on a binary task, below chance and below no base at all.
    """
    margins = rng.normal(size=300)
    backwards = (margins < 0).astype(np.float64)  # labels that argue for a < 0
    a, b = _platt(margins, backwards)
    assert a >= 0.0, f"calibration fitted a negative slope ({a:.1f})"


def test_refuses_to_interpolate(rng):
    """More features than rows must not be answered with a near-zero penalty."""
    X = rng.normal(size=(40, 300))
    y = rng.normal(size=40)
    base = DenseBase("regression", 1)
    base.fit(X, y)
    assert base.effective_dof_ < 0.9 * 40


def test_learns_a_real_linear_signal(rng):
    """The counterpart to the leak test: it must still find signal when present."""
    X_static, X_series, y = make_shape_amplitude_regression(n=400, seed=0)
    _, X_series_te, y_te = make_shape_amplitude_regression(n=1000, seed=99)

    base = DenseBase("regression", 1)
    base.fit(dense_bank(X_series), y)
    predictions = base.transform(dense_bank(X_series_te))[:, 0]

    rmse = np.sqrt(np.mean((predictions - y_te) ** 2))
    trivial = np.sqrt(np.mean((y_te - y.mean()) ** 2))
    assert rmse < 0.85 * trivial


def test_transform_before_fit_is_an_error():
    with pytest.raises(RuntimeError, match="not fitted"):
        DenseBase("regression", 1).transform(np.zeros((3, 5)))


# --------------------------------------------------------------- Lévy areas


def test_levy_area_matches_the_closed_form():
    """A quarter circle has signed area ½(π/2 − 1) relative to its start."""
    theta = np.linspace(0.0, np.pi / 2, 2000)
    path = np.stack([np.cos(theta), np.sin(theta)])[None, :, :]
    area = levy_area_columns(path)[0, 0]
    assert np.isclose(area, 0.5 * (np.pi / 2 - 1), atol=1e-4)


def test_levy_area_flips_sign_when_the_path_reverses():
    theta = np.linspace(0.0, np.pi / 2, 500)
    forward = np.stack([np.cos(theta), np.sin(theta)])[None, :, :]
    swapped = forward[:, ::-1, :]  # exchange the two channels
    assert np.isclose(
        levy_area_columns(forward)[0, 0], -levy_area_columns(swapped)[0, 0], atol=1e-9
    )


def test_levy_area_is_zero_for_a_straight_line(rng):
    """Two channels moving in lockstep enclose no area."""
    t = np.linspace(0, 1, 200)
    path = np.stack([t, 2.0 * t + 1.0])[None, :, :]
    assert abs(levy_area_columns(path)[0, 0]) < 1e-9


def test_levy_area_ignores_nan_padding(rng):
    """A padded series must give the same area as the unpadded one."""
    theta = np.linspace(0.0, np.pi / 2, 128)
    path = np.stack([np.cos(theta), np.sin(theta)])[None, :, :]
    padded = np.full((1, 2, 192), np.nan)
    padded[:, :, :128] = path
    assert np.isclose(levy_area_columns(path)[0, 0], levy_area_columns(padded)[0, 0], atol=1e-9)


def test_levy_areas_are_empty_for_one_channel(rng):
    assert levy_area_columns(rng.normal(size=(5, 1, 50))).shape == (5, 0)


def test_levy_areas_detect_which_channel_leads():
    """The point of the whole feature, on the scenario built for it."""
    X_static, X_series, y = make_lead_lag(n=800, seed=0)
    a_leads = (y.astype(bool) ^ (X_static[:, 0] == 1)).astype(int)
    areas = levy_area_columns(X_series)

    full_window = areas[:, 0]
    separation = abs(full_window[a_leads == 1].mean() - full_window[a_leads == 0].mean())
    assert separation > 0.5 * full_window.std()


# ------------------------------------------------------------- integration


@pytest.mark.parametrize("flags", [
    dict(dense_base=True),
    dict(levy_areas=True),
    dict(dense_base=True, levy_areas=True),
])
def test_the_flags_fit_and_predict(flags):
    X_static, X_series, y = make_lead_lag(n=200, seed=0)
    model = HeartwoodClassifier(n_estimators=15, random_state=0, **flags)
    model.fit(X_static, X_series, y)

    proba = model.predict_proba(X_static, X_series)
    assert proba.shape == (200, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.isfinite(proba).all()


def test_extra_columns_are_named_in_the_dump():
    X_static, X_series, y = make_lead_lag(n=250, seed=0)
    model = HeartwoodClassifier(
        n_estimators=25, random_state=0, dense_base=True, levy_areas=True
    ).fit(X_static, X_series, y)

    names = " ".join(model.feature_importances())
    assert "levy_area[" in names or "dense_margin[" in names, names
    assert all("<=" in description for description, _ in model.dump_splits())


def test_dense_base_survives_regression_and_early_stopping():
    X_static, X_series, y = make_shape_amplitude_regression(n=250, seed=0)
    X_val, S_val, y_val = make_shape_amplitude_regression(n=200, seed=5)

    model = HeartwoodRegressor(
        n_estimators=60, early_stopping_rounds=5, random_state=0, dense_base=True
    ).fit(X_static, X_series, y, eval_set=(X_val, S_val, y_val))

    assert np.isfinite(model.predict(X_val, S_val)).all()
    assert model.best_iteration_ >= 0


def test_predict_uses_full_fit_margins_not_leave_one_out():
    """Train-time and predict-time bases differ by design; both must be finite."""
    X_static, X_series, y = make_shape_amplitude_regression(n=200, seed=0)
    model = HeartwoodRegressor(n_estimators=10, random_state=0, dense_base=True)
    model.fit(X_static, X_series, y)

    dense = model._core.dense_
    loo = dense.fit(dense_bank(X_series), y)
    full = dense.transform(dense_bank(X_series))
    assert np.isfinite(loo).all() and np.isfinite(full).all()
    assert not np.allclose(loo, full, atol=1e-6)


def test_dense_base_is_opt_in_and_levy_is_not():
    """The defaults follow the measurements, not the original plan.

    ``dense_base`` helps only when the temporal signal has genuine linear
    structure and hurts when it is purely interaction-based, so it stays opt-in.
    ``levy_areas`` is a no-op on single-channel data, costs nothing on
    multichannel noise, and is worth ~10 accuracy points where lead-lag matters,
    so it is on.
    """
    model = HeartwoodClassifier()
    assert model.dense_base is False
    assert model.levy_areas is True


def test_levy_areas_are_a_no_op_on_single_channel_data():
    """Which is what makes turning them on by default safe."""
    X_static, X_series, y = make_shape_amplitude_regression(n=200, seed=0)
    assert X_series.shape[1] == 1

    off = HeartwoodRegressor(n_estimators=15, random_state=0, levy_areas=False)
    on = HeartwoodRegressor(n_estimators=15, random_state=0, levy_areas=True)
    predictions = [
        m.fit(X_static, X_series, y).predict(X_static, X_series) for m in (off, on)
    ]
    assert np.array_equal(*predictions)


# ------------------------------------------------------- the rocket bank
#
# Added in V6. The bank exists because greedy per-node selection is the measured
# ceiling on shape-regime data (validation/HEADROOM.md) and a ridge over a large
# fixed bank does not select at all. These pin the properties the ridge on top
# depends on: the bank is label-free, deterministic, and frozen after fitting.

import numpy as np
import pytest

from heartwood import HeartwoodClassifier
from heartwood.rocket import ALPHA_INDICES, KERNEL_LEN, N_KERNELS, RocketBank

def dense_bank(X_series, stats=STAT_NAMES):
    """A wide feature block, purely as a fixture for the ridge tests.

    This is ``heartwood.dense.dense_bank`` verbatim.  It was deleted from the
    library after V15 measured it at +0.4 points over eight UEA datasets and
    0.015% of RMSE over five synthetic scenarios, but the ridge tests below are
    calibrated against the exact bank it produced -- their leakage thresholds sit
    within a percentage point of the boundary -- so reproducing its shape
    approximately is not good enough. Kept here, as a fixture, so a deleted
    feature does not stay alive in the library merely to serve its own tests.

    ``dyadic_windows`` is still imported from the library because
    ``levy_area_columns`` still uses it, and Levy areas were kept.
    """
    n, n_channels, T = X_series.shape
    windows = dyadic_windows(T)
    columns = []
    for channel in range(n_channels):
        base = X_series[:, channel, :]
        differenced = np.diff(base, axis=1) if T > 1 else base
        for block in (base, differenced):
            for start, end in windows:
                piece = block[:, min(start, block.shape[1] - 1): min(end, block.shape[1])]
                if piece.shape[1] == 0:
                    continue
                for stat in stats:
                    columns.append(interval_stat(piece, stat))
    return np.column_stack(columns).astype(np.float64) if columns else np.zeros((n, 0))



def test_there_are_eighty_four_zero_sum_kernels():
    assert N_KERNELS == 84
    for alpha in ALPHA_INDICES:
        weights = np.full(KERNEL_LEN, -1.0)
        weights[alpha] = 2.0
        assert weights.sum() == pytest.approx(0.0), "a kernel that does not sum to zero"
        assert (weights == 2.0).sum() == 3


def test_features_are_proportions_and_are_finite(rng):
    X = rng.normal(size=(30, 3, 90))
    features = RocketBank(n_features=1500, random_state=0).fit_transform(X)
    assert features.shape[0] == 30 and features.shape[1] > 0
    assert np.isfinite(features).all()
    assert features.min() >= 0.0 and features.max() <= 1.0


def test_the_bank_never_sees_the_labels(rng):
    """The ridge on top is only honest if the bank underneath is label-free."""
    X = rng.normal(size=(25, 2, 60))
    first = RocketBank(n_features=1000, random_state=3).fit_transform(X)
    second = RocketBank(n_features=1000, random_state=3).fit_transform(X)
    assert np.array_equal(first, second), "same data and seed must give the same bank"


def test_biases_are_frozen_after_fitting(rng):
    """A test row must be measured against the thresholds a training row was.

    Recomputing biases per batch would make a row's features depend on whichever
    other rows happened to arrive with it.
    """
    X = rng.normal(size=(40, 2, 70))
    bank = RocketBank(n_features=1000, random_state=0).fit(X)
    whole = bank.transform(X)
    piecewise = np.vstack([bank.transform(X[:9]), bank.transform(X[9:])])
    assert np.allclose(whole, piecewise)


def test_a_shifted_pattern_is_seen_the_same_way(rng):
    """PPV pools over time, so where the pattern sits should barely matter."""
    base = np.zeros((2, 1, 120))
    pattern = np.sin(np.linspace(0, 4 * np.pi, 30))
    base[0, 0, 10:40] = pattern
    base[1, 0, 70:100] = pattern
    features = RocketBank(n_features=2000, random_state=0).fit_transform(base)
    assert np.abs(features[0] - features[1]).mean() < 0.05


def test_missing_values_do_not_produce_missing_features(rng):
    """A convolution has no NaN-aware form, so the bank imputes and says so."""
    X = rng.normal(size=(20, 2, 80))
    X[3, 0, 10:30] = np.nan
    X[7, 1, :] = np.nan  # an entirely absent channel
    features = RocketBank(n_features=1000, random_state=0).fit_transform(X)
    assert np.isfinite(features).all()


def test_transform_rejects_a_different_series_shape(rng):
    bank = RocketBank(n_features=500, random_state=0).fit(rng.normal(size=(10, 2, 50)))
    with pytest.raises(ValueError, match="expected series of shape"):
        bank.transform(rng.normal(size=(4, 3, 50)))


def test_rocket_base_predicts_out_of_sample(rng):
    """End to end: the base must help on new rows, not just training ones."""
    X = rng.normal(size=(120, 2, 96))
    y = (np.abs(X[:, 0, 30:60]).max(1) > 1.9).astype(int)
    model = HeartwoodClassifier(n_estimators=40, max_depth=3, random_state=0,
                                dense_base=True,
                                n_rocket_features=2000).fit(None, X[:80], y[:80])
    held_out = (model.predict(None, X[80:]) == y[80:]).mean()
    assert held_out > 0.6, f"rocket base predicts at {held_out:.2f} on held-out rows"



def test_a_barely_positive_loo_r2_is_not_enough(rng):
    """Beating the mean is not the bar; beating chance is.

    A leave-one-out R2 of +0.008 is indistinguishable from luck, and accepting
    one cost 36 points on bump_order -- an XOR task whose series has exactly
    zero marginal correlation with the label, so any apparent signal is noise by
    construction. The permutation null is what tells those apart.
    """
    n, p = 200, 400
    X = rng.normal(size=(n, p))
    y = rng.integers(0, 2, size=n).astype(np.float64)  # nothing to find
    base = DenseBase("classification", 1)
    assert base.fit(X, y) is None
    assert base.degenerate_
    assert base.loo_r2_ <= max(0.0, base.null_r2_)


def test_real_signal_clears_the_permutation_null(rng):
    """The guard must not be so strict that it refuses a base that works."""
    n, p = 150, 300
    X = rng.normal(size=(n, p))
    y = (X[:, :8] @ rng.normal(size=8) + 0.5 * rng.normal(size=n) > 0).astype(np.float64)
    base = DenseBase("classification", 1)
    assert base.fit(X, y) is not None, "declined a design with genuine signal"
    assert base.loo_r2_ > base.null_r2_


# ------------------------------- V10: the static block joins the base


def test_loo_stays_exact_when_statics_are_unpenalised(rng):
    """The hat matrix gains a projection term; the closed form must still hold.

    This is where the ridge base nearly shipped broken once already, so it is
    checked against literally refitting without each row rather than by
    inspection. Lambda and standardisation are held fixed, since the identity is
    a statement about one linear smoother, not about re-tuning on every subset.
    """
    import heartwood.dense as module

    n, p, k = 50, 15, 3
    X = rng.normal(size=(n, p))
    Z = rng.normal(size=(n, k))
    y = Z[:, 0] * 1.5 + X[:, :3] @ rng.normal(size=3) + 0.4 * rng.normal(size=n)

    grid = module.LAMBDA_GRID
    module.LAMBDA_GRID = np.array([1.0])
    try:
        base = DenseBase("regression", 1, use_static=True)
        loo = base.fit(X, y, static=Z)
        assert loo is not None

        design = base._static_design(Z, n, fitting=False)
        bank = base._prepare(X, fitting=False)
        centred = y - base.target_center_[0]
        penalty = np.zeros(design.shape[1] + bank.shape[1])
        penalty[design.shape[1]:] = base.lambda_

        expected = []
        for i in range(n):
            keep = np.array([j for j in range(n) if j != i])
            A = np.hstack([design[keep], bank[keep]])
            beta = np.linalg.solve(A.T @ A + np.diag(penalty), A.T @ centred[keep])
            expected.append(np.hstack([design[i], bank[i]]) @ beta + base.target_center_[0])
    finally:
        module.LAMBDA_GRID = grid

    assert np.allclose(loo[:, 0], expected, atol=1e-8), (
        f"leave-one-out drifted from an explicit refit by "
        f"{np.abs(loo[:, 0] - expected).max():.2e}"
    )


def test_static_coefficients_transfer_to_unseen_rows(rng):
    """Scoring one row must give what scoring it inside a batch gives.

    An earlier version orthonormalised the static block per batch and stored
    coefficients in that basis, which are meaningless for any other set of rows.
    """
    X = rng.normal(size=(60, 10))
    Z = rng.normal(size=(60, 3))
    y = Z[:, 0] * 2.0 + 0.3 * rng.normal(size=60)
    base = DenseBase("regression", 1, use_static=True)
    base.fit(X, y, static=Z)
    whole = base.transform(X, static=Z)
    piecewise = np.vstack([base.transform(X[i:i + 1], static=Z[i:i + 1]) for i in range(60)])
    assert np.allclose(whole, piecewise)


def test_base_can_use_statics_when_the_series_is_noise(rng):
    """The point of V10: a base that sees the statics is not a null model.

    With signal only in the static block, a series-only base has nothing and
    declines; one that sees the statics should find them.
    """
    X = rng.normal(size=(200, 40))          # pure noise "series" bank
    Z = rng.normal(size=(200, 3))
    y = (Z[:, 0] + 0.3 * rng.normal(size=200) > 0).astype(np.float64)

    blind = DenseBase("classification", 1, use_static=False).fit(X, y)
    seeing = DenseBase("classification", 1, use_static=True).fit(X, y, static=Z)
    assert blind is None, "a series-only base found signal in noise"
    assert seeing is not None, "a base with the statics declined a clear signal"


def test_the_base_can_express_a_product_of_statics(rng):
    """V11: a linear base cannot fit x0*x2, which is a third of static_control.

    Without interaction columns the base should find little; with them it should
    fit the same target well.
    """
    n = 400
    Z = rng.normal(size=(n, 4))
    bank = rng.normal(size=(n, 30))                       # a useless "series" bank
    y = 1.5 * Z[:, 0] - 1.2 * Z[:, 1] + 1.0 * Z[:, 0] * Z[:, 2]

    linear = DenseBase("regression", 1, use_static=True, static_interactions=False)
    linear.fit(bank, y, static=Z)
    expanded = DenseBase("regression", 1, use_static=True, static_interactions=True)
    expanded.fit(bank, y, static=Z)

    assert expanded.static_pairs_, "interaction columns were not added"
    assert expanded.loo_r2_ > linear.loo_r2_ + 0.1, (
        f"products bought nothing: {linear.loo_r2_:.3f} -> {expanded.loo_r2_:.3f}"
    )


def test_interaction_columns_are_dropped_when_they_would_crowd_the_rows(rng):
    """The size guard is on shapes, so it must fire on shapes alone."""
    wide = DenseBase("regression", 1, use_static=True)
    wide.fit(rng.normal(size=(40, 20)), rng.normal(size=40), static=rng.normal(size=(40, 12)))
    assert wide.static_pairs_ == [], "products were added to a block already near n"


def test_interactions_still_transfer_to_single_rows(rng):
    """Whatever the basis, one row scored alone must match it scored in a batch."""
    Z = rng.normal(size=(200, 4))
    bank = rng.normal(size=(200, 20))
    y = Z[:, 0] * Z[:, 1] + 0.3 * rng.normal(size=200)
    base = DenseBase("regression", 1, use_static=True, static_interactions=True)
    base.fit(bank, y, static=Z)
    assert base.static_pairs_, "this test is meaningless without the products"
    whole = base.transform(bank, static=Z)
    piecewise = np.vstack([base.transform(bank[i:i+1], static=Z[i:i+1]) for i in range(20)])
    assert np.allclose(whole[:20], piecewise)


def test_static_interactions_are_off_by_default(rng):
    """V11 measured them and they fail; the default must not carry them.

    Products extrapolate quadratically, so a held-out subject whose statics sit
    outside the training range gets an exploding term. Apnea-ECG fell from 0.856
    AUC to 0.478 -- below chance -- because its splits are subject-disjoint and
    leave-one-out, which only sees other rows of the *same* subjects, could not
    detect it.
    """
    Z = rng.normal(size=(200, 4))
    base = DenseBase("regression", 1, use_static=True)
    base.fit(rng.normal(size=(200, 20)), Z[:, 0] * 2.0 + rng.normal(size=200), static=Z)
    assert base.static_pairs_ == [], "interaction columns are on by default"


def test_leave_one_group_out_matches_refitting_without_each_group(rng):
    """The block hat matrix must equal actually holding each group out.

    Third time this project has touched the out-of-fold machinery and the second
    time a subtle error in it produced a confident wrong answer -- the first
    version of this added only the diagonal of the static projection to each
    block, which this check caught.
    """
    import heartwood.dense as module

    n, p, k, n_groups = 60, 10, 2, 6
    groups = np.repeat(np.arange(n_groups), n // n_groups)
    Z = rng.normal(size=(n, k))
    X = rng.normal(size=(n, p))
    y = Z[:, 0] * 1.2 + X[:, :2] @ rng.normal(size=2) + 0.4 * rng.normal(size=n)

    grid = module.LAMBDA_GRID
    module.LAMBDA_GRID = np.array([1.0])
    try:
        base = DenseBase("regression", 1, use_static=True)
        out = base.fit(X, y, static=Z, groups=groups)
        assert out is not None

        bank = base._prepare(X, fitting=False)
        design = base._static_design(Z, n, fitting=False)
        penalty = np.zeros(design.shape[1] + bank.shape[1])
        penalty[design.shape[1]:] = base.lambda_
        centred = y - base.target_center_[0]

        expected = np.empty(n)
        for key in np.unique(groups):
            held = np.nonzero(groups == key)[0]
            rest = np.nonzero(groups != key)[0]
            A = np.hstack([design[rest], bank[rest]])
            beta = np.linalg.solve(A.T @ A + np.diag(penalty), A.T @ centred[rest])
            expected[held] = np.hstack([design[held], bank[held]]) @ beta
            expected[held] += base.target_center_[0]
    finally:
        module.LAMBDA_GRID = grid

    assert np.allclose(out[:, 0], expected, atol=1e-8), (
        f"group hold-out drifted from a refit by {np.abs(out[:, 0] - expected).max():.2e}"
    )


def test_one_row_per_group_is_plain_leave_one_out(rng):
    """The grouped path must not disturb the ungrouped one."""
    X = rng.normal(size=(50, 8))
    Z = rng.normal(size=(50, 2))
    y = Z[:, 0] + X[:, 0] + 0.3 * rng.normal(size=50)
    plain = DenseBase("regression", 1, use_static=True).fit(X, y, static=Z)
    singletons = DenseBase("regression", 1, use_static=True).fit(
        X, y, static=Z, groups=np.arange(50))
    assert np.array_equal(plain, singletons)


def test_rank_products_stay_bounded_far_outside_the_training_range(rng):
    """Why V11 exploded and this should not.

    A subject 20 standard deviations out must not produce an unbounded term.
    """
    Z = rng.normal(size=(200, 3))
    base = DenseBase("regression", 1, use_static=True, static_interactions=True)
    base.fit(rng.normal(size=(200, 20)), Z[:, 0] * Z[:, 1] + rng.normal(size=200), static=Z)
    assert base.static_pairs_, "no products were built, so this proves nothing"
    wild = np.full((1, 3), 20.0)
    design = base._static_design(wild, 1, fitting=False)
    products = design[:, -len(base.static_pairs_):]
    # the linear columns are allowed to grow -- linear extrapolation is graceful,
    # and that is exactly why V10 survived where V11's quadratic terms did not
    assert np.abs(products).max() <= 0.25 + 1e-9, (
        f"a far-outside row produced a product term of {np.abs(products).max():.2f}"
    )
