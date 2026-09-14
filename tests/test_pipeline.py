"""
Test suite for the LSS cap/swaption pipeline.

Run with:   python -m pytest tests -q
       or:  python tests/test_pipeline.py     (no pytest needed)

The tests that matter most are the ones that pin the model to something
independent of the implementation:

  * a rank-one covariance matrix makes every forward exactly lognormal, so the
    simulated caplet must reproduce the Black closed form;
  * the numeraire-deflated bond prices must be exact martingales;
  * the realised variance of dF/F over one step must equal Sigma_ii * delta,
    which is what pins down the Jacobian scaling of equation (11);
  * the closed-form approximation must agree with the full simulation;
  * the cap variance rule must match the paper's own prescription;
  * a rank-one fit must satisfy the Merton (1973) bound.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lss import (                                              # noqa: E402
    approx, calibrate, caps, curve as curve_mod, historical,
    instruments, mathutil, paper_data, smm, strategy, synthetic,
)
from lss.conventions import (                                  # noqa: E402
    DELTA, N_FORWARDS, SWAPTION_GRID, CAP_MATURITIES,
)


# ---------------------------------------------------------------------------
# Fixtures (plain functions, so the file runs without pytest)
# ---------------------------------------------------------------------------
def make_curve():
    return curve_mod.curve_from_forwards(paper_data.FORWARD_MEAN, libor_6m=0.0570)


def rank_one_sigma(vol: float = 0.15) -> np.ndarray:
    """All forwards perfectly correlated with constant volatility."""
    return np.full((N_FORWARDS, N_FORWARDS), vol * vol)


def realistic_sigma() -> np.ndarray:
    basis = historical.paper_factor_basis()
    return basis.covariance(np.array([0.30, 0.05, 0.02, 0.005]))


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------
def test_swaption_grid_is_34():
    assert len(SWAPTION_GRID) == 34
    assert all(e + t <= 10.0 for e, t in SWAPTION_GRID)
    # Table 3 has six expiries for tenors 1-5 and four for tenor 7.
    assert sum(1 for _, t in SWAPTION_GRID if t == 7.0) == 4


def test_caplet_schedule_omits_first_caplet():
    # A five-year cap on six-month Libor has nine caplets, paying 1.0 .. 5.0.
    schedule = instruments.caplet_schedule(5.0)
    assert len(schedule) == 9
    assert schedule[0] == (0.5, 1.0)
    assert schedule[-1] == (4.5, 5.0)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------
def test_normal_functions():
    x = np.linspace(-6, 6, 501)
    # phi is the derivative of Phi
    numerical = np.gradient(mathutil.norm_cdf(x), x)
    # Central differences on a 0.024 grid: truncation error is O(h^2) ~ 4e-5,
    # so this bounds the truncation, not the accuracy of norm_cdf itself.
    assert np.max(np.abs(numerical - mathutil.norm_pdf(x))) < 1e-4
    # ppf inverts cdf
    p = np.linspace(1e-6, 1 - 1e-6, 501)
    assert np.max(np.abs(mathutil.norm_cdf(mathutil.norm_ppf(p)) - p)) < 1e-8


def test_cubic_spline_interpolates_knots():
    x = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    y = np.array([0.057, 0.0585, 0.060, 0.0612, 0.0630, 0.0642, 0.0655])
    spline = mathutil.CubicSpline(x, y)
    assert np.max(np.abs(spline(x) - y)) < 1e-12


def test_least_squares_solves_a_known_problem():
    target = np.array([1.0, 2.0, 3.0])

    def residuals(p):
        return p - target

    fit = mathutil.least_squares(residuals, np.zeros(3))
    assert np.max(np.abs(fit.x - target)) < 1e-8


def test_least_squares_respects_lower_bound():
    def residuals(p):
        return p - np.array([-1.0, 2.0])

    fit = mathutil.least_squares(residuals, np.array([0.5, 0.5]), lower=0.0)
    assert fit.x[0] >= 0.0
    assert abs(fit.x[1] - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# Curve
# ---------------------------------------------------------------------------
def test_forward_discount_round_trip():
    curve = make_curve()
    recovered = curve_mod.discounts_to_forwards(curve.discounts)
    assert np.max(np.abs(recovered - paper_data.FORWARD_MEAN)) < 1e-12


def test_discounts_are_positive_and_decreasing():
    curve = make_curve()
    assert np.all(curve.discounts > 0)
    assert np.all(np.diff(curve.discounts) < 0)


def test_par_swap_rate_reprices_par():
    """A swap struck at the par rate must have zero value."""
    curve = make_curve()
    for maturity in (2.0, 5.0, 10.0):
        rate = curve.par_swap_rate(maturity)
        value = instruments.forward_swap_value(curve, 0.0, maturity, rate)
        assert abs(value) < 1e-12


def test_forward_swap_rate_makes_forward_swap_worthless():
    curve = make_curve()
    for start, end in ((1.0, 5.0), (2.0, 10.0), (5.0, 10.0)):
        rate = curve.forward_swap_rate(start, end)
        value = instruments.forward_swap_value(curve, start, end, rate)
        assert abs(value) < 1e-12


def test_build_curve_from_quotes():
    swaps = {2.0: 0.058, 3.0: 0.0595, 4.0: 0.0605, 5.0: 0.0615,
             7.0: 0.0628, 10.0: 0.0640}
    curve = curve_mod.build_curve(0.0570, 0.0578, swaps)
    assert curve.forwards.shape == (N_FORWARDS,)
    assert np.all(curve.forwards > 0)
    # The bootstrap must reprice the input pillars.
    for maturity, quoted in swaps.items():
        assert abs(curve.par_swap_rate(maturity) - quoted) < 1e-10


# ---------------------------------------------------------------------------
# Black formulas
# ---------------------------------------------------------------------------
def test_atmf_swaption_formulas_agree():
    """Equation (7) must equal equation (6) struck at the forward swap rate."""
    curve = make_curve()
    for expiry, tenor in ((1.0, 5.0), (2.0, 3.0), (5.0, 5.0)):
        a = instruments.black_swaption_atmf(curve, expiry, tenor, 0.16)
        b = instruments.black_swaption_price(curve, expiry, tenor, 0.16)
        assert abs(a - b) < 1e-12


def test_put_call_parity_for_swaptions():
    curve = make_curve()
    expiry, tenor, coupon = 2.0, 5.0, 0.055
    payer = instruments.black_swaption_price(
        curve, expiry, tenor, 0.16, coupon, payer=True
    )
    receiver = instruments.black_swaption_price(
        curve, expiry, tenor, 0.16, coupon, payer=False
    )
    swap = instruments.forward_swap_value(curve, expiry, expiry + tenor, coupon)
    assert abs((receiver - payer) - swap) < 1e-12


def test_swaption_implied_vol_inverts():
    curve = make_curve()
    for vol in (0.10, 0.16, 0.25):
        price = instruments.black_swaption_atmf(curve, 2.0, 5.0, vol)
        assert abs(instruments.swaption_implied_vol(curve, 2.0, 5.0, price) - vol) < 1e-8


def test_cap_implied_vol_inverts():
    curve = make_curve()
    for vol in (0.12, 0.16, 0.22):
        price = instruments.black_cap_price(curve, 5.0, vol)
        assert abs(instruments.cap_implied_vol(curve, 5.0, price) - vol) < 1e-8


def test_cap_price_increases_with_vol():
    curve = make_curve()
    prices = [instruments.black_cap_price(curve, 5.0, v) for v in (0.10, 0.15, 0.20)]
    assert prices[0] < prices[1] < prices[2]


# ---------------------------------------------------------------------------
# The string market model
# ---------------------------------------------------------------------------
def test_simulation_is_a_martingale():
    """
    Numeraire-deflated bond prices must have expectation equal to today's
    price.  This is the discrete no-arbitrage condition and catches any error
    in the drift.
    """
    curve = make_curve()
    result = smm.simulate(
        curve, realistic_sigma(),
        smm.SimulationConfig(n_paths=20000, seed=4, max_steps=10),
    )
    errors = smm.martingale_diagnostic(result, curve)
    assert np.max(np.abs(errors)) < 5e-3


def test_jacobian_scaling_reproduces_sigma():
    """
    The realised variance of dF/F over one six-month step must equal
    Sigma_ii * delta.  This is the check that pins down the factor of delta in
    the Jacobian of equation (11), which the paper's printed matrix omits.
    """
    curve = make_curve()
    vol = 0.15
    result = smm.simulate(
        curve, rank_one_sigma(vol),
        smm.SimulationConfig(n_paths=40000, seed=5, max_steps=1),
    )
    before = smm.forwards_from_state(result.states[0])
    after = smm.forwards_from_state(result.states[1])
    realised = np.var(after / before[:, 1:] - 1.0, axis=0)
    target = vol * vol * DELTA
    # Converges to about 1.01 as paths increase; the residual one to three
    # percent is genuine second-order Euler error over a six-month step at a
    # fifteen percent volatility.  The error this test exists to catch -- using
    # the paper's printed Jacobian, which omits the 1/delta factor -- would
    # show up as a ratio of 0.25 or 4.0, so a five percent band still catches
    # it by a factor of thirty.
    assert np.max(np.abs(realised / target - 1.0)) < 0.05


def test_rank_one_simulation_matches_black_caplet():
    """
    With a rank-one Sigma every forward is exactly lognormal, so the Black
    caplet formula is exact and the simulation must reproduce it.
    """
    curve = make_curve()
    vol = 0.15
    result = smm.simulate(
        curve, rank_one_sigma(vol),
        smm.SimulationConfig(n_paths=100000, seed=6, max_steps=10),
    )
    strike = curve.par_swap_rate(5.0)
    for k in (1, 2, 4, 6, 10):
        reset = k * DELTA
        state = result.discounts_at(reset)
        libor = (1.0 / state[:, 0] - 1.0) / DELTA
        payoff = DELTA * np.maximum(libor - strike, 0.0) * state[:, 0]
        simulated = float(np.mean(payoff * result.numeraire_at(reset)))
        closed_form = instruments.black_caplet_price(curve, reset, strike, vol)
        relative = abs(simulated / closed_form - 1.0)
        # After a single Euler step the simulated forward is still nearly
        # normal rather than lognormal, which thins the right tail and
        # underprices this out-of-the-money caplet by around three percent.
        # From two steps on the log-CLT takes over and the error collapses.
        assert relative < (0.05 if k == 1 else 0.015)


def test_jacobian_solve_matches_dense_solve():
    """The banded forward substitution must equal a dense linear solve."""
    curve = make_curve()
    state = np.broadcast_to(curve.discounts, (3, 20)).copy()
    state[1] *= np.linspace(1.0, 0.99, 20)
    state[2] *= np.linspace(1.0, 1.01, 20)

    diag, sub = smm.jacobian_bands(state)
    rhs = np.linspace(1e-4, 5e-4, 19)[None, :] * np.ones((3, 1))
    fast = smm.solve_jacobian(diag, sub, rhs)

    for path in range(3):
        dense = np.zeros((19, 19))
        for r in range(19):
            dense[r, r] = diag[path, r]
            if r >= 1:
                dense[r, r - 1] = sub[path, r]
        assert np.allclose(np.linalg.solve(dense, rhs[path]), fast[path])


def test_simulated_swaption_prices_are_sensible():
    curve = make_curve()
    result = smm.simulate(curve, realistic_sigma(),
                          smm.SimulationConfig(n_paths=4000, seed=8))
    prices = smm.price_swaption_grid(result, curve, SWAPTION_GRID)
    assert np.all(prices > 0)
    vols = [
        instruments.swaption_implied_vol(curve, e, t, p)
        for (e, t), p in zip(SWAPTION_GRID, prices)
    ]
    # Implied vols should land in a plausible range, not at the bisection rails.
    assert all(0.05 < v < 0.40 for v in vols)


# ---------------------------------------------------------------------------
# The approximation
# ---------------------------------------------------------------------------
def test_caplet_variance_matches_paper_rule():
    """
    "the variance for the caplet maturing in six months is the first diagonal
     element, the variance for the caplet maturing in twelve months is the
     average of the first and second diagonal elements, etc."
    """
    sigma = realistic_sigma()
    variances = approx.caplet_variances(sigma)
    diagonal = np.diag(sigma)
    for i in range(1, 20):
        assert abs(variances[i - 1] - diagonal[:i].mean()) < 1e-14


def test_integrated_covariance_diagonal_matches_caplet_rule():
    sigma = realistic_sigma()
    for n in (1, 3, 6, 10):
        cov = approx.integrated_covariance(sigma, n)
        expected = DELTA * sum(np.diag(sigma)[n - 1 - k] for k in range(n))
        assert abs(cov[n - 1, n - 1] - expected) < 1e-14


def test_approximation_matches_simulation():
    """The closed-form approximation of footnote 18 must track the simulation."""
    curve = make_curve()
    sigma = realistic_sigma()
    result = smm.simulate(curve, sigma, smm.SimulationConfig(n_paths=20000, seed=9))
    simulated = smm.price_swaption_grid(result, curve, SWAPTION_GRID)
    approximate = approx.approx_swaption_grid(curve, sigma, SWAPTION_GRID)
    relative = approximate / simulated - 1.0
    assert np.max(np.abs(relative)) < 0.02
    assert np.sqrt(np.mean(relative ** 2)) < 0.01


def test_approximator_matches_direct_call():
    curve = make_curve()
    sigma = realistic_sigma()
    direct = approx.approx_swaption_grid(curve, sigma, SWAPTION_GRID)
    cached = approx.SwaptionApproximator(curve, SWAPTION_GRID).prices(sigma)
    assert np.allclose(direct, cached)


def test_forward_swap_weights_match_finite_differences():
    curve = make_curve()
    expiry, tenor = 2.0, 5.0
    analytic = approx.forward_swap_weights(curve, expiry, tenor)

    base = curve.forwards.copy()
    numerical = np.zeros(N_FORWARDS)
    step = 1e-7
    for i in range(N_FORWARDS):
        bumped = base.copy()
        bumped[i] += step
        up = curve_mod.curve_from_forwards(bumped, 0.0570)
        bumped[i] -= 2 * step
        down = curve_mod.curve_from_forwards(bumped, 0.0570)
        numerical[i] = (
            up.forward_swap_rate(expiry, expiry + tenor)
            - down.forward_swap_rate(expiry, expiry + tenor)
        ) / (2 * step)

    assert np.max(np.abs(analytic - numerical)) < 1e-5


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def test_factor_basis_is_orthonormal():
    basis = historical.paper_factor_basis()
    identity = basis.eigenvectors.T @ basis.eigenvectors
    assert np.max(np.abs(identity - np.eye(N_FORWARDS))) < 1e-12
    # Eigenvalues of a 19x19 correlation matrix sum to 19.
    assert abs(basis.eigenvalues.sum() - N_FORWARDS) < 1e-10
    assert np.all(np.diff(basis.eigenvalues) <= 1e-12)


def test_covariance_from_psi_has_requested_rank():
    basis = historical.paper_factor_basis()
    sigma = basis.covariance(np.array([0.3, 0.1, 0.05, 0.01]))
    assert np.linalg.matrix_rank(sigma, tol=1e-10) == 4
    assert np.all(np.linalg.eigvalsh(sigma) > -1e-12)


def test_rank_one_sigma_gives_unit_correlations():
    basis = historical.paper_factor_basis()
    sigma = basis.covariance(np.array([0.3]))
    correlation = historical.to_correlation(sigma)
    finite = correlation[np.isfinite(correlation)]
    assert np.max(np.abs(np.abs(finite) - 1.0)) < 1e-9


def test_calibration_recovers_a_known_sigma():
    """
    Generate swaption prices from a known Psi, then check the calibration
    finds it.  This is the identification test: it says the implied
    eigenvalues are actually pinned down by the swaption cross-section.
    """
    curve = make_curve()
    basis = historical.paper_factor_basis()
    true_psi = np.array([0.32, 0.09, 0.03, 0.012])

    config = smm.SimulationConfig(n_paths=4000, seed=21)
    result = smm.simulate(curve, basis.covariance(true_psi), config)
    prices = smm.price_swaption_grid(result, curve, SWAPTION_GRID)

    fitted = calibrate.calibrate(
        curve, prices, n_factors=4, basis=basis, sim_config=config
    )
    assert fitted.rmse < 0.01
    assert np.max(np.abs(fitted.psi - true_psi)) < 0.02


def test_calibration_improves_with_more_factors():
    """
    The fit must not get worse as rank increases -- the Table 5 likelihood
    ratio tests compare sums of squared errors across ranks, so a
    non-monotone fit would invalidate them.
    """
    curve = make_curve()
    vols = np.array(
        [paper_data.market_swaption_vol_means()[k] for k in SWAPTION_GRID]
    )
    prices = calibrate.swaption_prices_from_vols(curve, vols)

    previous = np.inf
    for n in (1, 2, 3, 4):
        fit = calibrate.calibrate(curve, prices, n_factors=n)
        assert fit.rmse <= previous + 1e-6
        previous = fit.rmse


def test_calibration_eigenvalues_are_non_negative():
    curve = make_curve()
    vols = np.array(
        [paper_data.market_swaption_vol_means()[k] for k in SWAPTION_GRID]
    )
    prices = calibrate.swaption_prices_from_vols(curve, vols)
    fit = calibrate.calibrate(curve, prices, n_factors=4)
    assert np.all(fit.psi >= 0.0)


def test_approx_and_simulation_calibration_agree():
    """The accelerated scheme must reach the same optimum as the slow one."""
    curve = make_curve()
    vols = np.array(
        [paper_data.market_swaption_vol_means()[k] for k in SWAPTION_GRID]
    )
    prices = calibrate.swaption_prices_from_vols(curve, vols)
    config = smm.SimulationConfig(n_paths=2000, seed=31)

    fast = calibrate.calibrate(curve, prices, n_factors=3, sim_config=config)
    slow = calibrate.calibrate(
        curve, prices, n_factors=3, method="simulation", sim_config=config
    )
    # Agreement is asserted on the objective, not on Psi or on individual
    # prices.  At rank three the market surface is not spanned by the model and
    # the objective has several near-equivalent minima: [.29, .15, .05] and
    # [.32, .22, .00] both give an RMSE of about 5.1% while differing by up to
    # 21% on individual swaptions.  So the two engines legitimately land on
    # different optima of equal quality, and only the attained fit is
    # comparable.  Psi *is* well identified when the data really is generated
    # by the model -- see test_calibration_recovers_a_known_sigma, which pins
    # all four eigenvalues to within 0.02.
    assert abs(fast.rmse - slow.rmse) < 0.005


# ---------------------------------------------------------------------------
# Caps and the Merton bound
# ---------------------------------------------------------------------------
def test_cap_price_from_sigma_is_positive_and_monotone():
    curve = make_curve()
    sigma = realistic_sigma()
    prices = caps.model_cap_prices(curve, sigma)
    assert np.all(prices > 0)
    assert np.all(np.diff(prices) > 0)      # longer caps have more caplets


def test_rank_one_cap_price_is_a_merton_lower_bound():
    """
    A portfolio of options is worth at least an option on the portfolio, with
    equality under perfect correlation.  So the cap price from a rank-one fit
    must not exceed the price from a higher-rank fit of the same swaptions.
    """
    curve = make_curve()
    vols = np.array(
        [paper_data.market_swaption_vol_means()[k] for k in SWAPTION_GRID]
    )
    prices = calibrate.swaption_prices_from_vols(curve, vols)

    fit1 = calibrate.calibrate(curve, prices, n_factors=1)
    fit4 = calibrate.calibrate(curve, prices, n_factors=4)

    bound = caps.model_cap_prices(curve, fit1.sigma)
    full = caps.model_cap_prices(curve, fit4.sigma)
    assert np.all(bound <= full * (1.0 + 1e-9))


def test_model_cap_vols_round_trip():
    curve = make_curve()
    sigma = realistic_sigma()
    vols = caps.model_cap_vols(curve, sigma)
    prices = caps.model_cap_prices(curve, sigma)
    for T, v, p in zip(CAP_MATURITIES, vols, prices):
        assert abs(instruments.black_cap_price(curve, T, v) - p) < 1e-10


def test_value_caps_error_sign():
    """Positive percentage error must mean the cap is cheap versus the model."""
    curve = make_curve()
    sigma = realistic_sigma()
    model_vols = caps.model_cap_vols(curve, sigma)
    # Quote the market a full vol point below the model: caps look cheap.
    valuation = caps.value_caps(curve, sigma, model_vols - 0.01)
    assert np.all(valuation.percentage_errors > 0)
    assert np.all(valuation.vol_spread > 0)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def test_synthetic_data_has_paper_like_statistics():
    market, wedge = synthetic.generate_market_data()
    assert market.n_weeks == 128                      # matches the paper
    assert market.forwards.shape == (128, N_FORWARDS)
    assert market.swaption_vols.shape == (128, 34)
    assert market.cap_vols.shape == (128, 6)
    assert np.all(market.forwards > 0)
    assert np.all(market.swaption_vols > 0)
    assert np.all(market.cap_vols > 0)

    # Forward levels should sit near the Table 1 means.
    assert np.max(np.abs(market.forwards.mean(axis=0) - paper_data.FORWARD_MEAN)) < 0.01
    # Ten crisis weeks, per Section 5.3.
    assert market.crisis_mask().sum() == 10
    assert wedge.shape == (128, 6)


def test_synthetic_curves_build_cleanly():
    market, _ = synthetic.generate_market_data()
    for week in (0, 50, 127):
        curve = market.curve(week)
        assert np.all(np.diff(curve.discounts) < 0)
        assert 0.0 < curve.par_swap_rate(10.0) < 0.20


def test_no_wedge_option_injects_nothing():
    _, wedge = synthetic.generate_market_data(with_wedge=False)
    assert np.max(np.abs(wedge)) < 1e-12


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
_SMALL_FIT_CACHE: tuple | None = None


def _small_fit():
    """
    A short calibrated sample shared by the strategy tests.

    Cached: several tests need the same fit, it costs a full calibration to
    build, and nothing downstream mutates it.
    """
    global _SMALL_FIT_CACHE
    if _SMALL_FIT_CACHE is None:
        from lss import empirics
        market, wedge = synthetic.generate_market_data(end="1998-06-26")
        fit = empirics.fit_sample(
            market, n_factors=4,
            sim_config=smm.SimulationConfig(n_paths=500, seed=41),
        )
        _SMALL_FIT_CACHE = (fit, wedge)
    return _SMALL_FIT_CACHE


def test_causal_zscore_uses_no_future_information():
    """
    The z-score at week t must be unchanged by anything after week t.  This is
    the leak that most easily manufactures a fake backtest.
    """
    import pandas as pd
    dates = synthetic.weekly_dates("1997-01-24", "1999-07-02")
    values = np.random.default_rng(3).standard_normal((len(dates), 2))
    frame = pd.DataFrame(values, index=dates, columns=["a", "b"])

    full = strategy.causal_zscore(frame)
    truncated = strategy.causal_zscore(frame.iloc[:80])
    common = truncated.index
    assert np.allclose(
        full.loc[common].to_numpy(), truncated.to_numpy(), equal_nan=True
    )


def test_positions_respect_entry_and_exit_bands():
    import pandas as pd
    z = pd.DataFrame(
        {"a": [np.nan, 0.0, 1.5, 0.9, 0.4, -0.1, -1.6, -0.2, 0.05]},
        index=range(9),
    )
    config = strategy.StrategyConfig(entry_z=1.25, exit_z=0.25)
    positions = strategy.target_positions(z, config)["a"].tolist()
    #  nan -> flat; 0.0 inside exit band -> flat; 1.5 -> long; 0.9 and 0.4 are
    #  inside the entry threshold but outside the exit band, so the long is
    #  held; -0.1 is inside the exit band, so it flattens; -1.6 -> short;
    #  -0.2 inside the exit band -> flat; 0.05 -> flat.
    assert positions == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, -1.0, 0.0, 0.0]


def test_hedge_basket_is_bounded():
    basket = strategy.swaption_hedge_basket(10.0, SWAPTION_GRID, max_instruments=3)
    assert len(basket) == 3
    basket = strategy.swaption_hedge_basket(2.0, SWAPTION_GRID, max_instruments=3)
    assert 1 <= len(basket) <= 3
    assert all(SWAPTION_GRID[k][0] + SWAPTION_GRID[k][1] <= 2.0 for k in basket)


def test_hedge_weights_are_vega_neutral():
    curve = make_curve()
    vols = np.array(
        [paper_data.market_swaption_vol_means()[k] for k in SWAPTION_GRID]
    )
    basket = strategy.swaption_hedge_basket(5.0, SWAPTION_GRID, 3)
    weights = strategy.hedge_weights(
        curve, 5.0, 0.168, SWAPTION_GRID, vols, basket
    )
    hedge_vega = sum(
        w * instruments.black_swaption_vega_atmf(
            curve, SWAPTION_GRID[k][0], SWAPTION_GRID[k][1], vols[k]
        )
        for w, k in zip(weights, basket)
    )
    cap_vega = instruments.black_cap_vega(curve, 5.0, 0.168)
    assert abs(hedge_vega / cap_vega - 1.0) < 1e-9


def test_backtest_runs_and_is_self_consistent():
    fit, wedge = _small_fit()
    result = strategy.backtest(fit)
    assert result.pnl.shape == (fit.n_weeks, len(CAP_MATURITIES))
    # net = gross - costs, exactly
    assert np.allclose(
        result.pnl.to_numpy(),
        result.gross_pnl.to_numpy() - result.costs.to_numpy(),
    )
    # No position can be taken before the z-score has enough history.
    warmup = strategy.StrategyConfig().min_history
    assert np.all(result.positions.to_numpy()[: warmup - 1] == 0.0)

    stats = strategy.summarize(result)
    assert "Total" in stats.index
    assert np.isfinite(stats.loc["Total", "gross_pnl"])


def test_zero_cost_backtest_has_no_costs():
    fit, _ = _small_fit()
    result = strategy.backtest(
        fit, strategy.StrategyConfig(vol_point_cost=0.0)
    )
    assert np.allclose(result.costs.to_numpy(), 0.0)
    assert np.allclose(result.pnl.to_numpy(), result.gross_pnl.to_numpy())


def test_unit_economics_is_config_independent():
    """
    The weekly repricing must not depend on the trading rules -- that is what
    lets `sensitivity` reprice once and sweep many configurations.
    """
    fit, _ = _small_fit()
    economics = strategy.unit_economics(fit)

    for config in (
        strategy.StrategyConfig(entry_z=0.5, exit_z=0.1),
        strategy.StrategyConfig(entry_z=2.0, vol_point_cost=0.0),
    ):
        cached = strategy.backtest(fit, config, economics)
        fresh = strategy.backtest(fit, config)
        assert np.allclose(cached.pnl.to_numpy(), fresh.pnl.to_numpy())


def test_costs_scale_linearly_with_bid_offer():
    fit, _ = _small_fit()
    economics = strategy.unit_economics(fit)
    one = strategy.backtest(
        fit, strategy.StrategyConfig(vol_point_cost=1.0), economics
    )
    two = strategy.backtest(
        fit, strategy.StrategyConfig(vol_point_cost=2.0), economics
    )
    assert np.allclose(2.0 * one.costs.to_numpy(), two.costs.to_numpy())
    assert np.allclose(one.gross_pnl.to_numpy(), two.gross_pnl.to_numpy())


def test_null_data_still_produces_a_spread():
    """
    The control that matters.  With no dislocation injected, the recovered
    spread does NOT collapse to zero -- it retains most of its variance, which
    is estimation noise from the calibration rather than any real wedge.

    This is why a positive backtest P&L on synthetic data is not by itself
    evidence: a mean-reversion rule earns in sample from persistent estimation
    noise reverting.  The strategy must always be run against `--no-wedge` as a
    control before any P&L is believed.
    """
    from lss import empirics
    market, wedge = synthetic.generate_market_data(
        end="1998-06-26", with_wedge=False
    )
    assert np.max(np.abs(wedge)) == 0.0

    fit = empirics.fit_sample(
        market, n_factors=4, sim_config=smm.SimulationConfig(n_paths=500, seed=41)
    )
    spread = strategy.wedge_spread(fit)
    # Not degenerate: the null spread is a substantial fraction of a vol point.
    assert spread.std().min() > 0.2
    # And persistent, which is exactly what makes it look tradable.
    assert spread.apply(lambda c: c.autocorr(1)).mean() > 0.3


def test_signal_recovers_the_injected_wedge():
    """
    The sharpest end-to-end test: calibrate to swaptions, price caps off the
    implied Sigma, and check the recovered spread correlates positively with
    the dislocation the generator actually injected.
    """
    fit, wedge = _small_fit()
    result = strategy.backtest(fit)
    diagnostics = strategy.signal_diagnostics(result, wedge)
    assert diagnostics["corr_with_true_wedge"].mean() > 0.2


# ---------------------------------------------------------------------------
# Runner (so the file works without pytest installed)
# ---------------------------------------------------------------------------
def _run_all() -> int:
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures.append((name, f"assertion failed: {exc}"))
            print(f"  FAIL  {name}")
        except Exception as exc:                      # noqa: BLE001
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")

    print()
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    for name, reason in failures:
        print(f"  {name}: {reason}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
