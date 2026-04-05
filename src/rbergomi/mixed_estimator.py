# src/rbergomi/mixed_estimator.py
"""
Mixed Estimator (Turbocharging) for rBergomi Inverse Option Pricing
====================================================================

Implements the conditional Black-Scholes approach from McCrickerd &
Pakkanen (2018), "Turbocharging Monte Carlo pricing for the rough
Bergomi model".

Key idea
--------
Under the rBergomi model the log-price decomposes as

    log(S_T / S_0) = -½ ∫₀ᵀ V_t dt
                     + ρ ∫₀ᵀ √V_t dW_t^⊥
                     + √(1−ρ²) ∫₀ᵀ √V_t dW_t^⊥⊥

where W^⊥ drives the variance and W^⊥⊥ is independent of the variance
path.  Conditioned on the variance path {V_t} (and hence on the first
two terms), the only remaining randomness comes from the third integral,
which is Gaussian with mean 0 and variance (1−ρ²)∫₀ᵀ V_t dt.

Therefore, conditional on the variance path,

    S_T | {V_t} ~ LogNormal(μ_c, σ_c²)

with
    μ_c   = −½ ∫₀ᵀ V_t dt + ρ ∫₀ᵀ √V_t dW_t^⊥
    σ_c²  = (1 − ρ²) ∫₀ᵀ V_t dt

and the option price reduces to a Black-Scholes evaluation per path:

    E[g(S_T)] = E[ E[g(S_T) | {V_t}] ]
              = (1/N) Σₘ BS_inverse(F_m, K, σ_c^(m))

where  F_m = S₀ exp(−½ρ² I_V^(m) + ρ I_corr^(m))  is the conditional
forward price and σ_c^(m) = √((1−ρ²) I_V^(m)) the conditional total
volatility.

This eliminates the spot-path Monte Carlo noise entirely, since each
conditional price is exact (up to numerical precision of the normal CDF).
The only remaining noise comes from sampling variance paths.

Usage
-----
The module is used internally by ``rBergomiPricer`` when
``pricing_method='mixed'``.  It can also be used standalone::

    from src.rbergomi.mixed_estimator import MixedEstimator
    me = MixedEstimator()
    prices, stds = me.price_inverse_options(
        S0=50000, strikes=np.array([48000, 50000, 52000]),
        var_paths=var_paths, dW_driving=dW, dt=dt, rho=-0.7
    )


References
----------
McCrickerd, R. & Pakkanen, M. (2018). "Turbocharging Monte Carlo
pricing for the rough Bergomi model", Quantitative Finance, 18(11).
"""

import numpy as np
from typing import Tuple, Optional
import logging
import warnings

# Numba imports with fallback
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. Mixed Estimator will use NumPy fallback.")
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args or callable(args[0]) else decorator(args[0])
    prange = range

logger = logging.getLogger("rBergomi.mixed_estimator")


# =============================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS
# =============================================================================

@njit(cache=True, fastmath=True)
def _norm_cdf_scalar(x: float) -> float:
    """
    Standard normal CDF via rational approximation (Abramowitz & Stegun 7.1.26).

    Uses erfc approximation: Φ(z) = ½ erfc(−z/√2).
    Absolute error < 1.5e-7 for all x.
    """
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    sign = 1.0 if x >= 0.0 else -1.0
    # The A&S coefficients approximate erfc(y), so transform: y = |x|/√2
    y = abs(x) * 0.7071067811865476  # 1/sqrt(2)
    t = 1.0 / (1.0 + p * y)
    erfc_approx = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-y * y)
    return 0.5 * (1.0 + sign * (1.0 - erfc_approx))


@njit(cache=True, fastmath=True, parallel=True)
def _compute_integrated_quantities(
    var_paths: np.ndarray,
    dW_driving: np.ndarray,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute integrated variance and correlation stochastic integral per path.

    For each MC path m:
        I_V^(m)    = Σᵢ V_{tᵢ}^(m) · Δt           (integrated variance)
        I_corr^(m) = Σᵢ √V_{tᵢ}^(m) · √Δt · ε_i   (∫ √V dW^⊥)

    where ε_i are the standard normal increments driving W^⊥.

    Parameters
    ----------
    var_paths : np.ndarray, shape (n_paths, n_steps)
        Simulated variance paths from the Hybrid Scheme.
    dW_driving : np.ndarray, shape (n_paths, n_steps - 1)
        Standard normal increments of the Brownian motion W^⊥ that drives
        the variance process.
    dt : float
        Time step size Δt.

    Returns
    -------
    I_V : np.ndarray, shape (n_paths,)
        Integrated variance per path.
    I_corr : np.ndarray, shape (n_paths,)
        Stochastic integral ∫√V dW^⊥ per path.
    """
    n_paths = var_paths.shape[0]
    n_increments = dW_driving.shape[1]
    sqrt_dt = np.sqrt(dt)

    I_V = np.empty(n_paths, dtype=np.float64)
    I_corr = np.empty(n_paths, dtype=np.float64)

    for m in prange(n_paths):
        iv = 0.0
        ic = 0.0
        for i in range(n_increments):
            V_i = max(var_paths[m, i], 1e-12)
            iv += V_i * dt
            ic += np.sqrt(V_i) * sqrt_dt * dW_driving[m, i]
        I_V[m] = iv
        I_corr[m] = ic

    return I_V, I_corr


@njit(cache=True, fastmath=True, parallel=True)
def _mixed_estimator_prices_multi_strike(
    S0: float,
    strikes: np.ndarray,
    I_V: np.ndarray,
    I_corr: np.ndarray,
    rho: float,
    is_put: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute inverse option prices for multiple strikes via conditional BS.

    For each path m and strike K:
        1. Conditional forward:  F_m = S₀ exp(−½ρ² I_V^(m) + ρ I_corr^(m))
        2. Conditional total vol: σ_c = √((1−ρ²) I_V^(m))
        3. BS inverse price using the conditional lognormal distribution.

    The inverse call payoff is max(S_T − K, 0)/S_T, priced via:
        C_inv^(m) = (F_m/S₀) Φ(d₁) − (K/S₀) Φ(d₂)

    The inverse put payoff is max(K − S_T, 0)/S_T, priced via:
        P_inv^(m) = (K/S₀) Φ(−d₂) − (F_m/S₀) Φ(−d₁)

    where  d₁ = [ln(F_m/K) + ½σ_c²] / σ_c,  d₂ = d₁ − σ_c.

    Parameters
    ----------
    S0 : float
        Current spot price.
    strikes : np.ndarray, shape (n_strikes,)
        Strike prices.
    I_V : np.ndarray, shape (n_paths,)
        Integrated variance per path.
    I_corr : np.ndarray, shape (n_paths,)
        Stochastic integral ∫√V dW^⊥ per path.
    rho : float
        Spot-variance correlation.
    is_put : bool
        True for inverse put, False for inverse call.

    Returns
    -------
    prices : np.ndarray, shape (n_strikes,)
        MC-estimated inverse option prices in BTC.
    std_errs : np.ndarray, shape (n_strikes,)
        Standard errors of the price estimates.
    """
    n_paths = len(I_V)
    n_strikes = len(strikes)
    rho_sq = rho * rho
    one_minus_rho_sq = 1.0 - rho_sq

    prices = np.empty(n_strikes, dtype=np.float64)
    std_errs = np.empty(n_strikes, dtype=np.float64)

    # Pre-compute per-path quantities (shared across strikes)
    log_F_over_S0 = np.empty(n_paths, dtype=np.float64)
    sigma_c = np.empty(n_paths, dtype=np.float64)
    F_over_S0 = np.empty(n_paths, dtype=np.float64)

    for m in range(n_paths):
        iv = I_V[m]
        ic = I_corr[m]
        log_F_over_S0[m] = -0.5 * rho_sq * iv + rho * ic
        sigma_c[m] = np.sqrt(max(one_minus_rho_sq * iv, 1e-15))
        F_over_S0[m] = np.exp(log_F_over_S0[m])

    for k in range(n_strikes):
        K = strikes[k]
        log_K_over_S0 = np.log(K / S0)
        K_over_S0 = K / S0

        # Accumulate payoffs across paths (parallelised)
        payoff_sum = 0.0
        payoff_sq_sum = 0.0

        for m in prange(n_paths):
            sc = sigma_c[m]

            if sc < 1e-10:
                # Degenerate: zero residual vol → intrinsic value
                fos = F_over_S0[m]
                if is_put:
                    cond_price = max(K_over_S0 - fos, 0.0)
                else:
                    cond_price = max(fos - K_over_S0, 0.0)
            else:
                d1 = (log_F_over_S0[m] - log_K_over_S0 + 0.5 * sc * sc) / sc
                d2 = d1 - sc
                fos = F_over_S0[m]

                if is_put:
                    cond_price = K_over_S0 * _norm_cdf_scalar(-d2) - fos * _norm_cdf_scalar(-d1)
                else:
                    cond_price = fos * _norm_cdf_scalar(d1) - K_over_S0 * _norm_cdf_scalar(d2)

            # Ensure non-negative (numerical guard)
            cond_price = max(cond_price, 0.0)

            payoff_sum += cond_price
            payoff_sq_sum += cond_price * cond_price

        mean_price = payoff_sum / n_paths
        var_price = payoff_sq_sum / n_paths - mean_price * mean_price
        prices[k] = mean_price
        std_errs[k] = np.sqrt(max(var_price, 0.0) / n_paths)

    return prices, std_errs


@njit(cache=True, fastmath=True, parallel=True)
def _mixed_estimator_single_strike(
    S0: float,
    K: float,
    I_V: np.ndarray,
    I_corr: np.ndarray,
    rho: float,
    is_put: bool,
) -> Tuple[float, float]:
    """
    Compute a single inverse option price via conditional BS.

    Same logic as ``_mixed_estimator_prices_multi_strike`` but optimised
    for a single strike (avoids array overhead).

    Returns
    -------
    price : float
        MC-estimated inverse option price in BTC.
    std_err : float
        Standard error of the estimate.
    """
    n_paths = len(I_V)
    rho_sq = rho * rho
    one_minus_rho_sq = 1.0 - rho_sq
    log_K_over_S0 = np.log(K / S0)
    K_over_S0 = K / S0

    payoff_sum = 0.0
    payoff_sq_sum = 0.0

    for m in prange(n_paths):
        iv = I_V[m]
        ic = I_corr[m]

        log_F_S0 = -0.5 * rho_sq * iv + rho * ic
        F_S0 = np.exp(log_F_S0)
        sc = np.sqrt(max(one_minus_rho_sq * iv, 1e-15))

        if sc < 1e-10:
            if is_put:
                cond_price = max(K_over_S0 - F_S0, 0.0)
            else:
                cond_price = max(F_S0 - K_over_S0, 0.0)
        else:
            d1 = (log_F_S0 - log_K_over_S0 + 0.5 * sc * sc) / sc
            d2 = d1 - sc

            if is_put:
                cond_price = K_over_S0 * _norm_cdf_scalar(-d2) - F_S0 * _norm_cdf_scalar(-d1)
            else:
                cond_price = F_S0 * _norm_cdf_scalar(d1) - K_over_S0 * _norm_cdf_scalar(d2)

        cond_price = max(cond_price, 0.0)
        payoff_sum += cond_price
        payoff_sq_sum += cond_price * cond_price

    mean_price = payoff_sum / n_paths
    var_price = payoff_sq_sum / n_paths - mean_price * mean_price
    std_err = np.sqrt(max(var_price, 0.0) / n_paths)

    return mean_price, std_err


# =============================================================================
# HIGH-LEVEL INTERFACE
# =============================================================================

class MixedEstimator:
    """
    Mixed Estimator (Turbocharging) for rBergomi inverse option pricing.

    Conditions on the simulated variance paths to replace the spot-path
    Monte Carlo with an exact Black-Scholes evaluation per path.  This
    eliminates spot-discretisation noise entirely and typically reduces
    the MC standard error by an order of magnitude for the same number
    of variance paths.

    .. note::
        Requires ``scheme='hybrid'`` in the pricer, because the Mixed
        Estimator needs the standard normal increments that drive W^⊥.
        The Cholesky scheme does not expose these increments on the fine
        grid.

    References
    ----------
    McCrickerd, R. & Pakkanen, M. (2018). "Turbocharging Monte Carlo
    pricing for the rough Bergomi model", *Quantitative Finance*, 18(11).
    """

    def __init__(self):
        # Trigger Numba JIT compilation on first call
        self._compiled = False

    def _warmup(self):
        """Trigger Numba compilation with tiny arrays."""
        if self._compiled:
            return
        dummy_var = np.ones((2, 3), dtype=np.float64)
        dummy_dw = np.zeros((2, 2), dtype=np.float64)
        _compute_integrated_quantities(dummy_var, dummy_dw, 0.01)
        _mixed_estimator_single_strike(
            100.0, 100.0,
            np.ones(2), np.zeros(2),
            -0.5, True,
        )
        self._compiled = True
        logger.debug("MixedEstimator: Numba JIT warmup complete.")

    def price_inverse_options(
        self,
        S0: float,
        strikes: np.ndarray,
        var_paths: np.ndarray,
        dW_driving: np.ndarray,
        dt: float,
        rho: float,
        option_type: str = 'put',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Price inverse options for multiple strikes via the Mixed Estimator.

        Parameters
        ----------
        S0 : float
            Current BTC spot price (USD).
        strikes : np.ndarray, shape (n_strikes,)
            Strike prices (USD).
        var_paths : np.ndarray, shape (n_paths, n_steps)
            Simulated variance paths from the Hybrid Scheme.
        dW_driving : np.ndarray, shape (n_paths, n_steps - 1)
            Standard normal increments of the Brownian motion W^⊥ that
            drives the variance process (returned by ``HybridScheme``).
        dt : float
            Time step Δt = T / (n_steps − 1).
        rho : float
            Spot-variance correlation.
        option_type : str
            ``'put'`` or ``'call'``.

        Returns
        -------
        prices : np.ndarray, shape (n_strikes,)
            Inverse option prices in BTC.
        std_errs : np.ndarray, shape (n_strikes,)
            Standard errors of the price estimates.
        """
        self._warmup()

        is_put = option_type.lower() in ['put', 'p']
        strikes_arr = np.ascontiguousarray(strikes, dtype=np.float64)

        # Step 1: compute integrated quantities across all paths
        I_V, I_corr = _compute_integrated_quantities(var_paths, dW_driving, dt)

        # Step 2: conditional BS evaluation for each path × strike
        prices, std_errs = _mixed_estimator_prices_multi_strike(
            S0, strikes_arr, I_V, I_corr, rho, is_put,
        )

        return prices, std_errs

    def price_single(
        self,
        S0: float,
        K: float,
        var_paths: np.ndarray,
        dW_driving: np.ndarray,
        dt: float,
        rho: float,
        option_type: str = 'put',
    ) -> Tuple[float, float]:
        """
        Price a single inverse option via the Mixed Estimator.

        Returns
        -------
        price : float
            Inverse option price in BTC.
        std_err : float
            Standard error of the estimate.
        """
        self._warmup()

        is_put = option_type.lower() in ['put', 'p']

        I_V, I_corr = _compute_integrated_quantities(var_paths, dW_driving, dt)

        price, std_err = _mixed_estimator_single_strike(
            S0, K, I_V, I_corr, rho, is_put,
        )

        return price, std_err
