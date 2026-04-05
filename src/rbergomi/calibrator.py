# src/rbergomi/calibrator.py
"""
rBergomi Calibration Engine
============================

Calibrates rBergomi model parameters to match observed implied volatility
surface from Deribit BTC options.

Calibration pipeline:
    1. Estimate forward variance curve xi_0(T) from delta-neutral ATM IVs
    2. Optimize (H, eta, rho) jointly
    3. Loss function uses maturity-aware weighting (configurable)


Usage:
    >>> from src.rbergomi import Calibrator
    >>>
    >>> cal = Calibrator(n_paths=10_000, n_steps=50)
    >>> result = cal.calibrate(market_data)
    >>> print(result)
"""

import numpy as np
import polars as pl
from scipy.optimize import differential_evolution, minimize
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Any, Union, Callable
import logging
import time
import json
import warnings

from .pricer import rBergomiPricer
from .utils import (
    implied_vol_from_price_inverse,
    implied_vol_batch,
    validate_params,
    PerformanceTracker,
    format_european,
    format_percentage,
    bs_d1,
)

logger = logging.getLogger("rBergomi.calibrator")


# =============================================================================
# PARAMETER BOUNDS
# =============================================================================

# Default parameter bounds for calibration (xi removed — estimated from data)
#
# Revision notes:
#   eta: upper bound raised 2.0 → 4.0 because several dates showed eta at the
#        old upper boundary, suggesting the true optimum was being clipped.
#        Lower bound lowered 0.5 → 0.3 to allow gentle vol-of-vol regimes.
#   rho: upper bound relaxed -0.10 → 0.0 to accommodate dates where the
#        spot-vol correlation is near zero (e.g. post-crash mean-reversion).
#        Lower bound kept at -0.99 (theoretical limit, BTC leverage effect).
DEFAULT_BOUNDS = {
    "H":   (0.01, 0.49),     # Hurst: (0, 0.5), excluding 0.5 (standard BM)
    "eta": (0.3,  4.0),      # Vol-of-vol: widened [0.3, 4.0] — BTC shows high eta
    "rho": (-0.99, 0.0),     # Correlation: full negative range, up to zero
}

# Default initial guess (xi removed — estimated from data)
DEFAULT_INITIAL_GUESS = {
    "H":   0.07,
    "eta": 1.5,
    "rho": -0.7,
}

# RMSE thresholds for quality flagging (in implied volatility percentage points)
# Dates exceeding POOR_FIT_THRESHOLD_PP are logged as warnings and flagged in
# the output CSV with quality_flag = "poor". These typically correspond to
# market stress events (exchange collapses, halvings) where the IV surface is
# noisy or the model is structurally unable to fit the observed skew.
GOOD_FIT_THRESHOLD_PP  = 10.0   # RMSE < 10 pp  → "good"
ACCEPTABLE_FIT_THRESHOLD_PP = 20.0  # RMSE < 20 pp  → "acceptable"
POOR_FIT_THRESHOLD_PP  = 30.0   # RMSE ≥ 30 pp  → "poor"


# =============================================================================
# FORWARD VARIANCE CURVE
# =============================================================================

class ForwardVarianceCurve:
    """
    Piecewise-constant forward variance curve xi_0(T).

    Callable object: xi0(t) returns the forward variance at time t
    using piecewise-constant interpolation from ATM IV estimates.

    Parameters
    ----------
    maturities : np.ndarray
        Sorted maturity points where ATM IV was observed
    xi_values : np.ndarray
        xi_0 = sigma_ATM(T)^2 at each maturity
    """

    def __init__(self, maturities: np.ndarray, xi_values: np.ndarray):
        order = np.argsort(maturities)
        self.maturities = np.asarray(maturities, dtype=np.float64)[order]
        self.xi_values = np.asarray(xi_values, dtype=np.float64)[order]

    def __call__(self, t: float) -> float:
        """Evaluate xi_0(t) via piecewise-constant interpolation."""
        idx = np.searchsorted(self.maturities, t, side='right') - 1
        idx = max(0, min(idx, len(self.xi_values) - 1))
        return float(self.xi_values[idx])

    @property
    def mean_xi(self) -> float:
        """Mean forward variance across maturities."""
        return float(np.mean(self.xi_values))

    def to_dict(self) -> dict:
        """Serialize for storage."""
        return {
            "maturities": self.maturities.tolist(),
            "xi_values": self.xi_values.tolist(),
        }

    def __repr__(self) -> str:
        n = len(self.maturities)
        return (
            f"ForwardVarianceCurve({n} points, "
            f"T=[{self.maturities[0]:.3f}, {self.maturities[-1]:.3f}], "
            f"mean_xi={self.mean_xi:.4f})"
        )


# =============================================================================
# CALIBRATION RESULT
# =============================================================================

class CalibrationResult:
    """
    Container for calibration output.

    Attributes
    ----------
    H : float
        Hurst parameter (calibrated or fixed)
    eta : float
        Calibrated vol-of-vol
    rho : float
        Calibrated spot-vol correlation
    xi : float
        Representative forward variance (mean of xi_0 curve)
    rmse : float
        Root mean squared error (in IV percentage points)
    mae : float
        Mean absolute error (pp)
    n_points : int
        Number of market points used
    n_evals : int
        Number of loss function evaluations
    elapsed_seconds : float
        Total calibration time
    date : datetime
        Calibration date
    spot : float
        Spot price at calibration
    success : bool
        Whether optimization converged
    method : str
        Optimization method used
    details : dict
        Additional details (residuals, xi0_curve, optimizer output)
    """

    def __init__(
        self,
        H: float, eta: float, rho: float, xi: float,
        rmse: float, mae: float,
        n_points: int, n_evals: int, elapsed_seconds: float,
        date: Optional[datetime] = None,
        spot: Optional[float] = None,
        success: bool = True,
        method: str = "",
        details: Optional[Dict] = None
    ):
        self.H = H
        self.eta = eta
        self.rho = rho
        self.xi = xi
        self.rmse = rmse
        self.mae = mae
        self.n_points = n_points
        self.n_evals = n_evals
        self.elapsed_seconds = elapsed_seconds
        self.date = date
        self.spot = spot
        self.success = success
        self.method = method
        self.details = details or {}

    @property
    def params(self) -> Dict[str, float]:
        """Return parameters as dict."""
        return {"H": self.H, "eta": self.eta, "rho": self.rho, "xi": self.xi}

    @property
    def params_array(self) -> np.ndarray:
        """Return parameters as array [H, eta, rho, xi]."""
        return np.array([self.H, self.eta, self.rho, self.xi])

    @property
    def sigma_atm(self) -> float:
        """ATM implied volatility from xi."""
        return np.sqrt(self.xi)

    @property
    def quality_flag(self) -> str:
        """
        Categorical fit-quality label based on final RMSE.

        Returns
        -------
        str
            'good'       if RMSE < GOOD_FIT_THRESHOLD_PP  (< 10 pp)
            'acceptable' if RMSE < ACCEPTABLE_FIT_THRESHOLD_PP (< 20 pp)
            'borderline' if RMSE < POOR_FIT_THRESHOLD_PP  (< 30 pp)
            'poor'       if RMSE >= POOR_FIT_THRESHOLD_PP (≥ 30 pp)
                         Likely a stress event or structurally unfit surface.
        """
        if np.isnan(self.rmse):
            return "unknown"
        if self.rmse < GOOD_FIT_THRESHOLD_PP:
            return "good"
        if self.rmse < ACCEPTABLE_FIT_THRESHOLD_PP:
            return "acceptable"
        if self.rmse < POOR_FIT_THRESHOLD_PP:
            return "borderline"
        return "poor"

    @property
    def is_poor_fit(self) -> bool:
        """True when RMSE >= POOR_FIT_THRESHOLD_PP (≥ 30 pp)."""
        return self.rmse >= POOR_FIT_THRESHOLD_PP

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for DataFrame creation."""
        return {
            "date": self.date,
            "spot": self.spot,
            "H": self.H,
            "eta": self.eta,
            "rho": self.rho,
            "xi": self.xi,
            "sigma_atm": self.sigma_atm,
            "rmse_pp": self.rmse,
            "mae_pp": self.mae,
            "n_points": self.n_points,
            "n_evals": self.n_evals,
            "time_seconds": self.elapsed_seconds,
            "success": self.success,
            "method": self.method,
            "quality_flag": self.quality_flag,
        }

    def __repr__(self) -> str:
        date_str = self.date.strftime("%Y-%m-%d") if self.date else "N/A"
        flag_str = f"  [!] POOR FIT — quality_flag='{self.quality_flag}'\n" if self.is_poor_fit else ""
        return (
            f"CalibrationResult({date_str})\n"
            f"  H     = {self.H:.4f}\n"
            f"  eta   = {self.eta:.4f}\n"
            f"  rho   = {self.rho:.4f}\n"
            f"  xi    = {self.xi:.4f}  (sigma_ATM = {self.sigma_atm*100:.1f}%)\n"
            f"  RMSE  = {self.rmse:.2f} pp  [{self.quality_flag}]\n"
            f"  MAE   = {self.mae:.2f} pp\n"
            f"  N     = {self.n_points} points, {self.n_evals} evaluations\n"
            f"  Time  = {self.elapsed_seconds:.1f}s\n"
            f"  OK    = {self.success}\n"
            f"{flag_str}"
        )


# =============================================================================
# CALIBRATOR
# =============================================================================

class Calibrator:
    """
    Calibration engine for the rBergomi model.

    Pipeline:
        1. Estimate forward variance curve xi_0(T) from delta-neutral ATM IVs
        2. Optimize (H, eta, rho) jointly
        3. Loss uses configurable maturity-aware weighting

    Parameters
    ----------
    n_paths : int
        MC paths for pricing during calibration (fewer = faster but noisier)
    n_paths_coarse : int
        MC paths for DE global search (default 2,000)
    n_steps : int
        Time steps for MC simulation
    seed : int
        Random seed for reproducibility
    loss_type : str
        'rmse' for equal-weighted, 'wrmse' for volume-weighted
    maturity_weighting : str or callable
        Maturity weighting scheme for the loss function.
        'inverse_sqrt': w(T) = 1/sqrt(T) (default, emphasizes short maturities)
        'exponential':  w(T) = exp(-T)
        'uniform':      w(T) = 1 (no maturity weighting)
        callable:       custom function T -> weight
    scheme : str
        fBm simulation scheme passed to rBergomiPricer.
        'cholesky' (default): coarse-grid Cholesky + interpolation
        'hybrid': full Hybrid Scheme (Bennedsen et al. 2017)
    kappa : int
        Truncation parameter for the hybrid scheme (default: 6)
    moneyness_range : tuple or None
        (min_moneyness, max_moneyness) filter applied to the market data
        inside calibrate() before fitting. Default (0.70, 1.35) removes
        deep OTM/ITM options with wide bid-ask spreads that degrade the fit.
        Set to None to disable (uses all available strikes).

    Example
    -------
    >>> from iv_surface_builder import IVSurfaceBuilder
    >>> from src.rbergomi import Calibrator
    >>>
    >>> builder = IVSurfaceBuilder("data/option")
    >>> builder.build_index()
    >>> surface = builder.get_iv_surface(datetime(2024, 3, 28, 12), window_hours=6)
    >>> market_data = builder.export_for_rbergomi(surface)
    >>>
    >>> cal = Calibrator(n_paths=10_000, n_steps=50)
    >>> result = cal.calibrate(market_data)
    >>> print(result)

    Notes
    -----
    Performance guidelines:
    - n_paths=5,000:  ~3 min per calibration  (rough, for exploration)
    - n_paths=10,000: ~8 min per calibration  (default, good balance)
    - n_paths=20,000: ~20 min per calibration (accurate, for final results)
    """

    def __init__(
        self,
        n_paths: int = 50_000,
        n_paths_coarse: int = 10_000,
        n_steps: int = 100,
        seed: int = 42,
        loss_type: str = 'rmse',
        maturity_weighting: Union[str, Callable] = 'inverse_sqrt',
        scheme: str = 'hybrid',
        kappa: int = 6,
        popsize: int = 15,
        moneyness_range: Optional[Tuple[float, float]] = (0.70, 1.35),
        antithetic: bool = False,
        pricing_method: str = 'euler',
    ):
        self.n_paths = n_paths
        self.n_paths_coarse = n_paths_coarse
        self.n_steps = n_steps
        self.seed = seed
        self.loss_type = loss_type
        self.maturity_weighting = maturity_weighting
        self.scheme = scheme
        self.kappa = kappa
        self.popsize = popsize
        self.moneyness_range = moneyness_range
        self.antithetic = antithetic
        self.pricing_method = pricing_method

        # Tracking
        self._eval_count = 0
        self._best_loss = np.inf
        self._best_params = None
        self._loss_history = []
        self._start_time = None

        # Pre-computed forward variance curve (set per calibration)
        self._xi0_curve = None

        # Output directory
        self.output_dir = Path("Results/calibration")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)

        logger.info(
            f"Calibrator initialized: n_paths={n_paths:,} (coarse={n_paths_coarse:,}), "
            f"n_steps={n_steps}, loss={loss_type}, "
            f"maturity_weighting={maturity_weighting}, scheme={scheme}, seed={seed}, "
            f"popsize={popsize}, moneyness_range={moneyness_range}, "
            f"antithetic={antithetic}, pricing_method={pricing_method}"
        )

    # =========================================================================
    # FORWARD VARIANCE ESTIMATION (Problem 1 + Problem 3)
    # =========================================================================

    @staticmethod
    def _find_atm_iv(
        K_group: np.ndarray,
        iv_group: np.ndarray,
        S0: float,
        T: float
    ) -> float:
        """
        Find ATM implied volatility for a maturity slice using delta-neutral
        identification.

        Strategy:
            1. Compute BS d1 for each option using its market IV.
               50-delta corresponds to d1 = 0.
            2. Prefer the option closest to 50-delta (|d1| < 0.5).
            3. Fallback: interpolate IV between nearest strikes straddling
               the forward price.
            4. Last resort: closest strike to forward.

        Parameters
        ----------
        K_group : np.ndarray
            Strikes for this maturity
        iv_group : np.ndarray
            Market IVs for this maturity
        S0 : float
            Spot price
        T : float
            Time to maturity

        Returns
        -------
        float
            ATM implied volatility estimate
        """
        if len(K_group) == 0:
            return np.nan

        # Filter valid options (positive IV)
        valid = (iv_group > 0) & ~np.isnan(iv_group)
        K_valid = K_group[valid]
        iv_valid = iv_group[valid]

        if len(K_valid) == 0:
            return np.nan

        # Strategy 1: closest to 50-delta via d1 ~ 0
        d1_values = np.array([
            bs_d1(S0, K, T, 0.0, sigma)
            for K, sigma in zip(K_valid, iv_valid)
        ])

        abs_d1 = np.abs(d1_values)
        best_idx = np.argmin(abs_d1)

        if abs_d1[best_idx] < 0.5:
            return float(iv_valid[best_idx])

        # Strategy 2: interpolate between strikes straddling forward
        F = S0  # Forward ~ spot for crypto (r ~ 0)
        above_mask = K_valid >= F
        below_mask = K_valid < F

        if above_mask.any() and below_mask.any():
            K_above = K_valid[above_mask]
            iv_above = iv_valid[above_mask]
            K_below = K_valid[below_mask]
            iv_below = iv_valid[below_mask]

            idx_above = np.argmin(K_above - F)
            idx_below = np.argmax(K_below)

            K1, iv1 = K_below[idx_below], iv_below[idx_below]
            K2, iv2 = K_above[idx_above], iv_above[idx_above]

            if K2 > K1:
                w = (F - K1) / (K2 - K1)
                return float(iv1 + w * (iv2 - iv1))

        # Strategy 3: closest strike to forward
        closest_idx = np.argmin(np.abs(K_valid - S0))
        return float(iv_valid[closest_idx])

    def _estimate_forward_variance_curve(
        self, market_data: Dict[str, Any]
    ) -> Tuple[ForwardVarianceCurve, Dict[float, float]]:
        """
        Estimate xi_0(T) from ATM market IVs per maturity.

        For each unique maturity in the data, identifies the ATM option
        (using delta-neutral selection) and computes xi_0(T) = sigma_ATM(T)^2.

        Returns
        -------
        Tuple[ForwardVarianceCurve, dict]
            (xi0_curve, {T: sigma_ATM(T)} mapping)
        """
        S0 = market_data["S"]
        K_arr = market_data["K"]
        tau_arr = market_data["tau"]
        iv_market = market_data["iv_market"]
        unique_taus = np.unique(tau_arr)

        maturities = []
        xi_values = []
        atm_ivs = {}

        for T in sorted(unique_taus):
            mask = tau_arr == T
            atm_iv = self._find_atm_iv(K_arr[mask], iv_market[mask], S0, T)

            if not np.isnan(atm_iv) and atm_iv > 0:
                maturities.append(T)
                xi_values.append(atm_iv ** 2)
                atm_ivs[float(T)] = float(atm_iv)

        if not maturities:
            # Fallback: overall closest-to-forward strike
            moneyness = market_data.get("moneyness", K_arr / S0)
            atm_idx = np.argmin(np.abs(moneyness - 1.0))
            atm_iv = float(iv_market[atm_idx])
            logger.warning(
                "Delta-neutral ATM failed for all maturities; "
                "falling back to closest moneyness."
            )
            return ForwardVarianceCurve(
                np.array([float(tau_arr.min())]),
                np.array([atm_iv ** 2])
            ), {float(tau_arr.min()): atm_iv}

        return ForwardVarianceCurve(
            np.array(maturities), np.array(xi_values)
        ), atm_ivs

    # =========================================================================
    # MATURITY WEIGHTING (Problem 4)
    # =========================================================================

    def _compute_maturity_weights(self, tau_arr: np.ndarray) -> np.ndarray:
        """
        Compute maturity-aware weights for the loss function.

        Parameters
        ----------
        tau_arr : np.ndarray
            Maturities for each data point

        Returns
        -------
        np.ndarray
            Normalized weights (sum to 1)
        """
        if self.maturity_weighting == 'inverse_sqrt':
            w = 1.0 / np.sqrt(np.maximum(tau_arr, 1e-6))
        elif self.maturity_weighting == 'exponential':
            w = np.exp(-tau_arr)
        elif self.maturity_weighting == 'uniform':
            w = np.ones_like(tau_arr)
        elif self.maturity_weighting == 'balanced':
            w = np.zeros_like(tau_arr, dtype=np.float64)
            unique_taus = np.unique(tau_arr)
            n_mats = len(unique_taus)
            for T in unique_taus:
                mask = tau_arr == T
                n_in_group = mask.sum()
                w[mask] = 1.0 / (n_mats * n_in_group)
            return w
        elif callable(self.maturity_weighting):
            w = np.array([self.maturity_weighting(t) for t in tau_arr])
        else:
            w = np.ones_like(tau_arr)

        w_sum = w.sum()
        if w_sum > 0:
            return w / w_sum
        return np.ones_like(tau_arr) / len(tau_arr)

    # =========================================================================
    # LOSS FUNCTION
    # =========================================================================

    def _loss_function(
        self,
        params: np.ndarray,
        market_data: Dict[str, Any],
        verbose: bool = False,
        n_paths_override: Optional[int] = None
    ) -> float:
        """
        Compute maturity-weighted loss between model and market IVs.

        Parameters
        ----------
        params : np.ndarray
            [H, eta, rho]. Forward variance xi is taken from self._xi0_curve.
        market_data : dict
            Output of IVSurfaceBuilder.export_for_rbergomi()
        verbose : bool
            Print progress during optimization
        n_paths_override : int, optional
            If set, overrides self.n_paths for this evaluation.

        Returns
        -------
        float
            Loss value (maturity-weighted RMSE in percentage points)
        """
        H, eta, rho = params

        # Forward variance from pre-computed curve (Problem 1)
        xi0 = self._xi0_curve

        # Validate parameters
        xi_check = xi0(0.1) if xi0 is not None else 0.04
        is_valid, _ = validate_params(H, eta, rho, xi_check)
        if not is_valid:
            return 1000.0  # Penalty for invalid parameters

        self._eval_count += 1
        n_paths = n_paths_override if n_paths_override is not None else self.n_paths

        try:
            # Create pricer with current parameters and xi_0 curve
            pricer = rBergomiPricer(
                H=H, eta=eta, rho=rho, xi=xi0,
                n_paths=n_paths,
                n_steps=self.n_steps,
                scheme=self.scheme,
                kappa=self.kappa,
                seed=self.seed,
                antithetic=self.antithetic,
                pricing_method=self.pricing_method,
            )

            S0 = market_data["S"]
            K_arr = market_data["K"]
            tau_arr = market_data["tau"]
            iv_market = market_data["iv_market"]
            weights = market_data.get("weights", np.ones(len(K_arr)) / len(K_arr))
            n_points = market_data["n_points"]

            # Compute model prices for all market points
            # Batch by maturity: generate paths ONCE per unique T, price all strikes
            model_prices = np.full(n_points, np.nan)
            unique_taus = np.unique(tau_arr)

            for T in unique_taus:
                mask = tau_arr == T
                K_group = K_arr[mask]
                indices = np.where(mask)[0]

                try:
                    prices, _ = pricer.price_multiple_strikes(
                        S0, K_group, T, option_type='put'
                    )
                    model_prices[indices] = prices
                    logger.debug(f"  [batch] T={T:.4f}, {len(K_group)} strikes OK")
                except Exception as e:
                    logger.warning(f"  [batch FAILED] T={T:.4f} ({e})")
                    for j, idx in enumerate(indices):
                        try:
                            price, _ = pricer.price_inverse_put(S0, K_group[j], T)
                            model_prices[idx] = price
                        except Exception:
                            continue

            # One batch IV inversion for all points
            model_ivs = implied_vol_batch(
                model_prices, S0, K_arr, tau_arr,
                option_type='put', r=0.0
            )

            # Filter out failed inversions
            valid_mask = ~np.isnan(model_ivs)
            n_valid = valid_mask.sum()

            if n_valid < max(3, n_points * 0.3):
                # Too few valid points - penalize
                return 500.0

            iv_model_valid = model_ivs[valid_mask]
            iv_market_valid = iv_market[valid_mask]
            tau_valid = tau_arr[valid_mask]
            weights_valid = weights[valid_mask]

            # Compute error in percentage points
            errors_pp = (iv_model_valid - iv_market_valid) * 100

            # Maturity-aware weighting (Problem 4)
            mat_weights = self._compute_maturity_weights(tau_valid)

            if self.loss_type == 'wrmse':
                # Combine maturity and volume weights
                combined_w = mat_weights * weights_valid
                combined_w = combined_w / combined_w.sum()
                loss = np.sqrt(np.sum(combined_w * errors_pp**2))
            else:
                # Maturity-weighted RMSE
                loss = np.sqrt(np.sum(mat_weights * errors_pp**2))

            # Track progress
            self._loss_history.append(loss)

            if loss < self._best_loss:
                self._best_loss = loss
                self._best_params = params.copy()

            if verbose and self._eval_count % 10 == 0:
                elapsed = time.time() - self._start_time if self._start_time else 0
                logger.info(
                    f"  Eval {self._eval_count:>4d}: "
                    f"H={H:.3f}, eta={eta:.2f}, rho={rho:.2f} | "
                    f"RMSE={loss:.2f}pp | Best={self._best_loss:.2f}pp | "
                    f"Valid={n_valid}/{n_points} | {elapsed:.0f}s"
                )

            return loss

        except Exception as e:
            logger.debug(f"Loss eval failed: {e}")
            return 999.0

    # =========================================================================
    # SINGLE DATE CALIBRATION
    # =========================================================================

    def calibrate(
        self,
        market_data: Dict[str, Any],
        initial_guess: Optional[Dict[str, float]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        method: str = 'differential_evolution',
        maxiter: int = 30,
        tol: float = 0.5,
        polish: bool = True,
        verbose: bool = True,
        date: Optional[datetime] = None
    ) -> CalibrationResult:
        """
        Calibrate rBergomi parameters on a single IV surface.

        Pipeline:
            1. Estimate forward variance curve xi_0(T) from delta-neutral
               ATM IVs per maturity (xi is NOT optimized).
            2. Optimize (H, eta, rho) jointly.
            3. Loss uses maturity-aware weighting.

        Parameters
        ----------
        market_data : dict
            Output of IVSurfaceBuilder.export_for_rbergomi().
            Must contain: S, K, tau, iv_market, n_points
        initial_guess : dict, optional
            Starting point {H, eta, rho}. If None, estimates from data.
        bounds : dict, optional
            Parameter bounds {H: (lo, hi), eta: ..., rho: ...}
        method : str
            'differential_evolution' (global, recommended) or
            'nelder-mead' (local, needs good initial guess)
        maxiter : int
            Maximum iterations for optimizer
        tol : float
            Convergence tolerance (in RMSE pp)
        polish : bool
            If True, refine DE result with local optimizer
        verbose : bool
            Print progress
        date : datetime, optional
            Date label for this calibration

        Returns
        -------
        CalibrationResult
            Calibrated parameters and diagnostics
        """
        # Reset tracking
        self._eval_count = 0
        self._best_loss = np.inf
        self._best_params = None
        self._loss_history = []
        self._start_time = time.time()

        # Validate market data
        required_keys = ["S", "K", "tau", "iv_market", "n_points"]
        for key in required_keys:
            if key not in market_data:
                raise ValueError(f"market_data missing key: {key}")

        n_points = market_data["n_points"]
        S0 = market_data["S"]

        if n_points < 5:
            raise ValueError(f"Too few market points: {n_points} (need >= 5)")

        # Step 0.5: Filter by moneyness range if configured
        if self.moneyness_range is not None:
            m_lo, m_hi = self.moneyness_range
            moneyness = market_data.get("moneyness", market_data["K"] / S0)
            keep = (moneyness >= m_lo) & (moneyness <= m_hi)
            n_kept = keep.sum()
            if n_kept < 5:
                logger.warning(
                    f"Moneyness filter [{m_lo}, {m_hi}] keeps only {n_kept} "
                    f"points (need >=5). Skipping filter."
                )
            else:
                if n_kept < n_points:
                    logger.info(
                        f"Moneyness filter [{m_lo}, {m_hi}]: "
                        f"{n_points} -> {n_kept} points"
                    )
                market_data = {
                    "S": S0,
                    "K": market_data["K"][keep],
                    "tau": market_data["tau"][keep],
                    "iv_market": market_data["iv_market"][keep],
                    "moneyness": moneyness[keep],
                    "n_points": int(n_kept),
                    "weights": market_data.get(
                        "weights", np.ones(n_points) / n_points
                    )[keep],
                }
                # Re-normalize weights
                w_sum = market_data["weights"].sum()
                if w_sum > 0:
                    market_data["weights"] = market_data["weights"] / w_sum
                n_points = int(n_kept)

        # Step 1: Estimate forward variance curve from ATM IVs
        # Uses delta-neutral ATM identification (Problem 1 + Problem 3)
        self._xi0_curve, atm_ivs = self._estimate_forward_variance_curve(
            market_data
        )

        # Set bounds (no xi — it comes from the curve)
        bnds = DEFAULT_BOUNDS.copy()
        if bounds:
            bnds.update(bounds)

        # Set initial guess (no xi)
        x0_dict = DEFAULT_INITIAL_GUESS.copy()
        if initial_guess:
            x0_dict.update(initial_guess)

        # Build scipy bounds and x0
        scipy_bounds = [bnds["H"], bnds["eta"], bnds["rho"]]
        x0 = np.array([x0_dict["H"], x0_dict["eta"], x0_dict["rho"]])

        if verbose:
            logger.info("=" * 60)
            logger.info("rBergomi CALIBRATION")
            logger.info("=" * 60)
            logger.info(f"Date: {date.strftime('%Y-%m-%d') if date else 'N/A'}")
            logger.info(f"Spot: ${S0:,.0f}")
            logger.info(f"Market points: {n_points}")
            logger.info(f"xi0 curve: {self._xi0_curve}")
            logger.info(f"Maturity weighting: {self.maturity_weighting}")
            logger.info(f"Method: {method}")
            logger.info(f"MC paths: {self.n_paths:,}, steps: {self.n_steps}")
            logger.info("-" * 60)

        # Step 2: Run optimization over (H, eta, rho)
        if method == 'differential_evolution':
            result = self._run_differential_evolution(
                market_data, scipy_bounds, maxiter, tol, polish, verbose
            )
        elif method == 'nelder-mead':
            result = self._run_nelder_mead(
                market_data, x0, scipy_bounds, maxiter, tol, verbose
            )
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                "Use 'differential_evolution' or 'nelder-mead'"
            )

        elapsed = time.time() - self._start_time

        # Extract optimal parameters
        H_opt, eta_opt, rho_opt = result.x

        # Representative xi from forward variance curve
        xi_repr = self._xi0_curve.mean_xi

        # Compute final loss with more detail (uses xi0 curve internally)
        final_loss, residuals = self._compute_detailed_loss(
            H_opt, eta_opt, rho_opt, market_data
        )

        # Build result
        cal_result = CalibrationResult(
            H=H_opt, eta=eta_opt, rho=rho_opt, xi=xi_repr,
            rmse=final_loss["rmse"],
            mae=final_loss["mae"],
            n_points=n_points,
            n_evals=self._eval_count,
            elapsed_seconds=elapsed,
            date=date,
            spot=S0,
            success=result.success,
            method=method,
            details={
                "residuals": residuals,
                "loss_history": self._loss_history,
                "xi0_curve": self._xi0_curve.to_dict(),
                "atm_ivs": atm_ivs,
                "maturity_weighting": str(self.maturity_weighting),
                "optimizer_message": (
                    str(result.message) if hasattr(result, 'message') else ""
                ),
            }
        )

        if verbose:
            logger.info("=" * 60)
            logger.info("CALIBRATION COMPLETED")
            logger.info("=" * 60)
            logger.info(str(cal_result))

        # E — quality flag warning: poor fits are logged at WARNING level so
        # they surface even when verbose=False (e.g. in time-series loops).
        if cal_result.is_poor_fit:
            date_str = date.strftime("%Y-%m-%d") if date else "N/A"
            logger.warning(
                f"[POOR FIT] {date_str}: RMSE={cal_result.rmse:.1f} pp "
                f"(threshold={POOR_FIT_THRESHOLD_PP} pp). "
                f"Possible causes: stress event, illiquid surface, or model "
                f"mis-specification. quality_flag='{cal_result.quality_flag}'."
            )
        elif cal_result.rmse >= ACCEPTABLE_FIT_THRESHOLD_PP:
            date_str = date.strftime("%Y-%m-%d") if date else "N/A"
            logger.warning(
                f"[BORDERLINE FIT] {date_str}: RMSE={cal_result.rmse:.1f} pp "
                f"(quality_flag='{cal_result.quality_flag}')."
            )

        return cal_result

    def calibrate_per_maturity(
        self,
        market_data: Dict[str, Any],
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        method: str = 'differential_evolution',
        maxiter: int = 50,
        tol: float = 0.5,
        polish: bool = True,
        verbose: bool = True,
        date: Optional[datetime] = None
    ) -> List['CalibrationResult']:
        """
        Calibrate (H, eta, rho) independently for each maturity slice.

        For each unique T, optimizes separately using only the strikes at
        that maturity. The xi_0 value is the ATM IV^2 at that T (scalar).

        Parameters
        ----------
        market_data : dict
            Output of IVSurfaceBuilder.export_for_rbergomi()
        bounds : dict, optional
            Parameter bounds
        method : str
            Optimizer method
        maxiter : int
            Max iterations per maturity slice
        tol : float
            Convergence tolerance
        polish : bool
            Refine DE result with local optimizer
        verbose : bool
            Print progress
        date : datetime, optional
            Date label

        Returns
        -------
        list of CalibrationResult
            One result per maturity slice, sorted by maturity.
            Each result.details contains 'maturity_days'.
        """
        S0 = market_data["S"]
        K_arr = market_data["K"]
        tau_arr = market_data["tau"]
        iv_market = market_data["iv_market"]
        moneyness = market_data.get("moneyness", K_arr / S0)

        # Apply moneyness filter if configured
        if self.moneyness_range is not None:
            m_lo, m_hi = self.moneyness_range
            keep = (moneyness >= m_lo) & (moneyness <= m_hi)
            if keep.sum() >= 5:
                K_arr, tau_arr = K_arr[keep], tau_arr[keep]
                iv_market, moneyness = iv_market[keep], moneyness[keep]

        unique_taus = np.sort(np.unique(tau_arr))

        if verbose:
            logger.info("=" * 60)
            logger.info("PER-MATURITY CALIBRATION")
            logger.info("=" * 60)
            logger.info(f"Date: {date.strftime('%Y-%m-%d') if date else 'N/A'}")
            logger.info(f"Spot: ${S0:,.0f}")
            logger.info(f"Maturities: {len(unique_taus)} slices")
            for T in unique_taus:
                logger.info(f"  T={T*365:.0f}d: {(tau_arr == T).sum()} strikes")
            logger.info("-" * 60)

        bnds = DEFAULT_BOUNDS.copy()
        if bounds:
            bnds.update(bounds)
        scipy_bounds = [bnds["H"], bnds["eta"], bnds["rho"]]

        results = []

        for T in unique_taus:
            mask = tau_arr == T
            K_slice = K_arr[mask]
            iv_slice = iv_market[mask]
            m_slice = moneyness[mask]
            n_slice = int(mask.sum())
            T_days = T * 365

            if n_slice < 3:
                logger.warning(f"  T={T_days:.0f}d: only {n_slice} points, skipping")
                continue

            atm_iv = self._find_atm_iv(K_slice, iv_slice, S0, T)
            if np.isnan(atm_iv) or atm_iv <= 0:
                closest = np.argmin(np.abs(m_slice - 1.0))
                atm_iv = float(iv_slice[closest])
            xi_0 = atm_iv ** 2

            slice_data = {
                "S": S0,
                "K": K_slice,
                "tau": np.full(n_slice, T),
                "iv_market": iv_slice,
                "moneyness": m_slice,
                "n_points": n_slice,
                "weights": np.ones(n_slice) / n_slice,
            }

            self._eval_count = 0
            self._best_loss = np.inf
            self._best_params = None
            self._loss_history = []
            self._start_time = time.time()
            self._xi0_curve = ForwardVarianceCurve(
                np.array([T]), np.array([xi_0])
            )

            if verbose:
                logger.info(f"\n  Calibrating T={T_days:.0f}d ({n_slice} strikes, "
                            f"sigma_ATM={atm_iv*100:.1f}%)...")

            if method == 'differential_evolution':
                opt_result = self._run_differential_evolution(
                    slice_data, scipy_bounds, maxiter, tol, polish, False
                )
            else:
                x0 = np.array([
                    DEFAULT_INITIAL_GUESS["H"],
                    DEFAULT_INITIAL_GUESS["eta"],
                    DEFAULT_INITIAL_GUESS["rho"],
                ])
                opt_result = self._run_nelder_mead(
                    slice_data, x0, scipy_bounds, maxiter, tol, False
                )

            elapsed = time.time() - self._start_time
            H_opt, eta_opt, rho_opt = opt_result.x

            final_loss, residuals = self._compute_detailed_loss(
                H_opt, eta_opt, rho_opt, slice_data
            )

            cal_result = CalibrationResult(
                H=H_opt, eta=eta_opt, rho=rho_opt, xi=xi_0,
                rmse=final_loss["rmse"],
                mae=final_loss["mae"],
                n_points=n_slice,
                n_evals=self._eval_count,
                elapsed_seconds=elapsed,
                date=date,
                spot=S0,
                success=opt_result.success,
                method=method,
                details={
                    "residuals": residuals,
                    "loss_history": self._loss_history,
                    "xi0_curve": self._xi0_curve.to_dict(),
                    "maturity_days": float(T_days),
                    "sigma_atm": float(atm_iv),
                }
            )
            results.append(cal_result)

            if verbose:
                logger.info(
                    f"    H={H_opt:.4f}, eta={eta_opt:.2f}, rho={rho_opt:.2f} | "
                    f"RMSE={final_loss['rmse']:.2f}pp | {elapsed:.0f}s"
                )

        if verbose and results:
            logger.info("\n" + "=" * 60)
            logger.info("PER-MATURITY RESULTS")
            logger.info("=" * 60)
            for r in results:
                T_d = r.details.get('maturity_days', 0)
                logger.info(
                    f"  T={T_d:>5.0f}d: H={r.H:.4f}, eta={r.eta:.2f}, "
                    f"rho={r.rho:.2f}, RMSE={r.rmse:.2f}pp ({r.n_points} pts)"
                )
            total_pts = sum(r.n_points for r in results)
            avg_rmse = np.sqrt(
                sum(r.rmse**2 * r.n_points for r in results) / total_pts
            )
            logger.info(f"  Average RMSE (weighted): {avg_rmse:.2f}pp")
            logger.info("=" * 60)

        return results

    def _run_differential_evolution(
        self,
        market_data: Dict,
        bounds: List[Tuple],
        maxiter: int,
        tol: float,
        polish: bool,
        verbose: bool
    ):
        """Run Differential Evolution global optimizer with coarse MC paths.

        Uses self.n_paths_coarse (default 2000) instead of self.n_paths during
        DE exploration to reduce ~5x the per-evaluation cost. The Nelder-Mead
        refinement that follows uses the full self.n_paths for final precision.
        """
        result = differential_evolution(
            self._loss_function,
            bounds=bounds,
            args=(market_data, verbose, self.n_paths_coarse),
            strategy='best1bin',
            maxiter=maxiter,
            tol=tol,
            popsize=self.popsize,
            mutation=(0.5, 1.5),
            recombination=0.8,
            seed=self.seed,
            polish=polish,
            disp=False,
            workers=1,          # Numba prange already uses all cores inside each eval
        )
        return result

    def _run_nelder_mead(
        self,
        market_data: Dict,
        x0: np.ndarray,
        bounds: List[Tuple],
        maxiter: int,
        tol: float,
        verbose: bool
    ):
        """Run Nelder-Mead local optimizer."""
        # Convert bounds to Nelder-Mead format (via penalty)
        bounds_arr = np.array(bounds)

        def bounded_loss(params):
            # Check bounds
            for i, (lo, hi) in enumerate(bounds_arr):
                if params[i] < lo or params[i] > hi:
                    return 1000.0
            return self._loss_function(params, market_data, verbose)

        result = minimize(
            bounded_loss,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': maxiter * 4 * len(x0),
                'xatol': 0.001,
                'fatol': tol,
                'adaptive': True,
                'disp': False
            }
        )
        return result

    def _compute_detailed_loss(
        self,
        H: float,
        eta: float,
        rho: float,
        market_data: Dict
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Compute detailed loss metrics and per-point residuals.

        Uses self._xi0_curve for forward variance.

        Returns
        -------
        Tuple[dict, dict]
            (metrics, residuals)
        """
        pricer = rBergomiPricer(
            H=H, eta=eta, rho=rho, xi=self._xi0_curve,
            n_paths=self.n_paths,  # Same paths as calibration for consistency
            n_steps=self.n_steps,
            scheme=self.scheme,
            kappa=self.kappa,
            seed=self.seed,
            antithetic=self.antithetic,
        )

        S0 = market_data["S"]
        K_arr = market_data["K"]
        tau_arr = market_data["tau"]
        iv_market = market_data["iv_market"]
        moneyness = market_data.get("moneyness", K_arr / S0)
        weights = market_data.get("weights", np.ones(len(K_arr)) / len(K_arr))

        # Batch price per maturity, then one batch IV call
        model_prices = np.full(len(K_arr), np.nan)
        unique_taus = np.unique(tau_arr)

        for T in unique_taus:
            mask = tau_arr == T
            K_group = K_arr[mask]
            indices = np.where(mask)[0]

            try:
                prices, _ = pricer.price_multiple_strikes(
                    S0, K_group, T, option_type='put'
                )
                model_prices[indices] = prices
            except Exception:
                for j, idx in enumerate(indices):
                    try:
                        price, _ = pricer.price_inverse_put(S0, K_group[j], T)
                        model_prices[idx] = price
                    except Exception:
                        continue

        model_ivs = implied_vol_batch(
            model_prices, S0, K_arr, tau_arr,
            option_type='put', r=0.0
        )

        valid = ~np.isnan(model_ivs)
        errors_pp = np.where(valid, (model_ivs - iv_market) * 100, np.nan)

        valid_errors = errors_pp[valid]

        metrics = {
            "rmse": float(np.sqrt(np.mean(valid_errors**2))) if len(valid_errors) > 0 else np.nan,
            "mae": float(np.mean(np.abs(valid_errors))) if len(valid_errors) > 0 else np.nan,
            "bias": float(np.mean(valid_errors)) if len(valid_errors) > 0 else np.nan,
            "max_error": float(np.max(np.abs(valid_errors))) if len(valid_errors) > 0 else np.nan,
            "n_valid": int(valid.sum()),
        }

        residuals = {
            "K": K_arr,
            "tau": tau_arr,
            "moneyness": moneyness,
            "iv_market": iv_market,
            "iv_model": model_ivs,
            "error_pp": errors_pp,
        }

        return metrics, residuals

    # =========================================================================
    # TIME SERIES CALIBRATION
    # =========================================================================

    def calibrate_timeseries(
        self,
        builder,
        dates: List[datetime],
        method: str = 'differential_evolution',
        maxiter: int = 20,
        tol: float = 1.0,
        window_hours: float = 6.0,
        min_points: int = 15,
        moneyness_range: Tuple[float, float] = (0.8, 1.2),
        min_ttm_days: int = 7,
        max_ttm_days: int = 90,
        save_intermediate: bool = True,
        resume: bool = True
    ) -> pl.DataFrame:
        """
        Calibrate rBergomi on multiple dates.

        Calibrates (H, eta, rho) jointly.

        Parameters
        ----------
        builder : IVSurfaceBuilder
            Initialized surface builder with index
        dates : list of datetime
            Dates to calibrate
        method : str
            Optimizer method
        maxiter : int
            Max iterations per date
        tol : float
            Convergence tolerance
        window_hours : float
            Time window for IV extraction
        min_points : int
            Minimum market points for valid calibration
        moneyness_range : tuple
            (min, max) moneyness filter
        min_ttm_days, max_ttm_days : int
            TTM filter
        save_intermediate : bool
            Save results after each date
        resume : bool
            If True and a partial CSV already exists, skip already-calibrated
            dates and append new results to the existing file.

        Returns
        -------
        pl.DataFrame
            Time series of calibrated parameters

        Example
        -------
        >>> dates = builder.get_calibration_dates(
        ...     datetime(2024, 1, 1), datetime(2024, 12, 31), frequency='weekly'
        ... )
        >>> results_df = calibrator.calibrate_timeseries(builder, dates)
        """
        scheme_tag = f"_{self.scheme}_{self.pricing_method}"
        output_file = self.output_dir / "tables" / f"calibration_timeseries{scheme_tag}.csv"
        parquet_file = self.output_dir / "tables" / f"calibration_timeseries{scheme_tag}.parquet"
        n_total = len(dates)

        # --- Resume: load previously completed dates ---
        df_existing = None
        done_dates: set = set()
        prev_result = None

        if resume and output_file.exists():
            try:
                df_existing = pl.read_csv(output_file)
                # Date column is stored as ISO string; compare on YYYY-MM-DD only
                done_dates = {
                    str(d)[:10]
                    for d in df_existing["date"].to_list()
                }
                if done_dates:
                    # Warm start from the last completed calibration
                    # Carry optimizer params (H, eta, rho); xi is re-estimated
                    last = df_existing[-1]
                    _H = float(last["H"][0])
                    _eta = float(last["eta"][0])
                    _rho = float(last["rho"][0])

                    class _WarmStart:
                        params = {"H": _H, "eta": _eta, "rho": _rho}

                    prev_result = _WarmStart()
                    logger.info(
                        f"Resume: {len(done_dates)} dates already done — skipping them."
                    )
            except Exception as exc:
                logger.warning(
                    f"Could not load existing results for resume ({exc}). "
                    "Starting fresh."
                )
                df_existing = None
                done_dates = set()

        # Keep only dates not yet calibrated
        remaining = [
            d for d in dates
            if d.strftime("%Y-%m-%d") not in done_dates
        ]
        n_done = n_total - len(remaining)
        n_remaining = len(remaining)

        logger.info("=" * 60)
        logger.info(f"TIME SERIES CALIBRATION: {n_total} dates total")
        if n_done:
            logger.info(f"  Already done: {n_done} | Remaining: {n_remaining}")
        logger.info(f"Method: {method}, maxiter: {maxiter}")
        logger.info("=" * 60)

        # New results from this run only
        new_results = []

        for i, date in enumerate(remaining):
            logger.info(
                f"\n[{n_done + i + 1}/{n_total}] "
                f"Calibrating {date.strftime('%Y-%m-%d')}..."
            )

            try:
                # Extract IV surface
                surface = builder.get_iv_surface(
                    target_time=date,
                    window_hours=window_hours,
                    min_volume=0.05,
                    moneyness_range=moneyness_range,
                    min_ttm_days=min_ttm_days,
                    max_ttm_days=max_ttm_days
                )

                if surface.height < min_points:
                    logger.warning(
                        f"  Skipping: only {surface.height} points "
                        f"(need {min_points})"
                    )
                    continue

                market_data = builder.export_for_rbergomi(surface)

                # Warm start from previous calibration (eta, rho only)
                init_guess = None
                if prev_result is not None:
                    init_guess = prev_result.params

                # Calibrate
                result = self.calibrate(
                    market_data=market_data,
                    initial_guess=init_guess,
                    method=method,
                    maxiter=maxiter,
                    tol=tol,
                    verbose=False,
                    date=date
                )

                new_results.append(result.to_dict())
                prev_result = result

                logger.info(
                    f"  Done: H={result.H:.3f}, eta={result.eta:.2f}, "
                    f"rho={result.rho:.2f}, xi={result.xi:.4f}, "
                    f"RMSE={result.rmse:.2f}pp ({result.elapsed_seconds:.0f}s)"
                )

                # Save intermediate: existing rows + all new rows so far
                if save_intermediate:
                    df_new = pl.DataFrame(new_results)
                    df_combined = (
                        pl.concat([df_existing, df_new], how="diagonal_relaxed")
                        if df_existing is not None
                        else df_new
                    )
                    df_combined.write_csv(output_file)

            except Exception as e:
                logger.error(f"  Error: {e}")
                continue

        # Build final DataFrame (existing + new)
        if new_results:
            df_new = pl.DataFrame(new_results)
            df_results = (
                pl.concat([df_existing, df_new], how="diagonal_relaxed")
                if df_existing is not None
                else df_new
            )
        elif df_existing is not None:
            df_results = df_existing
        else:
            logger.warning("No successful calibrations")
            return pl.DataFrame()

        # Final save
        df_results.write_csv(output_file)
        df_results.write_parquet(parquet_file)

        logger.info("=" * 60)
        logger.info(f"CALIBRATION COMPLETE: {len(df_results)}/{n_total} dates")
        logger.info(f"Results saved to: {output_file}")
        logger.info("=" * 60)

        # Print summary statistics
        self._print_timeseries_summary(df_results)

        return df_results

    def _print_timeseries_summary(self, df: pl.DataFrame):
        """Print summary statistics for time series calibration."""
        print("\n" + "=" * 60)
        print("CALIBRATION SUMMARY")
        print("=" * 60)

        for col in ["H", "eta", "rho", "xi", "rmse_pp"]:
            if col in df.columns:
                values = df[col].to_numpy()
                valid = values[~np.isnan(values)]
                if len(valid) > 0:
                    print(
                        f"  {col:<8}: mean={np.mean(valid):.4f}, "
                        f"std={np.std(valid):.4f}, "
                        f"range=[{np.min(valid):.4f}, {np.max(valid):.4f}]"
                    )

        print("=" * 60)

    # =========================================================================
    # SINGLE SNAPSHOT CALIBRATION (convenience)
    # =========================================================================

    def calibrate_snapshot(
        self,
        builder,
        target_date: datetime,
        method: str = 'differential_evolution',
        maxiter: int = 30,
        window_hours: float = 6.0,
        moneyness_range: Tuple[float, float] = (0.8, 1.2),
        min_ttm_days: int = 7,
        max_ttm_days: int = 90,
        save: bool = True
    ) -> CalibrationResult:
        """
        Convenience method: extract surface + calibrate in one call.

        Parameters
        ----------
        builder : IVSurfaceBuilder
            Initialized surface builder
        target_date : datetime
            Date to calibrate
        method : str
            Optimizer
        save : bool
            Save result to disk

        Returns
        -------
        CalibrationResult
        """
        # Extract surface
        surface = builder.get_iv_surface(
            target_time=target_date,
            window_hours=window_hours,
            min_volume=0.05,
            moneyness_range=moneyness_range,
            min_ttm_days=min_ttm_days,
            max_ttm_days=max_ttm_days
        )

        if surface.height < 10:
            raise ValueError(
                f"Insufficient data for {target_date}: {surface.height} points"
            )

        market_data = builder.export_for_rbergomi(surface)

        # Calibrate
        result = self.calibrate(
            market_data=market_data,
            method=method,
            maxiter=maxiter,
            date=target_date
        )

        # Save
        if save:
            self._save_snapshot_result(result, target_date)

        return result

    def _save_snapshot_result(self, result: CalibrationResult, date: datetime):
        """Save single calibration result to disk."""
        date_str = date.strftime("%Y%m%d")

        # Save parameters
        output = result.to_dict()
        scheme_tag = f"_{self.scheme}_{self.pricing_method}"
        output_file = self.output_dir / "tables" / f"calibration_{date_str}{scheme_tag}.csv"

        df = pl.DataFrame([output])
        df.write_csv(output_file)

        logger.info(f"Snapshot result saved to: {output_file}")

    # =========================================================================
    # EVENT CALIBRATION
    # =========================================================================

    def calibrate_events(
        self,
        builder,
        events: Optional[List[Dict]] = None,
        method: str = 'differential_evolution',
        maxiter: int = 25
    ) -> pl.DataFrame:
        """
        Calibrate on market event dates (pre/during/post).

        Parameters
        ----------
        builder : IVSurfaceBuilder
            Initialized surface builder
        events : list of dict, optional
            Event list. If None, uses MARKET_EVENTS from iv_surface_builder.
        method : str
            Optimizer method
        maxiter : int
            Max iterations per calibration

        Returns
        -------
        pl.DataFrame
            Calibration results for each event observation
        """
        if events is None:
            events = builder.get_event_dates()

        results = []

        logger.info(f"Event calibration: {len(events)} events")

        for event in events:
            event_date = event["date"]
            event_name = event["name"]

            logger.info(f"\nEvent: {event_name} ({event_date.strftime('%Y-%m-%d')})")

            # Calibrate pre, event, post
            from datetime import timedelta
            for label, offset in [("pre", -1), ("event", 0), ("post", 1)]:
                obs_date = event_date + timedelta(days=offset)

                try:
                    result = self.calibrate_snapshot(
                        builder=builder,
                        target_date=obs_date.replace(hour=12),
                        method=method,
                        maxiter=maxiter,
                        window_hours=8.0,
                        moneyness_range=(0.7, 1.4),
                        min_ttm_days=3,
                        max_ttm_days=90,
                        save=False
                    )

                    row = result.to_dict()
                    row["event_name"] = event_name
                    row["observation"] = label
                    results.append(row)

                    logger.info(
                        f"  {label}: H={result.H:.3f}, "
                        f"RMSE={result.rmse:.2f}pp"
                    )

                except Exception as e:
                    logger.warning(f"  {label}: failed - {e}")
                    continue

        if not results:
            return pl.DataFrame()

        df = pl.DataFrame(results)

        # Save
        scheme_tag = f"_{self.scheme}_{self.pricing_method}"
        output_file = self.output_dir / "tables" / f"calibration_events{scheme_tag}.csv"
        df.write_csv(output_file)
        logger.info(f"\nEvent results saved to: {output_file}")

        return df


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Calibrator - Module Test")
    print("=" * 60)

    # Quick test with synthetic data
    from .utils import bs_price_inverse_put

    # Generate synthetic market data (BS surface with known parameters)
    S0 = 50_000
    sigma_true = 0.60
    strikes = np.array([40_000, 45_000, 48_000, 50_000, 52_000, 55_000, 60_000])
    taus = np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])  # ~30 days

    iv_market = np.full(len(strikes), sigma_true)  # Flat smile for BS
    moneyness = strikes / S0
    weights = np.ones(len(strikes)) / len(strikes)

    market_data = {
        "S": S0,
        "K": strikes,
        "tau": taus,
        "iv_market": iv_market,
        "moneyness": moneyness,
        "weights": weights,
        "n_points": len(strikes)
    }

    print(f"\nSynthetic market: S0=${S0:,}, sigma={sigma_true*100:.0f}%")
    print(f"Points: {len(strikes)}")

    cal = Calibrator(n_paths=5_000, n_steps=30, seed=42)
    print("\nRunning quick calibration (Nelder-Mead)...")

    result = cal.calibrate(
        market_data,
        method='nelder-mead',
        maxiter=20,
        verbose=True
    )

    print(f"\n{result}")
