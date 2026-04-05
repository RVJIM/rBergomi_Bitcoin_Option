# src/rbergomi/utils.py
"""
Utility Functions for rBergomi Pricing
======================================

Contains:
- Black-Scholes pricing for inverse options
- Implied volatility inversion (Newton-Raphson)
- Performance timing decorators
- European number formatting

"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from functools import wraps
import time
import logging
from typing import Callable, Tuple, Optional

logger = logging.getLogger("rBergomi.utils")


# =============================================================================
# BLACK-SCHOLES FOR INVERSE OPTIONS
# =============================================================================

def bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate d1 for Black-Scholes formula.

    Parameters
    ----------
    S : float
        Spot price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate
    sigma : float
        Volatility (in decimals, e.g., 0.60 = 60%)

    Returns
    -------
    float
        d1 value
    """
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate d2 for Black-Scholes formula."""
    return bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_price_call_usd(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Standard Black-Scholes call price in USD.

    Parameters
    ----------
    S : float
        Spot price (USD)
    K : float
        Strike price (USD)
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate
    sigma : float
        Volatility (decimal)

    Returns
    -------
    float
        Call option price in USD
    """
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_price_put_usd(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Standard Black-Scholes put price in USD.

    Parameters
    ----------
    S : float
        Spot price (USD)
    K : float
        Strike price (USD)
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate
    sigma : float
        Volatility (decimal)

    Returns
    -------
    float
        Put option price in USD
    """
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_price_inverse_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes price for INVERSE call option (BTC denominated).

    Inverse call payoff: max(S_T - K, 0) / S_T = max(1 - K/S_T, 0)
    Price in BTC = (USD price) / S

    Parameters
    ----------
    S : float
        Spot price (USD)
    K : float
        Strike price (USD)
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate (typically 0 for crypto)
    sigma : float
        Volatility (decimal, e.g., 0.60 = 60%)

    Returns
    -------
    float
        Inverse call option price in BTC
    """
    usd_price = bs_price_call_usd(S, K, T, r, sigma)
    return usd_price / S


def bs_price_inverse_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes price for INVERSE put option (BTC denominated).

    Inverse put payoff: max(K - S_T, 0) / S_T = max(K/S_T - 1, 0)
    Price in BTC = (USD price) / S

    Parameters
    ----------
    S : float
        Spot price (USD)
    K : float
        Strike price (USD)
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate (typically 0 for crypto)
    sigma : float
        Volatility (decimal, e.g., 0.60 = 60%)

    Returns
    -------
    float
        Inverse put option price in BTC
    """
    usd_price = bs_price_put_usd(S, K, T, r, sigma)
    return usd_price / S


def bs_vega_inverse(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Vega for inverse option (same for call and put).

    Used for Newton-Raphson IV inversion.
    Vega_inverse = Vega_USD / S = S * sqrt(T) * phi(d1) / S = sqrt(T) * phi(d1)

    Returns
    -------
    float
        Vega in BTC per 1% change in volatility
    """
    d1 = bs_d1(S, K, T, r, sigma)
    # Vega_USD = S * sqrt(T) * phi(d1)
    # Vega_inverse = Vega_USD / S = sqrt(T) * phi(d1)
    return np.sqrt(T) * norm.pdf(d1)


# =============================================================================
# IMPLIED VOLATILITY INVERSION
# =============================================================================

def implied_vol_from_price_inverse(
    price_btc: float,
    S: float,
    K: float,
    T: float,
    option_type: str = 'put',
    r: float = 0.0,
    initial_guess: float = 0.5,
    tol: float = 1e-8,
    max_iter: int = 100
) -> float:
    """
    Calculate implied volatility from inverse option price using Brent's method.

    Uses Brent's method for robustness (handles edge cases better than Newton-Raphson).

    Parameters
    ----------
    price_btc : float
        Observed option price in BTC
    S : float
        Spot price (USD)
    K : float
        Strike price (USD)
    T : float
        Time to maturity (years)
    option_type : str
        'put' or 'call'
    r : float
        Risk-free rate (default 0 for crypto)
    initial_guess : float
        Starting volatility guess (not used with Brent, kept for API compatibility)
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations

    Returns
    -------
    float
        Implied volatility (decimal, e.g., 0.60 = 60%)
        Returns np.nan if inversion fails

    Example
    -------
    >>> iv = implied_vol_from_price_inverse(
    ...     price_btc=0.015,
    ...     S=50000,
    ...     K=48000,
    ...     T=0.25,
    ...     option_type='put'
    ... )
    >>> print(f"IV: {iv*100:.2f}%")
    """
    if price_btc <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan

    # Select pricing function
    if option_type.lower() in ['put', 'p']:
        price_func = bs_price_inverse_put
    else:
        price_func = bs_price_inverse_call

    # Objective function: model_price - market_price
    def objective(sigma):
        return price_func(S, K, T, r, sigma) - price_btc

    # Brent's method needs bracketing interval
    # IV typically between 1% and 500% for crypto
    sigma_low = 0.01
    sigma_high = 5.0

    try:
        # Check if solution exists in bracket
        f_low = objective(sigma_low)
        f_high = objective(sigma_high)

        if f_low * f_high > 0:
            # No sign change - solution outside bracket
            # Try to find valid bracket
            if f_low > 0:
                # Price too high even at low vol - no valid IV
                return np.nan
            else:
                # Price too low even at high vol - IV > 500%
                return np.nan

        iv = brentq(objective, sigma_low, sigma_high, xtol=tol, maxiter=max_iter)
        return iv

    except (ValueError, RuntimeError) as e:
        logger.debug(f"IV inversion failed: {e}")
        return np.nan


def implied_vol_newton(
    price_btc: float,
    S: float,
    K: float,
    T: float,
    option_type: str = 'put',
    r: float = 0.0,
    initial_guess: float = 0.5,
    tol: float = 1e-8,
    max_iter: int = 50
) -> float:
    """
    Calculate implied volatility using Newton-Raphson method.

    Faster than Brent but less robust. Use for batch processing
    when you have good initial guesses.

    Parameters
    ----------
    price_btc : float
        Observed option price in BTC
    S : float
        Spot price (USD)
    K : float
        Strike price (USD)
    T : float
        Time to maturity (years)
    option_type : str
        'put' or 'call'
    r : float
        Risk-free rate
    initial_guess : float
        Starting volatility
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations

    Returns
    -------
    float
        Implied volatility (decimal)
    """
    if price_btc <= 0 or T <= 0:
        return np.nan

    if option_type.lower() in ['put', 'p']:
        price_func = bs_price_inverse_put
    else:
        price_func = bs_price_inverse_call

    sigma = initial_guess

    for _ in range(max_iter):
        model_price = price_func(S, K, T, r, sigma)
        vega = bs_vega_inverse(S, K, T, r, sigma)

        if vega < 1e-12:
            break

        diff = model_price - price_btc

        if abs(diff) < tol:
            return sigma

        sigma = sigma - diff / vega

        # Keep sigma in reasonable bounds
        sigma = max(0.01, min(sigma, 5.0))

    return sigma


# =============================================================================
# BATCH IMPLIED VOLATILITY INVERSION
# =============================================================================

def implied_vol_batch(
    prices: np.ndarray,
    S: float,
    K_arr: np.ndarray,
    tau_arr: np.ndarray,
    option_type: str = 'put',
    r: float = 0.0,
    sigma_lo: float = 0.01,
    sigma_hi: float = 5.0,
    n_iter: int = 60
) -> np.ndarray:
    """
    Batch implied-volatility inversion via sequential scipy.brentq (CPU).

    Parameters
    ----------
    prices : np.ndarray
        Observed inverse-option prices in BTC, shape (N,)
    S : float
        Spot price (USD)
    K_arr : np.ndarray
        Strike prices, shape (N,)
    tau_arr : np.ndarray
        Times to maturity (years), shape (N,)
    option_type : str
        'put' or 'call'
    r : float
        Risk-free rate
    sigma_lo, sigma_hi : float
        Bisection bracket bounds
    n_iter : int
        Unused (kept for API compatibility)

    Returns
    -------
    np.ndarray
        Implied volatilities, shape (N,). np.nan for failed inversions.
    """
    prices = np.asarray(prices, dtype=np.float64)
    K_arr = np.asarray(K_arr, dtype=np.float64)
    tau_arr = np.asarray(tau_arr, dtype=np.float64)
    N = len(prices)

    if N == 0:
        return np.empty(0, dtype=np.float64)

    is_put = option_type.lower() in ('put', 'p')
    price_func = bs_price_inverse_put if is_put else bs_price_inverse_call
    out = np.full(N, np.nan, dtype=np.float64)

    for i in range(N):
        p, K, tau = prices[i], K_arr[i], tau_arr[i]
        if p <= 0 or tau <= 0 or K <= 0:
            continue
        try:
            def _obj(sigma, _p=p, _K=K, _tau=tau):
                return price_func(S, _K, _tau, r, sigma) - _p
            f_lo = _obj(sigma_lo)
            f_hi = _obj(sigma_hi)
            if f_lo * f_hi > 0:
                continue
            out[i] = brentq(_obj, sigma_lo, sigma_hi, xtol=1e-12, maxiter=100)
        except (ValueError, RuntimeError):
            continue

    return out


# =============================================================================
# PERFORMANCE UTILITIES
# =============================================================================

def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    Logs timing at DEBUG level and stores result in function attribute.

    Example
    -------
    >>> @timer
    ... def slow_function():
    ...     time.sleep(1)
    ...
    >>> slow_function()
    >>> print(f"Took {slow_function.last_time:.3f}s")
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        wrapper.last_time = elapsed
        wrapper.call_count = getattr(wrapper, 'call_count', 0) + 1
        wrapper.total_time = getattr(wrapper, 'total_time', 0.0) + elapsed

        logger.debug(f"{func.__name__}: {elapsed*1000:.2f} ms")
        return result

    wrapper.last_time = 0.0
    wrapper.call_count = 0
    wrapper.total_time = 0.0
    return wrapper


class PerformanceTracker:
    """
    Track performance metrics across multiple function calls.

    Example
    -------
    >>> tracker = PerformanceTracker()
    >>> with tracker.measure("path_generation"):
    ...     generate_paths()
    >>> print(tracker.summary())
    """

    def __init__(self):
        self.timings = {}
        self.counts = {}

    class _TimerContext:
        def __init__(self, tracker, name):
            self.tracker = tracker
            self.name = name
            self.start = None

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start
            if self.name not in self.tracker.timings:
                self.tracker.timings[self.name] = []
                self.tracker.counts[self.name] = 0
            self.tracker.timings[self.name].append(elapsed)
            self.tracker.counts[self.name] += 1

    def measure(self, name: str):
        """Context manager to measure a code block."""
        return self._TimerContext(self, name)

    def summary(self) -> str:
        """Return formatted summary of all timings."""
        lines = ["Performance Summary", "=" * 50]
        total = sum(sum(t) for t in self.timings.values())

        for name in sorted(self.timings.keys()):
            times = self.timings[name]
            count = self.counts[name]
            total_time = sum(times)
            avg_time = total_time / count if count > 0 else 0
            pct = (total_time / total * 100) if total > 0 else 0

            lines.append(
                f"{name:<30} {total_time*1000:>8.2f} ms "
                f"({count:>4} calls, {avg_time*1000:>6.2f} ms/call) "
                f"[{pct:>5.1f}%]"
            )

        lines.append("=" * 50)
        lines.append(f"{'TOTAL':<30} {total*1000:>8.2f} ms")
        return "\n".join(lines)

    def reset(self):
        """Clear all timings."""
        self.timings.clear()
        self.counts.clear()


# =============================================================================
# FORMATTING UTILITIES
# =============================================================================

def format_european(value: float, decimals: int = 2) -> str:
    """
    Format number with European convention (comma as decimal separator).

    Parameters
    ----------
    value : float
        Number to format
    decimals : int
        Decimal places

    Returns
    -------
    str
        Formatted string with comma decimal separator

    Example
    -------
    >>> format_european(12345.67, 2)
    '12.345,67'
    """
    # Format with specified decimals
    formatted = f"{value:,.{decimals}f}"
    # Swap . and , for European format
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def format_price_btc(price: float, decimals: int = 6) -> str:
    """Format BTC price with appropriate precision."""
    return f"{price:.{decimals}f} BTC"


def format_price_usd(price: float) -> str:
    """Format USD price with thousands separator."""
    return f"${price:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format as percentage."""
    return f"{value * 100:.{decimals}f}%"


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_params(H: float, eta: float, rho: float, xi: float) -> Tuple[bool, str]:
    """
    Validate rBergomi model parameters.

    Parameters
    ----------
    H : float
        Hurst exponent (should be in (0, 0.5])
    eta : float
        Vol-of-vol (should be positive)
    rho : float
        Spot-vol correlation (should be in [-1, 1])
    xi : float
        Forward variance (should be positive)

    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    errors = []

    if not (0 < H <= 0.5):
        errors.append(f"H={H} must be in (0, 0.5]. H=0.5 is standard Brownian motion.")

    if eta <= 0:
        errors.append(f"eta={eta} must be positive")

    if not (-1 <= rho <= 1):
        errors.append(f"rho={rho} must be in [-1, 1]")

    if xi <= 0:
        errors.append(f"xi={xi} must be positive")

    if errors:
        return False, "; ".join(errors)

    return True, ""


def validate_option_params(S: float, K: float, T: float) -> Tuple[bool, str]:
    """
    Validate option parameters.

    Returns
    -------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    errors = []

    if S <= 0:
        errors.append(f"S={S} must be positive")

    if K <= 0:
        errors.append(f"K={K} must be positive")

    if T <= 0:
        errors.append(f"T={T} must be positive")

    if errors:
        return False, "; ".join(errors)

    return True, ""
