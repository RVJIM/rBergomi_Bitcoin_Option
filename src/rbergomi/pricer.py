# src/rbergomi/pricer.py
"""
rBergomi Monte Carlo Pricer
============================

Implementation of the rough Bergomi model for pricing Bitcoin inverse options.

Two fBm simulation schemes are available (selected via `scheme` parameter):
    - 'cholesky' (default): Coarse-grid Cholesky with linear interpolation to
      a fine grid. Complexity O(n_c^3 + n_f * n_c). Fast and simple.
    - 'hybrid':  Full Hybrid Scheme of Bennedsen, Lunde & Pakkanen (2017),
      with kernel decomposition into distant (D) and recent (R) components
      controlled by truncation parameter kappa. Complexity O(n * kappa).

Model:
    dS_t / S_t = sqrt(V_t) * dW_t
    V_t = xi(t) * exp(eta * Z^H_t - 0.5 * eta^2 * t^(2H))

Where Z^H is fractional Brownian motion with Hurst parameter H.

Performance Optimization Levels:
    1. NumPy vectorized (baseline)
    2. Numba JIT compiled (default, ~10-50x speedup)
    3. Numba parallel (for large n_paths, additional ~4x on 4 cores)


References:
    - Bayer, Friz, Gatheral (2016): "Pricing under rough volatility"
    - Bennedsen, Lunde, Pakkanen (2017): "Hybrid Scheme for Brownian
      Semistationary Processes", Finance and Stochastics, 21
    - McCrickerd & Pakkanen (2018): "Turbocharging Monte Carlo pricing
      for the rough Bergomi model"
"""

import numpy as np
from scipy.linalg import cholesky
from typing import Tuple, Optional, Callable, Union
import logging
import warnings


# Numba imports with fallback
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. Using NumPy fallback (slower).")
    # Dummy decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args or callable(args[0]) else decorator(args[0])
    prange = range

from .utils import (
    validate_params,
    validate_option_params,
    implied_vol_batch,
    PerformanceTracker
)
from .mixed_estimator import MixedEstimator

logger = logging.getLogger("rBergomi.pricer")


# =============================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS
# =============================================================================

@njit(cache=True, fastmath=True)
def _fbm_covariance_matrix(times: np.ndarray, H: float) -> np.ndarray:
    """
    Compute covariance matrix of fractional Brownian motion.

    Cov(Z^H_s, Z^H_t) = 0.5 * (s^{2H} + t^{2H} - |t-s|^{2H})

    Parameters
    ----------
    times : np.ndarray
        Time grid points (must start from 0)
    H : float
        Hurst parameter

    Returns
    -------
    np.ndarray
        Covariance matrix (n x n)
    """
    n = len(times)
    cov = np.empty((n, n), dtype=np.float64)
    two_H = 2.0 * H

    for i in range(n):
        t_i = times[i]
        t_i_2H = t_i ** two_H

        for j in range(i + 1):
            t_j = times[j]
            t_j_2H = t_j ** two_H
            diff_2H = np.abs(t_i - t_j) ** two_H

            cov_ij = 0.5 * (t_i_2H + t_j_2H - diff_2H)
            cov[i, j] = cov_ij
            cov[j, i] = cov_ij

    return cov


def _joint_fbm_bm_covariance(times: np.ndarray, H: float) -> np.ndarray:
    """
    Compute joint covariance matrix of (Z^H, W^1) on the same time grid.

    Uses the Riemann-Liouville (one-sided) representation consistently:
        Z^H_t = sqrt(2H) * integral_0^t (t-s)^{H-1/2} dW^1_s

    The resulting 2n x 2n matrix has block structure:
        [Cov(Z^H, Z^H)   Cov(Z^H, W^1)]
        [Cov(W^1, Z^H)   Cov(W^1, W^1)]

    where (all derived from the RL kernel):
        Cov(Z^H_ti, Z^H_tj) = 2H * integral_0^{min(ti,tj)} (ti-u)^{H-1/2} (tj-u)^{H-1/2} du
        Cov(W^1_ti, W^1_tj)  = min(ti, tj)
        Cov(Z^H_ti, W^1_tj)  = sqrt(2H)/(H+0.5) * (ti^{H+0.5} - max(ti-tj,0)^{H+0.5})

    The ZZ block uses Gauss-Jacobi quadrature (not the standard fBm formula
    0.5*(t^{2H}+s^{2H}-|t-s|^{2H})), because the RL fBm has different
    off-diagonal covariances from the standard (two-sided) fBm. Using the
    standard formula would make the joint matrix inconsistent and non-PSD.

    Parameters
    ----------
    times : np.ndarray
        Time grid (n points, starting from 0)
    H : float
        Hurst parameter

    Returns
    -------
    np.ndarray
        Joint covariance matrix, shape (2n, 2n)
    """
    from scipy.special import roots_jacobi

    n = len(times)
    cov = np.zeros((2 * n, 2 * n), dtype=np.float64)
    two_H = 2.0 * H
    a_plus = H + 0.5
    a_minus = H - 0.5
    scale_zw = np.sqrt(two_H) / a_plus

    # Gauss-Jacobi quadrature for ZZ block.
    # After substitution w = min(ti,tj) - u, the RL covariance integral becomes:
    #   t_min^{2H} * integral_0^1 (s+w)^{H-1/2} w^{H-1/2} dw
    # where s = (t_max - t_min)/t_min.
    # Weight function w^{H-1/2} is handled by Gauss-Jacobi.
    n_quad = 50
    x_gj, w_gj = roots_jacobi(n_quad, 0.0, a_minus)
    v_nodes = (1.0 + x_gj) / 2.0   # nodes on [0, 1]
    v_weights = w_gj / (2.0 ** (a_minus + 1))  # weights for [0, 1]

    # Block (0,0): ZZ — RL fBm covariance
    for i in range(n):
        t_i = times[i]
        if t_i == 0.0:
            continue
        # Diagonal: Var(Z^H_t) = t^{2H} (exact)
        cov[i, i] = t_i ** two_H
        # Off-diagonal via Gauss-Jacobi quadrature
        for j in range(i):
            t_j = times[j]
            if t_j == 0.0:
                continue
            if t_i >= t_j:
                s = (t_i - t_j) / t_j
                t_min_2H = t_j ** two_H
            else:
                s = (t_j - t_i) / t_i
                t_min_2H = t_i ** two_H
            fvals = (s + v_nodes) ** a_minus
            integral = t_min_2H * np.dot(v_weights, fvals)
            c_zz = two_H * integral
            cov[i, j] = c_zz
            cov[j, i] = c_zz

    # Block (1,1): WW — standard BM covariance
    for i in range(n):
        for j in range(i + 1):
            c_ww = min(times[i], times[j])
            cov[n + i, n + j] = c_ww
            cov[n + j, n + i] = c_ww

    # Block (0,1) and (1,0): ZW — RL cross-covariance (closed form)
    for i in range(n):
        t_i = times[i]
        for j in range(n):
            t_j = times[j]
            diff = t_i - t_j
            if diff > 0.0:
                c_zw = scale_zw * (t_i ** a_plus - diff ** a_plus)
            else:
                c_zw = scale_zw * t_i ** a_plus
            cov[i, n + j] = c_zw
            cov[n + j, i] = c_zw

    return cov


@njit(cache=True, fastmath=True, parallel=True)
def _generate_fbm_paths_numba(
    n_paths: int,
    n_steps: int,
    chol_lower: np.ndarray,
    random_normals: np.ndarray
) -> np.ndarray:
    """
    Generate fBm paths using pre-computed Cholesky factor.

    Parameters
    ----------
    n_paths : int
        Number of Monte Carlo paths
    n_steps : int
        Number of time steps (including t=0)
    chol_lower : np.ndarray
        Lower Cholesky factor of covariance matrix (n_steps x n_steps)
    random_normals : np.ndarray
        Pre-generated standard normals (n_paths x n_steps)

    Returns
    -------
    np.ndarray
        fBm paths (n_paths x n_steps), with paths[i, 0] = 0
    """
    paths = np.empty((n_paths, n_steps), dtype=np.float64)

    for i in prange(n_paths):
        # Z^H = L @ z where z ~ N(0, I)
        for j in range(n_steps):
            total = 0.0
            for k in range(j + 1):
                total += chol_lower[j, k] * random_normals[i, k]
            paths[i, j] = total

    return paths


@njit(cache=True, fastmath=True, parallel=True)
def _generate_variance_paths_numba(
    fbm_paths: np.ndarray,
    times: np.ndarray,
    xi: float,
    eta: float,
    H: float
) -> np.ndarray:
    """
    Generate variance paths from fBm paths.

    V_t = xi * exp(eta * Z^H_t - 0.5 * eta^2 * t^{2H})

    Parameters
    ----------
    fbm_paths : np.ndarray
        Fractional Brownian motion paths (n_paths x n_steps)
    times : np.ndarray
        Time grid (n_steps,)
    xi : float
        Forward variance level
    eta : float
        Vol-of-vol parameter
    H : float
        Hurst parameter

    Returns
    -------
    np.ndarray
        Variance paths (n_paths x n_steps)
    """
    n_paths, n_steps = fbm_paths.shape
    var_paths = np.empty((n_paths, n_steps), dtype=np.float64)

    two_H = 2.0 * H
    eta_sq_half = 0.5 * eta * eta

    for i in prange(n_paths):
        for j in range(n_steps):
            t = times[j]
            # Avoid 0^{2H} issues at t=0
            if t > 1e-10:
                correction = eta_sq_half * (t ** two_H)
            else:
                correction = 0.0

            exponent = eta * fbm_paths[i, j] - correction
            var_paths[i, j] = xi * np.exp(exponent)

    return var_paths


@njit(cache=True, fastmath=True, parallel=True)
def _generate_variance_paths_curve_numba(
    fbm_paths: np.ndarray,
    times: np.ndarray,
    xi_arr: np.ndarray,
    eta: float,
    H: float
) -> np.ndarray:
    """
    Generate variance paths with time-dependent forward variance.

    V_t = xi_arr[j] * exp(eta * Z^H_t - 0.5 * eta^2 * t^{2H})

    Parameters
    ----------
    fbm_paths : np.ndarray
        Fractional Brownian motion paths (n_paths x n_steps)
    times : np.ndarray
        Time grid (n_steps,)
    xi_arr : np.ndarray
        Forward variance at each time point (n_steps,)
    eta : float
        Vol-of-vol parameter
    H : float
        Hurst parameter

    Returns
    -------
    np.ndarray
        Variance paths (n_paths x n_steps)
    """
    n_paths, n_steps = fbm_paths.shape
    var_paths = np.empty((n_paths, n_steps), dtype=np.float64)

    two_H = 2.0 * H
    eta_sq_half = 0.5 * eta * eta

    for i in prange(n_paths):
        for j in range(n_steps):
            t = times[j]
            if t > 1e-10:
                correction = eta_sq_half * (t ** two_H)
            else:
                correction = 0.0

            exponent = eta * fbm_paths[i, j] - correction
            var_paths[i, j] = xi_arr[j] * np.exp(exponent)

    return var_paths


@njit(cache=True, fastmath=True, parallel=True)
def _generate_spot_paths_simple(
    S0: float,
    var_paths: np.ndarray,
    dt: float,
    dW: np.ndarray
) -> np.ndarray:
    """
    Generate spot price paths using log-Euler scheme.

    dS_t / S_t = sqrt(V_t) * dW_t

    Parameters
    ----------
    S0 : float
        Initial spot price
    var_paths : np.ndarray
        Variance paths (n_paths x n_steps)
    dt : float
        Time step size
    dW : np.ndarray
        Brownian increments (n_paths x n_steps-1), already correlated

    Returns
    -------
    np.ndarray
        Spot price paths (n_paths x n_steps)
    """
    n_paths, n_steps = var_paths.shape
    spot_paths = np.empty((n_paths, n_steps), dtype=np.float64)

    sqrt_dt = np.sqrt(dt)

    for i in prange(n_paths):
        spot_paths[i, 0] = S0

        for j in range(n_steps - 1):
            V_t = var_paths[i, j]
            sqrt_V = np.sqrt(max(V_t, 1e-10))

            # Log-Euler: S_{t+dt} = S_t * exp(-0.5*V*dt + sqrt(V)*sqrt(dt)*dW)
            log_return = -0.5 * V_t * dt + sqrt_V * sqrt_dt * dW[i, j]
            spot_paths[i, j + 1] = spot_paths[i, j] * np.exp(log_return)

    return spot_paths


@njit(cache=True, fastmath=True, parallel=True)
def _compute_inverse_put_payoffs(spot_terminal: np.ndarray, K: float) -> np.ndarray:
    """
    Compute inverse put option payoffs.

    Payoff = max(K/S_T - 1, 0) in BTC

    Parameters
    ----------
    spot_terminal : np.ndarray
        Terminal spot prices (n_paths,)
    K : float
        Strike price

    Returns
    -------
    np.ndarray
        Payoffs in BTC (n_paths,)
    """
    n = len(spot_terminal)
    payoffs = np.empty(n, dtype=np.float64)

    for i in prange(n):
        S_T = spot_terminal[i]
        if S_T > 1e-10:  # Avoid division by zero
            payoff = K / S_T - 1.0
            payoffs[i] = max(payoff, 0.0)
        else:
            # S_T ~ 0 means massive payoff (but clip for numerical stability)
            payoffs[i] = K / 1e-10 - 1.0

    return payoffs


@njit(cache=True, fastmath=True, parallel=True)
def _compute_inverse_call_payoffs(spot_terminal: np.ndarray, K: float) -> np.ndarray:
    """
    Compute inverse call option payoffs.

    Payoff = max(1 - K/S_T, 0) in BTC

    Parameters
    ----------
    spot_terminal : np.ndarray
        Terminal spot prices (n_paths,)
    K : float
        Strike price

    Returns
    -------
    np.ndarray
        Payoffs in BTC (n_paths,)
    """
    n = len(spot_terminal)
    payoffs = np.empty(n, dtype=np.float64)

    for i in prange(n):
        S_T = spot_terminal[i]
        if S_T > 1e-10:
            payoff = 1.0 - K / S_T
            payoffs[i] = max(payoff, 0.0)
        else:
            payoffs[i] = 0.0  # K/S_T >> 1 means payoff = 0

    return payoffs


# =============================================================================
# HYBRID SCHEME IMPLEMENTATION
# =============================================================================

class CholeskyScheme:
    """
    Coarse-grid joint Cholesky scheme for fBm + BM simulation.

    Jointly simulates the fractional BM Z^H and its driving standard BM W^1
    on a coarse grid via Cholesky decomposition of the 2n x 2n joint
    covariance matrix, then linearly interpolates both to the fine grid.

    The joint simulation ensures that the correlation structure between
    Z^H and W^1 is exactly preserved, which is needed for correct spot
    path construction in the rBergomi model.

    Complexity: O((2*n_coarse)^3 + n_fine * n_coarse), much cheaper than
    a full Cholesky on the fine grid O((2*n_fine)^3).

    Parameters
    ----------
    H : float
        Hurst parameter (0 < H <= 0.5)
    n_coarse : int
        Number of coarse grid points (default: 50)
    """

    def __init__(self, H: float, n_coarse: int = 50):
        self.H = H
        self.n_coarse = n_coarse

        self._T = None
        self._times_coarse = None
        self._times_fine = None
        self._chol_joint = None  # Cholesky of 2n x 2n joint covariance
        self._fine_to_coarse_left = None
        self._fine_to_coarse_right = None
        self._interp_weights = None

    def initialize(self, T: float, n_fine: int):
        """
        Initialize the scheme for a specific maturity and fine grid.

        Builds the joint covariance matrix of (Z^H, W^1) on the grid
        and computes its Cholesky decomposition. If n_fine > n_coarse,
        n_coarse is automatically raised to n_fine to avoid BM
        interpolation artefacts.

        Parameters
        ----------
        T : float
            Time to maturity
        n_fine : int
            Total number of fine grid points (including t=0)
        """
        self._T = T
        # Ensure coarse grid is at least as dense as fine grid
        if n_fine > self.n_coarse:
            self.n_coarse = n_fine
        self._times_coarse = np.linspace(0, T, self.n_coarse)
        self._times_fine = np.linspace(0, T, n_fine)

        # Joint covariance matrix of (Z^H, W^1) on coarse grid
        cov_joint = _joint_fbm_bm_covariance(self._times_coarse, self.H)

        # Regularization for numerical stability
        cov_joint += np.eye(2 * self.n_coarse) * 1e-10

        # Cholesky decomposition of the 2n x 2n joint matrix
        self._chol_joint = cholesky(cov_joint, lower=True, overwrite_a=False)

        # Pre-compute interpolation weights
        self._precompute_interpolation_weights()

        logger.debug(
            f"CholeskyScheme initialized: T={T}, n_coarse={self.n_coarse}, "
            f"n_fine={n_fine}, joint_dim={2 * self.n_coarse}"
        )

    def _precompute_interpolation_weights(self):
        """Pre-compute linear interpolation weights from coarse to fine grid."""
        n_fine = len(self._times_fine)
        n_coarse = self.n_coarse

        self._fine_to_coarse_left = np.empty(n_fine, dtype=np.int32)
        self._fine_to_coarse_right = np.empty(n_fine, dtype=np.int32)
        self._interp_weights = np.empty(n_fine, dtype=np.float64)

        for i, t in enumerate(self._times_fine):
            idx = np.searchsorted(self._times_coarse, t)

            if idx == 0:
                self._fine_to_coarse_left[i] = 0
                self._fine_to_coarse_right[i] = 0
                self._interp_weights[i] = 0.0
            elif idx >= n_coarse:
                self._fine_to_coarse_left[i] = n_coarse - 1
                self._fine_to_coarse_right[i] = n_coarse - 1
                self._interp_weights[i] = 0.0
            else:
                self._fine_to_coarse_left[i] = idx - 1
                self._fine_to_coarse_right[i] = idx
                t_left = self._times_coarse[idx - 1]
                t_right = self._times_coarse[idx]
                self._interp_weights[i] = (t - t_left) / (t_right - t_left)

    def _interpolate_to_fine(self, values_coarse: np.ndarray) -> np.ndarray:
        """Linearly interpolate (n_paths, n_coarse) to (n_paths, n_fine)."""
        n_fine = len(self._times_fine)
        n_paths = values_coarse.shape[0]
        values_fine = np.empty((n_paths, n_fine), dtype=np.float64)

        for i in range(n_fine):
            li = self._fine_to_coarse_left[i]
            ri = self._fine_to_coarse_right[i]
            w = self._interp_weights[i]

            if li == ri:
                values_fine[:, i] = values_coarse[:, li]
            else:
                values_fine[:, i] = (1 - w) * values_coarse[:, li] + w * values_coarse[:, ri]

        return values_fine

    def generate_paths(
        self,
        n_paths: int,
        random_state: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate jointly-simulated fBm and BM paths on the fine grid.

        Simulates (Z^H, W^1) on the coarse grid via the joint Cholesky
        factor, then interpolates both to the fine grid. Returns the fBm
        paths and the BM *increments* (normalised) on the fine grid.

        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths
        random_state : np.random.Generator, optional
            Random number generator

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            fbm_fine : shape (n_paths, n_fine) — fBm Z^H on fine grid
            dW1_fine : shape (n_paths, n_fine - 1) — normalised BM increments
                       on the fine grid, i.e. (W^1_{t+dt} - W^1_t) / sqrt(dt)
        """
        if self._chol_joint is None:
            raise RuntimeError("CholeskyScheme not initialized. Call initialize() first.")

        if random_state is None:
            random_state = np.random.default_rng()

        nc = self.n_coarse
        dim = 2 * nc

        # Generate i.i.d. normals and apply joint Cholesky
        Z = random_state.standard_normal((n_paths, dim))
        joint_coarse = _generate_fbm_paths_numba(n_paths, dim, self._chol_joint, Z)

        # Split into fBm and BM on coarse grid
        fbm_coarse = joint_coarse[:, :nc]   # Z^H at coarse times
        bm_coarse = joint_coarse[:, nc:]     # W^1 at coarse times

        # Interpolate both to fine grid
        fbm_fine = self._interpolate_to_fine(fbm_coarse)
        bm_fine = self._interpolate_to_fine(bm_coarse)

        # Convert BM levels to normalised increments: dW = (W_{t+dt} - W_t) / sqrt(dt)
        dt_fine = self._T / (len(self._times_fine) - 1)
        sqrt_dt = np.sqrt(dt_fine)
        dW1_fine = (bm_fine[:, 1:] - bm_fine[:, :-1]) / sqrt_dt

        return fbm_fine, dW1_fine


# =============================================================================
# TRUE HYBRID SCHEME (Bennedsen, Lunde & Pakkanen 2017)
# =============================================================================

@njit(cache=True, fastmath=True, parallel=True)
def _hybrid_scheme_cumulative(
    dW: np.ndarray,
    n_times: int,
    delta: float,
    H: float,
    kappa: int
) -> np.ndarray:
    """
    Generate fBm paths via Volterra convolution with Hybrid kernel split.

    Computes standard fBm at each grid point using a convolution:

        Z^H_{t_i} = sum_{l=1}^{i} c_l * epsilon_{i-l}

    where the convolution weights c_l are split into:
        - Recent (l <= kappa): exact variance-matched weights
          c_l = sqrt((l*d)^{2H} - ((l-1)*d)^{2H})
          These capture the singularity of the Volterra kernel near s=0 exactly.
        - Distant (l > kappa): integral-averaged Volterra weights (scaled)
          c_l = sqrt(2H*d) * b_l, where
          b_l = [(l*d)^{H+1/2} - ((l-1)*d)^{H+1/2}] / [(H+1/2)*d]
          These are accurate for large lag where the kernel is smooth.

    The recent weights give exact marginal variance Var(Z^H_{t_i}) = t_i^{2H}
    via a telescoping sum.  For l >= 2 the two weight families nearly coincide;
    the split matters mainly at l=1 where the kernel singularity is strongest.

    Parameters
    ----------
    dW : np.ndarray
        Standard normal increments, shape (n_paths, n_steps)
    n_times : int
        Number of time points (n_steps + 1)
    delta : float
        Time step size
    H : float
        Hurst parameter
    kappa : int
        Truncation parameter controlling recent/distant split.
        Larger kappa = better accuracy near the singularity.

    Returns
    -------
    np.ndarray
        fBm paths, shape (n_paths, n_times), with paths[:, 0] = 0
    """
    n_paths = dW.shape[0]
    n_steps = n_times - 1
    two_H = 2.0 * H
    a_plus = H + 0.5  # exponent for integral-averaged weights
    sqrt_2H_delta = np.sqrt(two_H * delta)

    # Pre-compute ALL weights up to n_steps
    weights = np.empty(n_steps, dtype=np.float64)
    max_kappa = min(kappa, n_steps)

    # Recent weights (l=1..kappa): variance-matched telescoping
    # c_l = sqrt((l*delta)^{2H} - ((l-1)*delta)^{2H})
    for l in range(1, max_kappa + 1):
        weights[l - 1] = np.sqrt((l * delta) ** two_H - ((l - 1) * delta) ** two_H)

    # Distant weights (l=kappa+1..n_steps): integral-averaged Volterra kernel
    # c_l = sqrt(2H * delta) * b_l
    for l in range(max_kappa + 1, n_steps + 1):
        b_l = ((l * delta) ** a_plus - ((l - 1) * delta) ** a_plus) / (a_plus * delta)
        weights[l - 1] = sqrt_2H_delta * b_l

    # Build fBm paths via cumulative convolution (parallelised over paths)
    fbm = np.zeros((n_paths, n_times), dtype=np.float64)

    for p in prange(n_paths):
        for i in range(1, n_times):
            # Z^H_{t_i} = sum_{l=1}^{i} weights[l-1] * dW[p, i-l]
            total = 0.0
            for l in range(1, i + 1):
                total += weights[l - 1] * dW[p, i - l]
            fbm[p, i] = total

    return fbm


class HybridScheme:
    """
    Full Hybrid Scheme for fBm simulation (Bennedsen, Lunde & Pakkanen 2017).

    Decomposes the Volterra kernel K(t-s) = (t-s)^{H-1/2} into:
    - Recent component (R): exact kernel evaluation for the last kappa steps
    - Distant component (D): averaged kernel weights for older increments

    This gives O(n * kappa) complexity per path, compared to O(n^2) for
    naive convolution or O(n^3) for full Cholesky.

    Parameters
    ----------
    H : float
        Hurst parameter (0 < H <= 0.5)
    kappa : int
        Truncation parameter controlling the recent/distant split.
        Larger kappa = more accurate but slower. Default: 6.

    References
    ----------
    Bennedsen, Lunde, Pakkanen (2017): "Hybrid Scheme for Brownian
    Semistationary Processes", Finance and Stochastics, Vol. 21.
    """

    def __init__(self, H: float, kappa: int = 6):
        self.H = H
        self.kappa = kappa

        self._T = None
        self._times = None
        self._n_steps = None

    def initialize(self, T: float, n_fine: int):
        """
        Initialize the scheme for a specific maturity and grid.

        Parameters
        ----------
        T : float
            Time to maturity
        n_fine : int
            Total number of time points (including t=0)
        """
        self._T = T
        self._n_steps = n_fine - 1
        self._times = np.linspace(0, T, n_fine)

        logger.debug(
            f"HybridScheme initialized: T={T}, n_steps={self._n_steps}, "
            f"kappa={self.kappa}"
        )

    def generate_paths(
        self,
        n_paths: int,
        random_state: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate fBm paths via the full Hybrid Scheme.

        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths
        random_state : np.random.Generator, optional
            Random number generator for reproducibility

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (fbm_paths, brownian_increments)
            fbm_paths: shape (n_paths, n_steps + 1)
            brownian_increments: shape (n_paths, n_steps) - for correlation
        """
        if self._times is None:
            raise RuntimeError(
                "HybridScheme not initialized. Call initialize() first."
            )

        if random_state is None:
            random_state = np.random.default_rng()

        # Generate standard Brownian increments
        dW = random_state.standard_normal((n_paths, self._n_steps))

        # Build fBm via Volterra convolution with hybrid kernel split
        delta = self._T / self._n_steps
        n_times = self._n_steps + 1
        fbm_paths = _hybrid_scheme_cumulative(
            dW, n_times, delta, self.H, self.kappa
        )

        return fbm_paths, dW


# =============================================================================
# MAIN PRICER CLASS
# =============================================================================

class rBergomiPricer:
    """
    Monte Carlo pricer for inverse options under the rough Bergomi model.

    The rough Bergomi model is defined by:
        dS_t / S_t = sqrt(V_t) * dW_t
        V_t = xi * exp(eta * Z^H_t - 0.5 * eta^2 * t^{2H})

    Where:
        - H is the Hurst parameter (0 < H <= 0.5, typically 0.05-0.15)
        - eta is the vol-of-vol parameter
        - rho is the correlation between spot and variance
        - xi is the forward variance level (can be constant or a curve)
        - Z^H is fractional Brownian motion

    For H = 0.5, the model reduces to standard log-normal volatility.
    For H < 0.5, volatility paths are "rougher" than Brownian motion.

    Parameters
    ----------
    H : float
        Hurst parameter (0 < H <= 0.5)
    eta : float
        Vol-of-vol parameter (eta > 0)
    rho : float
        Spot-vol correlation (-1 <= rho <= 1, typically negative)
    xi : float or callable
        Forward variance. If float, uses flat curve xi(t) = xi.
        If callable, should accept time t and return variance.
    n_paths : int
        Number of Monte Carlo paths (default: 50,000)
    n_steps : int
        Number of time steps (default: 100)
    n_coarse : int
        Coarse grid points for CholeskyScheme (default: 50, ignored if scheme='hybrid')
    scheme : str
        fBm simulation scheme:
        - 'cholesky': coarse-grid Cholesky + interpolation (default, fast)
        - 'hybrid':   full Hybrid Scheme with kernel decomposition (Bennedsen et al. 2017)
    kappa : int
        Truncation parameter for HybridScheme (default: 6, ignored if scheme='cholesky')
    seed : int, optional
        Random seed for reproducibility

    Attributes
    ----------
    tracker : PerformanceTracker
        Timing statistics for optimization analysis

    Example
    -------
    >>> pricer = rBergomiPricer(H=0.07, eta=1.9, rho=-0.7, xi=0.04)
    >>> price, std_err = pricer.price_inverse_put(S0=50000, K=48000, T=0.25)
    >>> print(f"Price: {price:.6f} BTC (+/- {std_err:.6f})")
    Price: 0.012345 BTC (+/- 0.000123)

    >>> # Using Cholesky scheme instead
    >>> pricer_ch = rBergomiPricer(H=0.07, eta=1.9, rho=-0.7, xi=0.04, scheme='cholesky')
    >>> price, std_err = pricer_hs.price_inverse_put(S0=50000, K=48000, T=0.25)

    Notes
    -----
    Performance tips:
    - For calibration, use n_paths=10,000-20,000 (faster, ~5% RMSE)
    - For final pricing, use n_paths=100,000+ (accurate, ~1% RMSE)
    - n_steps=50-100 is usually sufficient
    - scheme='hybrid' (default) is the Bennedsen et al. (2017) algorithm;
      scheme='cholesky' is ~2-3x faster
    """

    # Default configuration
    DEFAULT_N_PATHS = 50_000
    DEFAULT_N_STEPS = 100
    DEFAULT_N_COARSE = 50

    VALID_SCHEMES = ('cholesky', 'hybrid')
    VALID_PRICING_METHODS = ('euler', 'mixed')

    def __init__(
        self,
        H: float,
        eta: float,
        rho: float,
        xi: Union[float, Callable[[float], float]],
        n_paths: int = DEFAULT_N_PATHS,
        n_steps: int = DEFAULT_N_STEPS,
        n_coarse: int = DEFAULT_N_COARSE,
        scheme: str = 'hybrid',
        kappa: int = 6,
        seed: Optional[int] = None,
        antithetic: bool = False,
        pricing_method: str = 'euler',
    ):
        # Validate scheme and pricing method
        if scheme not in self.VALID_SCHEMES:
            raise ValueError(
                f"Unknown scheme '{scheme}'. Must be one of {self.VALID_SCHEMES}"
            )
        if pricing_method not in self.VALID_PRICING_METHODS:
            raise ValueError(
                f"Unknown pricing_method '{pricing_method}'. "
                f"Must be one of {self.VALID_PRICING_METHODS}"
            )
        if pricing_method == 'mixed' and scheme != 'hybrid':
            raise ValueError(
                "Mixed Estimator (pricing_method='mixed') requires scheme='hybrid'. "
                "The Cholesky scheme does not expose the driving BM increments on the "
                "fine grid needed for the conditional expectation."
            )

        # Validate parameters
        xi_val = xi if isinstance(xi, float) else xi(0.1)
        is_valid, error_msg = validate_params(H, eta, rho, xi_val)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")

        # Store parameters
        self.H = H
        self.eta = eta
        self.rho = rho
        self.xi = xi if callable(xi) else lambda t: xi
        self._xi_is_flat = not callable(xi) or isinstance(xi, (int, float))
        self._xi_flat_value = xi if isinstance(xi, (int, float)) else None

        # Simulation settings
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.n_coarse = n_coarse
        self.scheme = scheme
        self.kappa = kappa
        self.antithetic = antithetic
        self.pricing_method = pricing_method

        # Mixed Estimator instance (lazy, shared across calls)
        self._mixed_estimator = MixedEstimator() if pricing_method == 'mixed' else None

        # Random number generator
        self.rng = np.random.default_rng(seed)
        self._seed = seed

        # Initialize fBm scheme (will be re-initialized per maturity)
        if scheme == 'cholesky':
            self._fbm_scheme = CholeskyScheme(H, n_coarse)
        else:
            self._fbm_scheme = HybridScheme(H, kappa)
        self._current_T = None

        # Performance tracking
        self.tracker = PerformanceTracker()

        # Cache for repeated calls with same parameters
        self._cache = {}

        logger.info(
            f"rBergomiPricer initialized: H={H:.3f}, eta={eta:.2f}, "
            f"rho={rho:.2f}, xi={xi_val:.4f}, n_paths={n_paths:,}, "
            f"n_steps={n_steps}, scheme={scheme}"
        )

    def _ensure_initialized(self, T: float):
        """Initialize fBm scheme for given maturity if needed."""
        if self._current_T != T:
            self._fbm_scheme.initialize(T, self.n_steps)
            self._current_T = T


    def _generate_all_paths(
        self,
        S0: float,
        T: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate spot and variance paths via selected fBm scheme + Numba.

        Returns
        -------
        Tuple of:
            spot_paths : np.ndarray (n_paths, n_steps)
            var_paths  : np.ndarray (n_paths, n_steps)
            times      : np.ndarray (n_steps,)
        """
        return self._generate_all_paths_cpu(S0, T)

    def _generate_all_paths_cpu(
        self,
        S0: float,
        T: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Path generation via the selected fBm scheme + Numba JIT kernels.

        Pipeline:
            1. fBm scheme generates fractional BM on fine grid
            2. Variance paths from fBm (Numba)
            3. Correlated spot paths via log-Euler (Numba)
        """
        self._ensure_initialized(T)

        times = np.linspace(0, T, self.n_steps)
        dt = T / (self.n_steps - 1)

        # Orthogonal BM component for spot (independent of variance BM)
        with self.tracker.measure("random_generation"):
            dW_perp = self.rng.standard_normal((self.n_paths, self.n_steps - 1))

        # Generate fBm paths via the selected scheme
        with self.tracker.measure("fbm_generation"):
            fbm_paths, Z_driving = self._fbm_scheme.generate_paths(
                self.n_paths, self.rng
            )

        # Variance paths: V_t = xi(t) * exp(eta * Z^H_t - 0.5 * eta^2 * t^{2H})
        with self.tracker.measure("variance_paths"):
            if self._xi_is_flat and self._xi_flat_value is not None:
                xi_arr = np.full(len(times), self._xi_flat_value)
            else:
                xi_arr = np.array([self.xi(t) for t in times])
            var_paths = _generate_variance_paths_curve_numba(
                fbm_paths, times, xi_arr, self.eta, self.H
            )

        # Correlated spot paths
        # dW_spot = rho * dW1 + sqrt(1-rho^2) * dW_perp
        # Both schemes now return normalised BM increments on the fine grid
        with self.tracker.measure("spot_paths"):
            sqrt_one_minus_rho_sq = np.sqrt(1.0 - self.rho ** 2)

            # Ensure correct shape (n_paths, n_steps - 1)
            dW1 = Z_driving
            if dW1.shape[1] >= self.n_steps:
                dW1 = dW1[:, :self.n_steps - 1]

            dW_spot = self.rho * dW1 + sqrt_one_minus_rho_sq * dW_perp
            spot_paths = _generate_spot_paths_simple(S0, var_paths, dt, dW_spot)

        return spot_paths, var_paths, times

    def _generate_variance_paths_only(
        self,
        T: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate variance paths and driving BM increments (no spot paths).

        Used by the Mixed Estimator, which does not need simulated spot
        paths—it replaces them with conditional Black-Scholes evaluations.

        Returns
        -------
        var_paths : np.ndarray, shape (n_paths, n_steps)
            Simulated variance paths.
        dW_driving : np.ndarray, shape (n_paths, n_steps - 1)
            Standard normal increments of the BM W^⊥ that drives the
            variance process (needed for the correlation integral).
        dt : float
            Time step size.
        """
        self._ensure_initialized(T)

        times = np.linspace(0, T, self.n_steps)
        dt = T / (self.n_steps - 1)

        # Generate fBm paths via the Hybrid Scheme
        with self.tracker.measure("fbm_generation"):
            fbm_paths, dW_driving = self._fbm_scheme.generate_paths(
                self.n_paths, self.rng
            )

        # Variance paths: V_t = xi(t) * exp(eta * Z^H_t - 0.5 * eta^2 * t^{2H})
        with self.tracker.measure("variance_paths"):
            if self._xi_is_flat and self._xi_flat_value is not None:
                xi_arr = np.full(len(times), self._xi_flat_value)
            else:
                xi_arr = np.array([self.xi(t) for t in times])
            var_paths = _generate_variance_paths_curve_numba(
                fbm_paths, times, xi_arr, self.eta, self.H
            )

        # Ensure dW_driving has shape (n_paths, n_steps - 1)
        if dW_driving.shape[1] >= self.n_steps:
            dW_driving = dW_driving[:, :self.n_steps - 1]

        return var_paths, dW_driving, dt

    def price_inverse_put(
        self,
        S0: float,
        K: float,
        T: float,
        return_std: bool = True
    ) -> Union[float, Tuple[float, float]]:
        """
        Price an inverse put option using Monte Carlo simulation.

        Inverse put payoff: max(K/S_T - 1, 0) in BTC

        Parameters
        ----------
        S0 : float
            Current spot price (USD)
        K : float
            Strike price (USD)
        T : float
            Time to maturity (years)
        return_std : bool
            If True, also return standard error estimate

        Returns
        -------
        float or Tuple[float, float]
            Option price in BTC, and optionally standard error

        Example
        -------
        >>> price, std_err = pricer.price_inverse_put(50000, 48000, 0.25)
        >>> print(f"Price: {price:.6f} BTC (+/- {std_err:.6f})")
        """
        # Validate inputs
        is_valid, error_msg = validate_option_params(S0, K, T)
        if not is_valid:
            raise ValueError(error_msg)

        # Mixed Estimator path (no spot simulation needed)
        if self.pricing_method == 'mixed':
            with self.tracker.measure("total_pricing"):
                var_paths, dW_driving, dt = self._generate_variance_paths_only(T)
                with self.tracker.measure("mixed_estimator"):
                    price, std_err = self._mixed_estimator.price_single(
                        S0, K, var_paths, dW_driving, dt, self.rho,
                        option_type='put',
                    )
            if return_std:
                return price, std_err
            return price

        # Euler path (standard log-Euler spot simulation)
        with self.tracker.measure("total_pricing"):
            spot_paths, _, _ = self._generate_all_paths(S0, T)

            with self.tracker.measure("payoff_computation"):
                spot_terminal = spot_paths[:, -1]
                payoffs = _compute_inverse_put_payoffs(spot_terminal, K)
                price = float(np.mean(payoffs))
                std_err = float(np.std(payoffs)) / np.sqrt(self.n_paths)

        if return_std:
            return price, std_err
        return price

    def price_inverse_call(
        self,
        S0: float,
        K: float,
        T: float,
        return_std: bool = True
    ) -> Union[float, Tuple[float, float]]:
        """
        Price an inverse call option using Monte Carlo simulation.

        Inverse call payoff: max(1 - K/S_T, 0) in BTC

        Parameters
        ----------
        S0 : float
            Current spot price (USD)
        K : float
            Strike price (USD)
        T : float
            Time to maturity (years)
        return_std : bool
            If True, also return standard error estimate

        Returns
        -------
        float or Tuple[float, float]
            Option price in BTC, and optionally standard error
        """
        is_valid, error_msg = validate_option_params(S0, K, T)
        if not is_valid:
            raise ValueError(error_msg)

        # Mixed Estimator path
        if self.pricing_method == 'mixed':
            with self.tracker.measure("total_pricing"):
                var_paths, dW_driving, dt = self._generate_variance_paths_only(T)
                with self.tracker.measure("mixed_estimator"):
                    price, std_err = self._mixed_estimator.price_single(
                        S0, K, var_paths, dW_driving, dt, self.rho,
                        option_type='call',
                    )
            if return_std:
                return price, std_err
            return price

        # Euler path
        with self.tracker.measure("total_pricing"):
            spot_paths, _, _ = self._generate_all_paths(S0, T)

            with self.tracker.measure("payoff_computation"):
                spot_terminal = spot_paths[:, -1]
                payoffs = _compute_inverse_call_payoffs(spot_terminal, K)
                price = float(np.mean(payoffs))
                std_err = float(np.std(payoffs)) / np.sqrt(self.n_paths)

        if return_std:
            return price, std_err
        return price

    def price_inverse_option(
        self,
        S0: float,
        K: float,
        T: float,
        option_type: str = 'put',
        return_std: bool = True
    ) -> Union[float, Tuple[float, float]]:
        """
        Price an inverse option (wrapper for put/call).

        Parameters
        ----------
        S0 : float
            Current spot price (USD)
        K : float
            Strike price (USD)
        T : float
            Time to maturity (years)
        option_type : str
            'put' or 'call'
        return_std : bool
            If True, also return standard error

        Returns
        -------
        float or Tuple[float, float]
            Option price in BTC
        """
        if option_type.lower() in ['put', 'p']:
            return self.price_inverse_put(S0, K, T, return_std)
        elif option_type.lower() in ['call', 'c']:
            return self.price_inverse_call(S0, K, T, return_std)
        else:
            raise ValueError(f"Unknown option type: {option_type}")

    def implied_vol_inverse_put(
        self,
        S0: float,
        K: float,
        T: float,
        r: float = 0.0
    ) -> float:
        """
        Calculate model-implied volatility for an inverse put.

        Uses the Black-Scholes model as numeraire for IV calculation.

        Parameters
        ----------
        S0 : float
            Current spot price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate (default 0 for crypto)

        Returns
        -------
        float
            Implied volatility (decimal)
        """
        price, _ = self.price_inverse_put(S0, K, T)
        iv_arr = implied_vol_batch(
            np.array([price]), S0,
            np.array([K]), np.array([T]),
            option_type='put', r=r
        )
        return float(iv_arr[0])

    def implied_vol_inverse_call(
        self,
        S0: float,
        K: float,
        T: float,
        r: float = 0.0
    ) -> float:
        """Calculate model-implied volatility for an inverse call."""
        price, _ = self.price_inverse_call(S0, K, T)
        iv_arr = implied_vol_batch(
            np.array([price]), S0,
            np.array([K]), np.array([T]),
            option_type='call', r=r
        )
        return float(iv_arr[0])

    def price_multiple_strikes(
        self,
        S0: float,
        strikes: np.ndarray,
        T: float,
        option_type: str = 'put'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Price multiple strikes from a single set of MC paths.

        Generates paths ONCE, then computes payoffs for ALL strikes in a
        vectorized batch. This is much faster for calibration where the
        same (H, eta, rho, xi) are evaluated across many strikes.

        If antithetic=True, generates n_paths/2 base paths and mirrors them
        (negated BM increments) to get n_paths/2 antithetic paths. The final
        price is the average of both, halving variance at no extra path cost.

        Parameters
        ----------
        S0 : float
            Current spot price
        strikes : np.ndarray
            Array of strike prices
        T : float
            Time to maturity (same for all strikes)
        option_type : str
            'put' or 'call'

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            prices: shape (len(strikes),)
            std_errs: shape (len(strikes),)
        """
        # Mixed Estimator: generate variance paths once, price all strikes
        if self.pricing_method == 'mixed':
            var_paths, dW_driving, dt = self._generate_variance_paths_only(T)
            prices, std_errs = self._mixed_estimator.price_inverse_options(
                S0, strikes, var_paths, dW_driving, dt, self.rho,
                option_type=option_type,
            )
            return prices, std_errs

        # Euler path (standard log-Euler spot simulation)
        if self.antithetic:
            return self._price_multiple_strikes_antithetic(S0, strikes, T, option_type)

        spot_paths, _, _ = self._generate_all_paths(S0, T)
        spot_terminal = spot_paths[:, -1]  # (n_paths,)

        n_strikes = len(strikes)
        prices = np.empty(n_strikes)
        std_errs = np.empty(n_strikes)

        for i, K in enumerate(strikes):
            if option_type.lower() in ['put', 'p']:
                payoffs = _compute_inverse_put_payoffs(spot_terminal, K)
            else:
                payoffs = _compute_inverse_call_payoffs(spot_terminal, K)
            prices[i] = np.mean(payoffs)
            std_errs[i] = np.std(payoffs) / np.sqrt(self.n_paths)

        return prices, std_errs

    def _price_multiple_strikes_antithetic(
        self,
        S0: float,
        strikes: np.ndarray,
        T: float,
        option_type: str = 'put'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Antithetic variates pricing for multiple strikes.

        Generates n_paths/2 base paths, then constructs antithetic paths
        by negating the driving BM increments. For each path pair (i, i'),
        the payoff estimate is (payoff_i + payoff_i') / 2, which reduces
        variance when payoff is monotone in the underlying.
        """
        # Save original n_paths, generate half
        orig_n_paths = self.n_paths
        half_n = orig_n_paths // 2
        self.n_paths = half_n

        # Generate base paths
        spot_paths_base, var_paths_base, times = self._generate_all_paths(S0, T)
        S_base = spot_paths_base[:, -1]

        # Generate antithetic paths by negating driving BM
        # Re-seed to same state, then negate random draws
        self.rng = np.random.default_rng(self._seed)
        self._current_T = None  # force re-init

        self._ensure_initialized(T)
        dt = T / (self.n_steps - 1)

        # Generate same random draws but negate them
        dW_perp_anti = -self.rng.standard_normal((half_n, self.n_steps - 1))

        # Re-generate fBm with negated increments
        self.rng = np.random.default_rng(self._seed)
        self._current_T = None
        self._ensure_initialized(T)
        # Consume same random state for fBm
        _ = self.rng.standard_normal((half_n, self.n_steps - 1))  # dW_perp (thrown away)
        fbm_anti, Z_driving_anti = self._fbm_scheme.generate_paths(half_n, self.rng)
        fbm_anti = -fbm_anti  # negate fBm

        if self._xi_is_flat and self._xi_flat_value is not None:
            xi_arr = np.full(len(times), self._xi_flat_value)
        else:
            xi_arr = np.array([self.xi(t) for t in times])

        var_paths_anti = _generate_variance_paths_curve_numba(
            fbm_anti, times, xi_arr, self.eta, self.H
        )

        sqrt_one_minus_rho_sq = np.sqrt(1.0 - self.rho ** 2)
        Z_driving_anti = -Z_driving_anti
        dW1_anti = Z_driving_anti
        if dW1_anti.shape[1] >= self.n_steps:
            dW1_anti = dW1_anti[:, :self.n_steps - 1]

        dW2_anti = self.rho * dW1_anti + sqrt_one_minus_rho_sq * dW_perp_anti
        spot_paths_anti = _generate_spot_paths_simple(S0, var_paths_anti, dt, dW2_anti)
        S_anti = spot_paths_anti[:, -1]

        # Restore
        self.n_paths = orig_n_paths
        self.rng = np.random.default_rng(self._seed)

        # Compute paired payoffs
        n_strikes = len(strikes)
        prices = np.empty(n_strikes)
        std_errs = np.empty(n_strikes)
        is_put = option_type.lower() in ['put', 'p']

        for i, K in enumerate(strikes):
            if is_put:
                pay_base = _compute_inverse_put_payoffs(S_base, K)
                pay_anti = _compute_inverse_put_payoffs(S_anti, K)
            else:
                pay_base = _compute_inverse_call_payoffs(S_base, K)
                pay_anti = _compute_inverse_call_payoffs(S_anti, K)

            # Antithetic estimator: average of paired payoffs
            paired = 0.5 * (pay_base + pay_anti)
            prices[i] = np.mean(paired)
            std_errs[i] = np.std(paired) / np.sqrt(half_n)

        return prices, std_errs

    def price_surface(
        self,
        S0: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        option_type: str = 'put'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Price options across a grid of strikes and maturities.

        Parameters
        ----------
        S0 : float
            Current spot price
        strikes : np.ndarray
            Array of strike prices
        maturities : np.ndarray
            Array of maturities (years)
        option_type : str
            'put' or 'call'

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            prices: shape (len(strikes), len(maturities))
            std_errs: shape (len(strikes), len(maturities))
        """
        n_K = len(strikes)
        n_T = len(maturities)

        prices = np.empty((n_K, n_T))
        std_errs = np.empty((n_K, n_T))

        for j, T in enumerate(maturities):
            # Batch all strikes for this maturity in one MC run
            p, se = self.price_multiple_strikes(S0, strikes, T, option_type)
            prices[:, j] = p
            std_errs[:, j] = se
            logger.info(f"Surface pricing: maturity {j+1}/{n_T} complete")

        return prices, std_errs

    def implied_vol_surface(
        self,
        S0: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        option_type: str = 'put',
        r: float = 0.0
    ) -> np.ndarray:
        """
        Calculate implied volatility surface.

        Prices the surface, then inverts all points in one
        batch bisection call (flattened grid).

        Parameters
        ----------
        S0 : float
            Current spot price
        strikes : np.ndarray
            Array of strike prices
        maturities : np.ndarray
            Array of maturities (years)
        option_type : str
            'put' or 'call'
        r : float
            Risk-free rate

        Returns
        -------
        np.ndarray
            IV surface, shape (len(strikes), len(maturities))
        """
        prices, _ = self.price_surface(S0, strikes, maturities, option_type)

        n_K = len(strikes)
        n_T = len(maturities)

        # Flatten grid for single batch IV call
        prices_flat = prices.ravel(order='C')  # row-major: K varies fastest per T-column
        K_flat = np.tile(strikes, n_T)
        tau_flat = np.repeat(maturities, n_K)

        ivs_flat = implied_vol_batch(
            prices_flat, S0, K_flat, tau_flat,
            option_type=option_type, r=r
        )
        return ivs_flat.reshape(n_K, n_T, order='C')

    def reset_seed(self, seed: int):
        """Reset random number generator with new seed."""
        self.rng = np.random.default_rng(seed)
        self._seed = seed

    def get_performance_summary(self) -> str:
        """Return formatted performance timing summary."""
        return self.tracker.summary()

    def __repr__(self) -> str:
        xi_str = f"{self._xi_flat_value:.4f}" if self._xi_flat_value else "curve"
        return (
            f"rBergomiPricer(H={self.H:.3f}, eta={self.eta:.2f}, "
            f"rho={self.rho:.2f}, xi={xi_str}, "
            f"n_paths={self.n_paths:,}, n_steps={self.n_steps}, "
            f"scheme={self.scheme})"
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_price_inverse_put(
    S0: float,
    K: float,
    T: float,
    H: float = 0.07,
    eta: float = 1.9,
    rho: float = -0.7,
    xi: float = 0.04,
    n_paths: int = 10_000
) -> Tuple[float, float]:
    """
    Quick pricing function for one-off calculations.

    Uses smaller n_paths for speed. For production, use rBergomiPricer class.

    Parameters
    ----------
    S0, K, T : float
        Spot, strike, maturity
    H, eta, rho, xi : float
        rBergomi parameters
    n_paths : int
        Number of MC paths (default 10,000 for speed)

    Returns
    -------
    Tuple[float, float]
        (price, standard_error) in BTC
    """
    pricer = rBergomiPricer(H, eta, rho, xi, n_paths=n_paths, n_steps=50)
    return pricer.price_inverse_put(S0, K, T)


def quick_price_inverse_call(
    S0: float,
    K: float,
    T: float,
    H: float = 0.07,
    eta: float = 1.9,
    rho: float = -0.7,
    xi: float = 0.04,
    n_paths: int = 10_000
) -> Tuple[float, float]:
    """Quick pricing for inverse call. See quick_price_inverse_put for details."""
    pricer = rBergomiPricer(H, eta, rho, xi, n_paths=n_paths, n_steps=50)
    return pricer.price_inverse_call(S0, K, T)

