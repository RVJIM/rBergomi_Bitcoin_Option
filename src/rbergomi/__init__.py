# src/rbergomi/__init__.py
"""
rBergomi Model Implementation for Bitcoin Inverse Options
==========================================================

Pricing and calibration of Bitcoin inverse options on Deribit using the
rough Bergomi model. fBm simulation via coarse-grid Cholesky scheme
(Bennedsen et al. 2017; McCrickerd & Pakkanen 2018), with an optional
full Hybrid Scheme (kernel decomposition with truncation parameter kappa).

Modules:
    pricer: Main rBergomiPricer class with Monte Carlo simulation
    calibrator: Calibration engine (Differential Evolution + Nelder-Mead)
    visualizer: Publication-quality figures for thesis
    utils: Helper functions (IV inversion, timing, formatting)

Optional Modules (require compilation):
    rbergomi_core: Cython-optimized kernels with OpenMP (~10-50x speedup)
        Build with: cd src/rbergomi && python setup.py build_ext --inplace

Example:
    >>> from src.rbergomi import rBergomiPricer, Calibrator, CalibrationVisualizer
    >>>
    >>> # Pricing
    >>> pricer = rBergomiPricer(H=0.07, eta=1.9, rho=-0.7, xi=0.04)
    >>> price, std_err = pricer.price_inverse_put(S0=50000, K=48000, T=0.25)
    >>> print(f"Price: {price:.6f} BTC (+/- {std_err:.6f})")
    >>>
    >>> # Calibration
    >>> cal = Calibrator(n_paths=10_000, n_steps=50)
    >>> result = cal.calibrate(market_data)
    >>> print(result)
    >>>
    >>> # Visualization
    >>> viz = CalibrationVisualizer()
    >>> viz.plot_parameter_stability(results_df)
    >>> viz.plot_skew_reproduction(result, market_data)

"""

from .pricer import rBergomiPricer
from .calibrator import Calibrator, CalibrationResult, ForwardVarianceCurve
from .mixed_estimator import MixedEstimator
from .visualizer import CalibrationVisualizer, create_visualizer
from .utils import (
    implied_vol_from_price_inverse,
    bs_price_inverse_put,
    bs_price_inverse_call,
    timer,
    format_european,
    PerformanceTracker
)

# Try to import Cython-optimized module
CYTHON_AVAILABLE = False
try:
    from .rbergomi_core import (
        fbm_covariance_matrix,
        generate_fbm_paths,
        generate_variance_paths,
        generate_spot_paths,
        compute_inverse_put_payoffs,
        compute_inverse_call_payoffs,
        price_inverse_put_cython,
        price_inverse_call_cython,
        price_multiple_strikes,
        simulate_rbergomi_paths
    )
    CYTHON_AVAILABLE = True
except ImportError:
    pass  # Cython module not compiled


__all__ = [
    # Core classes
    'rBergomiPricer',
    'MixedEstimator',
    'Calibrator',
    'CalibrationResult',
    'ForwardVarianceCurve',
    'CalibrationVisualizer',
    'create_visualizer',
    # Utility functions
    'implied_vol_from_price_inverse',
    'bs_price_inverse_put',
    'bs_price_inverse_call',
    'timer',
    'format_european',
    'PerformanceTracker',
    # Flags
    'CYTHON_AVAILABLE',
]

# Conditionally add Cython exports
if CYTHON_AVAILABLE:
    __all__.extend([
        'fbm_covariance_matrix',
        'generate_fbm_paths',
        'generate_variance_paths',
        'generate_spot_paths',
        'compute_inverse_put_payoffs',
        'compute_inverse_call_payoffs',
        'price_inverse_put_cython',
        'price_inverse_call_cython',
        'price_multiple_strikes',
        'simulate_rbergomi_paths',
    ])

__version__ = '2.1.0'
