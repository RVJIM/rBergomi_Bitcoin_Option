# src/config/methods.py
"""Shared method configuration for the thesis pipeline."""

# Full calibration config (used by run_all_calibrations.py)
METHODS = {
    "cholesky_euler": dict(
        label       = "Cholesky + Euler MC",
        scheme      = "cholesky",
        pricing_method = "euler",
        kappa       = 6,
        n_paths     = 10_000,
        n_paths_coarse = 3_000,
        n_steps     = 100,
        seed        = 42,
        moneyness_range = (0.80, 1.20),
        popsize     = 12,
        antithetic  = False,
        maxiter     = 25,
    ),
    "hybrid_euler": dict(
        label       = "Hybrid Scheme + Euler MC",
        scheme      = "hybrid",
        pricing_method = "euler",
        kappa       = 6,
        n_paths     = 10_000,
        n_paths_coarse = 3_000,
        n_steps     = 100,
        seed        = 42,
        moneyness_range = (0.80, 1.20),
        popsize     = 12,
        antithetic  = False,
        maxiter     = 25,
    ),
    "hybrid_mixed": dict(
        label       = "Hybrid Scheme + Mixed Estimator",
        scheme      = "hybrid",
        pricing_method = "mixed",
        kappa       = 6,
        n_paths     = 10_000,
        n_paths_coarse = 3_000,
        n_steps     = 100,
        seed        = 42,
        moneyness_range = (0.80, 1.20),
        popsize     = 10,
        antithetic  = True,
        maxiter     = 25,
    ),
}

# Figure generation config (higher n_paths for publication quality)
METHOD_CFG = {
    "cholesky_euler": dict(
        scheme="cholesky", pricing_method="euler",
        n_paths=50_000, n_steps=100, kappa=6, seed=42, antithetic=False,
    ),
    "hybrid_euler": dict(
        scheme="hybrid", pricing_method="euler",
        n_paths=50_000, n_steps=100, kappa=6, seed=42, antithetic=False,
    ),
    "hybrid_mixed": dict(
        scheme="hybrid", pricing_method="mixed",
        n_paths=50_000, n_steps=100, kappa=6, seed=42, antithetic=True,
    ),
}

METHOD_LABELS = {
    "cholesky_euler": "Cholesky + Euler",
    "hybrid_euler":   "Hybrid + Euler",
    "hybrid_mixed":   "Hybrid + Mixed",
}

METHOD_LABELS_SHORT = {
    "cholesky_euler": "Chol.+Euler",
    "hybrid_euler":   "Hyb.+Euler",
    "hybrid_mixed":   "Hyb.+Mixed",
}

METHOD_COLORS = {
    "cholesky_euler": "#1f77b4",   # blue
    "hybrid_euler":   "#ff7f0e",   # orange
    "hybrid_mixed":   "#2ca02c",   # green
}
