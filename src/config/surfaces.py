# src/config/surfaces.py
"""Shared IV surface extraction settings."""

SURFACE_CFG = dict(
    window_hours=4.0,
    min_volume=0.05,
    moneyness_range=(0.8, 1.2),
    min_ttm_days=7,
    max_ttm_days=90,
)
