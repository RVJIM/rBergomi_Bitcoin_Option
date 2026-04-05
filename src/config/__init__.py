# src/config/__init__.py
"""Shared configuration for the thesis pipeline."""

from .plot_style import apply_thesis_style, THESIS_RCPARAMS, PALETTE, SURFACE_CMAP
from .surfaces import SURFACE_CFG
from .dates import (
    EVENT_DATES, LOW_IV_DATES, MEDIUM_IV_DATES, HIGH_IV_DATES,
    BASELINE_DATES, ALL_DATES, SNAPSHOT_DATE_STRINGS,
)
from .methods import (
    METHODS, METHOD_CFG, METHOD_LABELS, METHOD_LABELS_SHORT,
    METHOD_COLORS,
)
