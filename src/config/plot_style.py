# src/config/plot_style.py
"""Unified matplotlib styling for the thesis."""

import matplotlib.pyplot as plt
import matplotlib.cm as cm

THESIS_RCPARAMS = {
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
}

# ── Plasma-derived thesis palette ──────────────────────────────────────────
# All figures share colours sampled from the plasma colourmap for consistency.
_plasma = cm.get_cmap('plasma')
PALETTE = {
    'primary':    '#0d0887',   # plasma(0.00) — deep blue-purple
    'secondary':  '#7e03a8',   # plasma(0.25) — purple
    'tertiary':   '#cc4778',   # plasma(0.50) — magenta-pink
    'quaternary': '#f89540',   # plasma(0.75) — orange
    'accent':     '#f0f921',   # plasma(1.00) — bright yellow
    'neutral':    '#888888',   # grey for reference lines
}

SURFACE_CMAP = 'plasma'

METHOD_COLORS = {
    "cholesky_euler": PALETTE['primary'],    # deep blue-purple
    "hybrid_euler":   PALETTE['tertiary'],   # magenta-pink
    "hybrid_mixed":   PALETTE['quaternary'], # orange
}


def apply_thesis_style():
    """Apply the unified thesis plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update(THESIS_RCPARAMS)
