# inverse_options.py
"""
Inverse Options Analysis Module
===============================

Unified module for Bitcoin inverse options analysis:

1. COMPARISON - Greeks comparison between Direct and Inverse options
   - Delta and Gamma for Calls and Puts
   - Analysis across different maturities
   - Following Alexander et al. (2023)

2. DIDACTIC - Payoff analysis and mathematical decomposition
   - Inverse vs standard payoff structure
   - Decomposition into components (digital + reciprocal)
   - Key characteristics explanation

NOTE: This module should be run ONLY via main_c.py

Available commands:
    python main_c.py --inverse-comparison   # Greeks comparison: Direct vs Inverse
    python main_c.py --inverse-didactic     # Didactic payoff analysis
    python main_c.py --inverse-all          # All analyses
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path
import argparse
import sys
import logging

logger = logging.getLogger("InverseOptions")

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

CONFIG = {
    "output_dir": Path("Results/inverse_options"),
    "dpi_save": 300,
    "dpi_display": 150,

    # Default parameters
    "S": 50_000,           # Spot price
    "K": 50_000,           # Strike price
    "r": 0.0,              # Risk-free rate (0 per crypto)
    "r_btc": 0.0,          # BTC rate
    "sigma": 0.60,         # Volatility 60%

    # For multi-maturity comparison
    "comparison": {
        "S_fixed": 25_000,
        "sigma": 0.75,
        "taus": [10/365, 30/365, 90/365],
        "tau_labels": ['10 days', '30 days', '90 days'],
        "K_range": (10_000, 40_000)
    },

    # For didactic analysis
    "didactic": {
        "S_range_call": (20_000, 100_000),
        "S_range_put": (10_000, 100_000),
        "maturities": [0.02, 0.25, 1.0],  # 1 week, 3 months, 1 year
        "maturity_labels": ['1 week', '3 months', '1 year']
    }
}

from src.config.plot_style import apply_thesis_style, PALETTE
apply_thesis_style()
plt.rcParams.update({
    'figure.dpi': CONFIG["dpi_display"],
    'savefig.dpi': CONFIG["dpi_save"],
    'text.usetex': False,
})


# =============================================================================
# BLACK-SCHOLES FUNCTIONS (delegate to src.rbergomi.utils, adapt arg order)
# =============================================================================
from src.rbergomi.utils import bs_d1 as _bs_d1, bs_d2 as _bs_d2, bs_price_put_usd


def d1(S, K, r, sigma, tau):
    """Calculate d1 for Black-Scholes formula."""
    return _bs_d1(S, K, tau, r, sigma)


def d2(S, K, r, sigma, tau):
    """Calculate d2 for Black-Scholes formula."""
    return _bs_d2(S, K, tau, r, sigma)


# =============================================================================
# DIRECT (STANDARD) OPTION GREEKS
# Payoff: (S_T - K)+ for call, (K - S_T)+ for put [USD denominated]
# =============================================================================

def direct_call_delta(S, K, r, sigma, tau):
    """Delta of a direct (standard) call: dC/dS = Phi(d1)"""
    return norm.cdf(d1(S, K, r, sigma, tau))


def direct_put_delta(S, K, r, sigma, tau):
    """Delta of a direct (standard) put: dP/dS = Phi(d1) - 1"""
    return norm.cdf(d1(S, K, r, sigma, tau)) - 1


def direct_gamma(S, K, r, sigma, tau):
    """Gamma of a direct option: d2V/dS2 = phi(d1)/(S*sigma*sqrt(tau))"""
    d1_val = d1(S, K, r, sigma, tau)
    return norm.pdf(d1_val) / (S * sigma * np.sqrt(tau))


def direct_put_price(S, K, T, r, sigma):
    """Black-Scholes put option price (USD)"""
    return bs_price_put_usd(S, K, T, r, sigma)


# =============================================================================
# INVERSE OPTION GREEKS
# Payoff: (S_T - K)+/S_T for call, (K - S_T)+/S_T for put [BTC denominated]
# =============================================================================

def inverse_call_delta(S, K, r, r_btc, sigma, tau):
    """Delta of an inverse call option (BTC denominated)."""
    d2_val = d2(S, K, r, sigma, tau)
    return (K / S**2) * np.exp(-r_btc * tau) * norm.cdf(-d2_val)


def inverse_put_delta(S, K, r, r_btc, sigma, tau):
    """Delta of an inverse put option (BTC denominated)."""
    d2_val = d2(S, K, r, sigma, tau)
    return -(K / S**2) * np.exp(-r_btc * tau) * norm.cdf(d2_val)


def inverse_call_gamma(S, K, r, r_btc, sigma, tau):
    """Gamma of an inverse call (BTC denominated)."""
    d2_val = d2(S, K, r, sigma, tau)
    factor = (K / S**3) * np.exp(-r_btc * tau)
    return factor * (norm.pdf(d2_val) / (sigma * np.sqrt(tau)) - 2 * norm.cdf(-d2_val))


def inverse_put_gamma(S, K, r, r_btc, sigma, tau):
    """Gamma of an inverse put (BTC denominated)."""
    d2_val = d2(S, K, r, sigma, tau)
    factor = (K / S**3) * np.exp(-r_btc * tau)
    return factor * (norm.pdf(d2_val) / (sigma * np.sqrt(tau)) + 2 * norm.cdf(d2_val))


def inverse_put_delta_from_price(S, K, T, r, sigma):
    """Inverse put delta: Delta_USD/S - P_USD/S^2"""
    delta_usd = direct_put_delta(S, K, r, sigma, T)
    p_usd = direct_put_price(S, K, T, r, sigma)
    return delta_usd / S - p_usd / S**2


# =============================================================================
# COMPARISON MODE - Greeks Direct vs Inverse
# =============================================================================

class ComparisonAnalyzer:
    """Systematic Greeks comparison between Direct and Inverse options."""

    def __init__(self):
        self.output_dir = CONFIG["output_dir"] / "comparison"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = CONFIG["comparison"]

        self.S = self.cfg["S_fixed"]
        self.r = CONFIG["r"]
        self.r_btc = CONFIG["r_btc"]
        self.sigma = self.cfg["sigma"]
        self.taus = self.cfg["taus"]
        self.tau_labels = self.cfg["tau_labels"]
        self.K_range = np.linspace(*self.cfg["K_range"], 500)

        self.linestyles = ['-', '--', ':']
        self.linewidths = [2.5, 2, 1.5]

    def plot_call_delta(self):
        """Figure: Call Delta - Direct vs Inverse"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: Direct Call Delta
        ax1 = axes[0]
        for tau, label, ls, lw in zip(self.taus, self.tau_labels, self.linestyles, self.linewidths):
            delta_vals = direct_call_delta(self.S, self.K_range, self.r, self.sigma, tau)
            ax1.plot(self.K_range/1000, delta_vals, color=PALETTE['primary'], linestyle=ls, linewidth=lw, label=label)

        ax1.set_xlabel(r'Strike $K$ (thousands USD)')
        ax1.set_ylabel(r'Delta $\delta$')
        ax1.set_title(r'Direct Call Delta: $\frac{\partial C^{\$}}{\partial S}$')
        ax1.legend(loc='upper right', title='Maturity')
        ax1.set_xlim([10, 40])
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=self.S/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.5)

        # Panel B: Inverse Call Delta
        ax2 = axes[1]
        for tau, label, ls, lw in zip(self.taus, self.tau_labels, self.linestyles, self.linewidths):
            delta_vals = inverse_call_delta(self.S, self.K_range, self.r, self.r_btc, self.sigma, tau)
            ax2.plot(self.K_range/1000, delta_vals, color=PALETTE['tertiary'], linestyle=ls, linewidth=lw, label=label)

        ax2.set_xlabel(r'Strike $K$ (thousands USD)')
        ax2.set_ylabel(r'Delta $\delta$')
        ax2.set_title(r'Inverse Call Delta: $\frac{\partial C^{\mathbb{B}}}{\partial S}$')
        ax2.legend(loc='upper right', title='Maturity')
        ax2.set_xlim([10, 40])
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=self.S/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.5)

        plt.tight_layout()
        self._save_fig(fig, 'delta_call_direct_vs_inverse')

    def plot_put_delta(self):
        """Figure: Put Delta - Direct vs Inverse"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: Direct Put Delta
        ax1 = axes[0]
        for tau, label, ls, lw in zip(self.taus, self.tau_labels, self.linestyles, self.linewidths):
            delta_vals = direct_put_delta(self.S, self.K_range, self.r, self.sigma, tau)
            ax1.plot(self.K_range/1000, delta_vals, color=PALETTE['primary'], linestyle=ls, linewidth=lw, label=label)

        ax1.set_xlabel(r'Strike $K$ (thousands USD)')
        ax1.set_ylabel(r'Delta $\delta$')
        ax1.set_title(r'Direct Put Delta: $\frac{\partial P^{\$}}{\partial S}$')
        ax1.legend(loc='lower right', title='Maturity')
        ax1.set_xlim([10, 40])
        ax1.set_ylim([-2, 0.5])
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linewidth=0.5)
        ax1.axvline(x=self.S/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.5)

        # Panel B: Inverse Put Delta
        ax2 = axes[1]
        for tau, label, ls, lw in zip(self.taus, self.tau_labels, self.linestyles, self.linewidths):
            delta_vals = inverse_put_delta(self.S, self.K_range, self.r, self.r_btc, self.sigma, tau)
            ax2.plot(self.K_range/1000, delta_vals, color=PALETTE['tertiary'], linestyle=ls, linewidth=lw, label=label)

        ax2.set_xlabel(r'Strike $K$ (thousands USD)')
        ax2.set_ylabel(r'Delta $\delta$')
        ax2.set_title(r'Inverse Put Delta: $\frac{\partial P^{\mathbb{B}}}{\partial S}$')
        ax2.legend(loc='lower right', title='Maturity')
        ax2.set_xlim([10, 40])
        ax2.set_ylim([-2, 0.5])
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.axvline(x=self.S/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.5)

        plt.tight_layout()
        self._save_fig(fig, 'fig3_delta_put_BS')

    def plot_call_gamma(self):
        """Figure: Call Gamma - Direct vs Inverse"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: Direct Call Gamma
        ax1 = axes[0]
        for tau, label, ls, lw in zip(self.taus, self.tau_labels, self.linestyles, self.linewidths):
            gamma_vals = direct_gamma(self.S, self.K_range, self.r, self.sigma, tau) * 1e4
            ax1.plot(self.K_range/1000, gamma_vals, color=PALETTE['primary'], linestyle=ls, linewidth=lw, label=label)

        ax1.set_xlabel(r'Strike $K$ (thousands USD)')
        ax1.set_ylabel(r'Gamma $\gamma$ ($\times 10^{-4}$)')
        ax1.set_title(r'Direct Call Gamma: $\frac{\partial^2 C^{\$}}{\partial S^2}$')
        ax1.legend(loc='upper right', title='Maturity')
        ax1.set_xlim([10, 40])
        ax1.set_ylim([-1, 2])
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linewidth=0.5)
        ax1.axvline(x=self.S/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.5)

        # Panel B: Inverse Call Gamma
        ax2 = axes[1]
        for tau, label, ls, lw in zip(self.taus, self.tau_labels, self.linestyles, self.linewidths):
            gamma_vals = inverse_call_gamma(self.S, self.K_range, self.r, self.r_btc, self.sigma, tau) * 1e4
            ax2.plot(self.K_range/1000, gamma_vals, color=PALETTE['tertiary'], linestyle=ls, linewidth=lw, label=label)

        ax2.set_xlabel(r'Strike $K$ (thousands USD)')
        ax2.set_ylabel(r'Gamma $\gamma$ ($\times 10^{-4}$)')
        ax2.set_title(r'Inverse Call Gamma: $\frac{\partial^2 C^{\mathbb{B}}}{\partial S^2}$')
        ax2.legend(loc='upper right', title='Maturity')
        ax2.set_xlim([10, 40])
        ax2.set_ylim([-1, 2])
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.axvline(x=self.S/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.5)

        plt.tight_layout()
        self._save_fig(fig, 'gamma_call_direct_vs_inverse')

    def plot_put_gamma(self):
        """Figure: Put Gamma - Direct vs Inverse"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: Direct Put Gamma
        ax1 = axes[0]
        for tau, label, ls, lw in zip(self.taus, self.tau_labels, self.linestyles, self.linewidths):
            gamma_vals = direct_gamma(self.S, self.K_range, self.r, self.sigma, tau) * 1e4
            ax1.plot(self.K_range/1000, gamma_vals, color=PALETTE['primary'], linestyle=ls, linewidth=lw, label=label)

        ax1.set_xlabel(r'Strike $K$ (thousands USD)')
        ax1.set_ylabel(r'Gamma $\gamma$ ($\times 10^{-4}$)')
        ax1.set_title(r'Direct Put Gamma: $\frac{\partial^2 P^{\$}}{\partial S^2}$')
        ax1.legend(loc='upper right', title='Maturity')
        ax1.set_xlim([10, 40])
        ax1.set_ylim([0, 2])
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=self.S/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.5)

        # Panel B: Inverse Put Gamma
        ax2 = axes[1]
        for tau, label, ls, lw in zip(self.taus, self.tau_labels, self.linestyles, self.linewidths):
            gamma_vals = inverse_put_gamma(self.S, self.K_range, self.r, self.r_btc, self.sigma, tau) * 1e4
            ax2.plot(self.K_range/1000, gamma_vals, color=PALETTE['tertiary'], linestyle=ls, linewidth=lw, label=label)

        ax2.set_xlabel(r'Strike $K$ (thousands USD)')
        ax2.set_ylabel(r'Gamma $\gamma$ ($\times 10^{-4}$)')
        ax2.set_title(r'Inverse Put Gamma: $\frac{\partial^2 P^{\mathbb{B}}}{\partial S^2}$')
        ax2.legend(loc='upper right', title='Maturity')
        ax2.set_xlim([10, 40])
        ax2.set_ylim([0, 2])
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=self.S/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.5)

        plt.tight_layout()
        self._save_fig(fig, 'gamma_put_direct_vs_inverse')

    def plot_greeks_panel_put(self):
        """Figure: 2x2 Panel - Put Greeks (Delta & Gamma)"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Row 1: Delta
        for ax, delta_fn, color, title in [
            (axes[0, 0], lambda K, tau: direct_put_delta(self.S, K, self.r, self.sigma, tau), PALETTE['primary'], 'Direct'),
            (axes[0, 1], lambda K, tau: inverse_put_delta(self.S, K, self.r, self.r_btc, self.sigma, tau), PALETTE['tertiary'], 'Inverse')
        ]:
            for tau, label, ls, lw in zip(self.taus, self.tau_labels, self.linestyles, self.linewidths):
                ax.plot(self.K_range/1000, delta_fn(self.K_range, tau), color=color, linestyle=ls, linewidth=lw, label=label)
            ax.set_xlabel(r'Strike $K$ (thousands USD)')
            ax.set_ylabel(r'Delta $\delta$')
            ax.set_title(title)
            ax.legend(loc='lower right', title='Maturity')
            ax.set_xlim([10, 40])
            ax.set_ylim([-2, 0.5])
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.axvline(x=self.S/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.5)

        # Row 2: Gamma
        for ax, gamma_fn, color, title in [
            (axes[1, 0], lambda K, tau: direct_gamma(self.S, K, self.r, self.sigma, tau), PALETTE['primary'], 'Direct'),
            (axes[1, 1], lambda K, tau: inverse_put_gamma(self.S, K, self.r, self.r_btc, self.sigma, tau), PALETTE['tertiary'], 'Inverse')
        ]:
            for tau, label, ls, lw in zip(self.taus, self.tau_labels, self.linestyles, self.linewidths):
                ax.plot(self.K_range/1000, gamma_fn(self.K_range, tau) * 1e4, color=color, linestyle=ls, linewidth=lw, label=label)
            ax.set_xlabel(r'Strike $K$ (thousands USD)')
            ax.set_ylabel(r'Gamma $\gamma$ ($\times 10^{-4}$)')
            ax.set_title(title)
            ax.legend(loc='upper right', title='Maturity')
            ax.set_xlim([10, 40])
            ax.set_ylim([0, 2])
            ax.grid(True, alpha=0.3)
            ax.axvline(x=self.S/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.5)

        fig.text(0.02, 0.75, 'Delta', ha='center', va='center', rotation='vertical', fontsize=14, fontweight='bold')
        fig.text(0.02, 0.3, 'Gamma', ha='center', va='center', rotation='vertical', fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0.03, 0, 1, 1])
        self._save_fig(fig, 'greeks_panel_put')

    def plot_greeks_panel_call(self):
        """Figure: 2x2 Panel - Call Greeks (Delta & Gamma)"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Row 1: Delta
        for ax, delta_fn, color, title in [
            (axes[0, 0], lambda K, tau: direct_call_delta(self.S, K, self.r, self.sigma, tau), PALETTE['primary'], 'Direct'),
            (axes[0, 1], lambda K, tau: inverse_call_delta(self.S, K, self.r, self.r_btc, self.sigma, tau), PALETTE['tertiary'], 'Inverse')
        ]:
            for tau, label, ls, lw in zip(self.taus, self.tau_labels, self.linestyles, self.linewidths):
                ax.plot(self.K_range/1000, delta_fn(self.K_range, tau), color=color, linestyle=ls, linewidth=lw, label=label)
            ax.set_xlabel(r'Strike $K$ (thousands USD)')
            ax.set_ylabel(r'Delta $\delta$')
            ax.set_title(title)
            ax.legend(loc='upper right', title='Maturity')
            ax.set_xlim([10, 40])
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
            ax.axvline(x=self.S/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.5)

        # Row 2: Gamma
        for ax, gamma_fn, color, title in [
            (axes[1, 0], lambda K, tau: direct_gamma(self.S, K, self.r, self.sigma, tau), PALETTE['primary'], 'Direct'),
            (axes[1, 1], lambda K, tau: inverse_call_gamma(self.S, K, self.r, self.r_btc, self.sigma, tau), PALETTE['tertiary'], 'Inverse')
        ]:
            for tau, label, ls, lw in zip(self.taus, self.tau_labels, self.linestyles, self.linewidths):
                ax.plot(self.K_range/1000, gamma_fn(self.K_range, tau) * 1e4, color=color, linestyle=ls, linewidth=lw, label=label)
            ax.set_xlabel(r'Strike $K$ (thousands USD)')
            ax.set_ylabel(r'Gamma $\gamma$ ($\times 10^{-4}$)')
            ax.set_title(title)
            ax.legend(loc='upper right', title='Maturity')
            ax.set_xlim([10, 40])
            ax.set_ylim([-1, 1.5])
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.axvline(x=self.S/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.5)

        fig.text(0.02, 0.75, 'Delta', ha='center', va='center', rotation='vertical', fontsize=14, fontweight='bold')
        fig.text(0.02, 0.3, 'Gamma', ha='center', va='center', rotation='vertical', fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0.03, 0, 1, 1])
        self._save_fig(fig, 'greeks_panel_call')

    def _save_fig(self, fig, name):
        """Save figure in PNG and PDF formats."""
        for ext in ['png']:
            path = self.output_dir / f"{name}.{ext}"
            fig.savefig(path, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved: {name}.png")
        plt.close(fig)

    def print_summary(self):
        """Print summary of key differences."""
        print("\n" + "="*70)
        print("DIRECT vs INVERSE OPTIONS - KEY DIFFERENCES")
        print("="*70)
        print("\nPAYOFF STRUCTURE:")
        print("  Direct Call:  (S_T - K)+         [USD denominated]")
        print("  Inverse Call: (S_T - K)+ / S_T   [BTC denominated]")
        print("  Direct Put:   (K - S_T)+         [USD denominated]")
        print("  Inverse Put:  (K - S_T)+ / S_T   [BTC denominated]")
        print("\nDELTA DIFFERENCES:")
        print("  - Direct call delta: monotonically decreasing in K (0 to 1)")
        print("  - Inverse call delta: NON-MONOTONIC, peaks near ATM")
        print("  - Direct put delta: bounded by -1")
        print("  - Inverse put delta: can exceed -1 (unbounded!)")
        print("\nGAMMA DIFFERENCES:")
        print("  - Direct gamma: always POSITIVE")
        print("  - Inverse call gamma: can be NEGATIVE for ITM")
        print("  - Inverse put gamma: higher than direct for high strikes")
        print("\nHEDGING IMPLICATIONS:")
        print("  - Inverse options require different hedging strategies")
        print("  - Higher hedging costs due to unbounded delta")
        print("="*70)

    def run(self):
        """Run all comparison analyses."""
        logger.info("Starting Greeks COMPARISON analysis...")

        self.plot_call_delta()
        self.plot_put_delta()
        self.plot_call_gamma()
        self.plot_put_gamma()
        self.plot_greeks_panel_call()
        self.plot_greeks_panel_put()

        self.print_summary()
        logger.info(f"Analysis completed. Output: {self.output_dir}")


# =============================================================================
# DIDACTIC MODE - Payoff Analysis and Decomposition
# =============================================================================

class DidacticAnalyzer:
    """Didactic payoff analysis and inverse option decomposition."""

    def __init__(self):
        self.output_dir = CONFIG["output_dir"] / "didactic"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = CONFIG["didactic"]

        self.K = CONFIG["K"]
        self.r = CONFIG["r"]
        self.sigma = CONFIG["sigma"]

    def plot_call_payoff_comparison(self):
        """Figure: Standard vs Inverse Call Payoff"""
        S_T = np.linspace(*self.cfg["S_range_call"], 1000)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel A: Standard Call (USD)
        ax1 = axes[0]
        standard_call = np.maximum(S_T - self.K, 0)

        ax1.plot(S_T/1000, standard_call/1000, color=PALETTE['primary'], linestyle='-', linewidth=2.5,
                 label=r'Standard Call: $\max(S_T - K, 0)$')
        ax1.axvline(x=self.K/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.7)
        ax1.fill_between(S_T/1000, 0, standard_call/1000, alpha=0.15, color=PALETTE['primary'])

        ax1.set_xlabel(r'Spot Price $S_T$ (thousands USD)')
        ax1.set_ylabel('Payoff (thousands USD)')
        ax1.set_title('(a) Standard Call Option Payoff (USD)')
        ax1.legend(loc='upper left')
        ax1.set_xlim([20, 100])
        ax1.set_ylim([0, 55])
        ax1.grid(True, alpha=0.3)

        # Panel B: Inverse Call (BTC)
        ax2 = axes[1]
        inverse_call = np.maximum(1 - self.K/S_T, 0)

        ax2.plot(S_T/1000, inverse_call, color=PALETTE['tertiary'], linestyle='-', linewidth=2.5,
                 label=r'Inverse Call: $\max\left(1 - \frac{K}{S_T}, 0\right)$')
        ax2.axvline(x=self.K/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.7)
        ax2.axhline(y=1, color=PALETTE['quaternary'], linestyle=':', alpha=0.7, label='Asymptotic limit = 1 BTC')
        ax2.fill_between(S_T/1000, 0, inverse_call, alpha=0.15, color=PALETTE['tertiary'])

        ax2.set_xlabel(r'Spot Price $S_T$ (thousands USD)')
        ax2.set_ylabel('Payoff (BTC)')
        ax2.set_title('(b) Inverse Call Option Payoff (BTC)')
        ax2.legend(loc='right')
        ax2.set_xlim([20, 100])
        ax2.set_ylim([0, 1.1])
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_fig(fig, 'fig1_payoff_comparison_call')

    def plot_put_payoff_comparison(self):
        """Figure: Standard vs Inverse Put Payoff"""
        S_T = np.linspace(*self.cfg["S_range_put"], 1000)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Panel A: Standard Put (USD)
        ax1 = axes[0]
        standard_put = np.maximum(self.K - S_T, 0)

        ax1.plot(S_T/1000, standard_put/1000, color=PALETTE['primary'], linestyle='-', linewidth=2.5,
                 label=r'Standard Put: $\max(K - S_T, 0)$')
        ax1.axvline(x=self.K/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.7)
        ax1.fill_between(S_T/1000, 0, standard_put/1000, alpha=0.15, color=PALETTE['primary'])

        ax1.set_xlabel(r'Spot Price $S_T$ (thousands USD)')
        ax1.set_ylabel('Payoff (thousands USD)')
        ax1.set_title('(a) Standard Put Option Payoff (USD)')
        ax1.legend(loc='upper right')
        ax1.set_xlim([10, 100])
        ax1.set_ylim([0, 45])
        ax1.grid(True, alpha=0.3)

        # Panel B: Inverse Put (BTC)
        ax2 = axes[1]
        inverse_put = np.maximum(self.K/S_T - 1, 0)

        ax2.plot(S_T/1000, inverse_put, color=PALETTE['tertiary'], linestyle='-', linewidth=2.5,
                 label=r'Inverse Put: $\max\left(\frac{K}{S_T} - 1, 0\right)$')
        ax2.axvline(x=self.K/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.7)
        ax2.fill_between(S_T/1000, 0, inverse_put, alpha=0.15, color=PALETTE['tertiary'])

        ax2.annotate('Payoff diverges\nas $S_T \\to 0$', xy=(12, 3.5), fontsize=10, style='italic',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax2.set_xlabel(r'Spot Price $S_T$ (thousands USD)')
        ax2.set_ylabel('Payoff (BTC)')
        ax2.set_title('(b) Inverse Put Option Payoff (BTC)')
        ax2.legend(loc='upper right')
        ax2.set_xlim([10, 100])
        ax2.set_ylim([0, 4.5])
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_fig(fig, 'fig1_payoff_comparison_put')

    def plot_call_decomposition(self):
        """Figure: Inverse Call Decomposition"""
        S_T = np.linspace(*self.cfg["S_range_call"], 1000)

        fig, ax = plt.subplots(figsize=(10, 6))

        digital_option = (S_T > self.K).astype(float)
        reciprocal_put = self.K * np.maximum(1/S_T - 1/self.K, 0)
        inverse_call = np.maximum(1 - self.K/S_T, 0)

        ax.plot(S_T/1000, digital_option, 'g-', linewidth=2,
                label=r'Digital Option: $\mathbf{1}_{S_T > K}$')
        ax.plot(S_T/1000, reciprocal_put, 'm-', linewidth=2,
                label=r'Reciprocal Put: $K \cdot \max\left(\frac{1}{S_T} - \frac{1}{K}, 0\right)$')
        ax.plot(S_T/1000, inverse_call, color=PALETTE['tertiary'], linestyle='-', linewidth=2.5,
                label=r'Inverse Call: $\max\left(1 - \frac{K}{S_T}, 0\right)$')

        ax.axvline(x=self.K/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.7)
        ax.fill_between(S_T/1000, reciprocal_put, digital_option, where=(S_T > self.K), alpha=0.1, color=PALETTE['tertiary'])

        ax.set_xlabel(r'Spot Price $S_T$ (thousands USD)')
        ax.set_ylabel('Payoff (BTC)')
        ax.set_title(r'Inverse Call = Digital Option $-$ Reciprocal Put')
        ax.legend(loc='upper right', framealpha=0.95)
        ax.set_xlim([20, 100])
        ax.set_ylim([-0.05, 1.15])
        ax.grid(True, alpha=0.3)

        textbox = "The inverse call equals a digital option\nminus K times a 'reciprocal put' on 1/S"
        ax.text(0.02, 0.98, textbox, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        self._save_fig(fig, 'fig2_decomposition_call')

    def plot_put_decomposition(self):
        """Figure: Inverse Put Decomposition"""
        S_T = np.linspace(*self.cfg["S_range_put"], 1000)

        fig, ax = plt.subplots(figsize=(10, 6))

        reciprocal_call = self.K * np.maximum(1/S_T - 1/self.K, 0)
        inverse_put = np.maximum(self.K/S_T - 1, 0)
        digital_put = (S_T < self.K).astype(float)

        ax.plot(S_T/1000, inverse_put, color=PALETTE['tertiary'], linestyle='-', linewidth=2.5,
                label=r'Inverse Put: $\max\left(\frac{K}{S_T} - 1, 0\right)$')
        ax.plot(S_T/1000, digital_put, 'g-', linewidth=2, alpha=0.8,
                label=r'Digital Put: $\mathbf{1}_{S_T < K}$')
        ax.plot(S_T/1000, reciprocal_call, 'm--', linewidth=2, alpha=0.8,
                label=r'$K \cdot \max\left(\frac{1}{S_T} - \frac{1}{K}, 0\right)$')

        ax.axvline(x=self.K/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.7)

        ax.set_xlabel(r'Spot Price $S_T$ (thousands USD)')
        ax.set_ylabel('Payoff (BTC)')
        ax.set_title(r'Inverse Put = $K \times$ Reciprocal Call on $1/S_T$')
        ax.legend(loc='upper right', framealpha=0.95)
        ax.set_xlim([10, 100])
        ax.set_ylim([-0.1, 4.5])
        ax.grid(True, alpha=0.3)

        textbox = "The inverse put equals K times a\n'reciprocal call' on 1/S with strike 1/K"
        ax.text(0.55, 0.75, textbox, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        self._save_fig(fig, 'decomposition_put')

    def plot_delta_maturity_comparison(self):
        """Figure: Delta across maturities"""
        S = np.linspace(20_000, 100_000, 500)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = ['green', PALETTE['primary'], 'purple']

        # Panel A: Standard Put Delta
        ax1 = axes[0]
        for T, color, label in zip(self.cfg["maturities"], colors, self.cfg["maturity_labels"]):
            delta = direct_put_delta(S, self.K, self.r, self.sigma, T)
            ax1.plot(S/1000, delta, color=color, linewidth=2, label=f'T = {label}')

        ax1.axvline(x=self.K/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.7)
        ax1.axhline(y=-1, color=PALETTE['neutral'], linestyle=':', alpha=0.5)
        ax1.set_xlabel(r'Spot Price $S$ (thousands USD)')
        ax1.set_ylabel(r'$\Delta^{USD}$')
        ax1.set_title(f'(a) Standard Put Delta ($\\sigma$={self.sigma:.0%})')
        ax1.legend(loc='lower right')
        ax1.set_xlim([20, 100])
        ax1.set_ylim([-1.1, 0.1])
        ax1.grid(True, alpha=0.3)

        # Panel B: Inverse Put Delta
        ax2 = axes[1]
        for T, color, label in zip(self.cfg["maturities"], colors, self.cfg["maturity_labels"]):
            delta = inverse_put_delta_from_price(S, self.K, T, self.r, self.sigma)
            ax2.plot(S/1000, delta * 1e5, color=color, linewidth=2, label=f'T = {label}')

        ax2.axvline(x=self.K/1000, color=PALETTE['neutral'], linestyle='--', alpha=0.7)
        ax2.set_xlabel(r'Spot Price $S$ (thousands USD)')
        ax2.set_ylabel(r'$\Delta^{BTC} \times 10^5$')
        ax2.set_title(f'(b) Inverse Put Delta ($\\sigma$={self.sigma:.0%})')
        ax2.legend(loc='lower right')
        ax2.set_xlim([20, 100])
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_fig(fig, 'delta_maturities_put')

    def _save_fig(self, fig, name):
        """Save figure in PNG and PDF formats."""
        for ext in ['png']:
            path = self.output_dir / f"{name}.{ext}"
            fig.savefig(path, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved: {name}.png")
        plt.close(fig)

    def print_summary(self):
        """Print educational summary."""
        print("\n" + "="*70)
        print("INVERSE OPTIONS - KEY INSIGHTS")
        print("="*70)

        print("\nCALL OPTIONS:")
        print("  - Standard call: LINEAR payoff, UNBOUNDED upside (USD)")
        print("  - Inverse call: CONCAVE payoff, CAPPED at 1 BTC")
        print("  - Decomposition: Inverse Call = Digital - K x Reciprocal Put")

        print("\nPUT OPTIONS:")
        print("  - Standard put: LINEAR payoff, CAPPED at K (USD)")
        print("  - Inverse put: CONVEX payoff, UNBOUNDED as S->0 (BTC)")
        print("  - Decomposition: Inverse Put = K x Reciprocal Call")

        print("\nCRITICAL RISK:")
        print("  - Inverse put BTC payoff is UNBOUNDED")
        print("  - This is a key risk characteristic on Deribit!")

        print("="*70)

    def run(self):
        """Run all didactic analyses."""
        logger.info("Starting DIDACTIC payoff analysis...")

        self.plot_call_payoff_comparison()
        self.plot_put_payoff_comparison()
        self.plot_call_decomposition()
        self.plot_put_decomposition()
        self.plot_delta_maturity_comparison()

        self.print_summary()
        logger.info(f"Analysis completed. Output: {self.output_dir}")


# =============================================================================
# PUBLIC FUNCTIONS (called from main_c.py)
# =============================================================================

def run_comparison_analysis():
    """Run Greeks comparison analysis."""
    analyzer = ComparisonAnalyzer()
    analyzer.run()


def run_didactic_analysis():
    """Run didactic payoff analysis."""
    analyzer = DidacticAnalyzer()
    analyzer.run()


def run_all_inverse_analysis():
    """Run all analyses."""
    logger.info("="*60)
    logger.info("FULL INVERSE OPTIONS ANALYSIS")
    logger.info("="*60)

    run_comparison_analysis()
    run_didactic_analysis()

    logger.info("="*60)
    logger.info("ALL ANALYSES COMPLETED")
    logger.info("="*60)

    # Summary
    print("\nGenerated output in Results/inverse_options/:")
    for subdir in ["comparison", "didactic"]:
        path = CONFIG["output_dir"] / subdir
        if path.exists():
            files = list(path.glob("*.png"))
            print(f"  {subdir}/: {len(files)} figures (PNG + PDF)")

# NOTE: This module should be run only via main_c.py
# Example: python main_c.py --inverse-comparison
