# src/rbergomi/visualizer.py
"""
rBergomi Calibration Visualizer
================================

Generates publication-quality figures for thesis Chapter 3:
1. Parameter stability time series (H, eta, rho, xi over 2022-2025)
2. Skew reproduction plots (market IV vs model IV)
3. Residual heatmap (moneyness x maturity -> error)
4. LaTeX-ready tables with European formatting

"""

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm

import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
import logging

from .utils import format_european, format_percentage

logger = logging.getLogger("rBergomi.visualizer")

# =============================================================================
# PLOT STYLING
# =============================================================================

from src.config.plot_style import apply_thesis_style, PALETTE
apply_thesis_style()

# Plasma-derived palette for consistency across all thesis figures
COLORS = {
    'H': PALETTE['primary'],       # deep blue-purple
    'eta': PALETTE['tertiary'],    # magenta-pink
    'rho': PALETTE['quaternary'],  # orange
    'xi': PALETTE['secondary'],    # purple
    'market': PALETTE['primary'],  # deep blue-purple
    'model': PALETTE['tertiary'],  # magenta-pink
    'error_pos': PALETTE['tertiary'],   # magenta-pink for positive errors
    'error_neg': PALETTE['primary'],    # deep blue-purple for negative errors
    'spot': PALETTE['quaternary'],      # orange
    'atm': PALETTE['quaternary'],       # orange
}


# =============================================================================
# CALIBRATION VISUALIZER
# =============================================================================

class CalibrationVisualizer:
    """
    Generate publication-quality visualizations for rBergomi calibration results.

    Parameters
    ----------
    output_dir : str or Path
        Directory to save figures

    Example
    -------
    >>> from src.rbergomi import CalibrationVisualizer
    >>>
    >>> viz = CalibrationVisualizer("Results/calibration/figures")
    >>> viz.plot_parameter_stability(calibration_results_df)
    >>> viz.plot_skew_reproduction(result, market_data)
    >>> viz.plot_residuals_heatmap(result)
    """

    # Map (scheme, pricing_method) combinations to model subfolder names
    MODEL_SUBFOLDER = {
        ("cholesky", "euler"): "rbergomi",
        ("cholesky", "mixed"): "rbergomi",
        ("cholesky", ""):      "rbergomi",
        ("hybrid", "euler"):   "hybrid",
        ("hybrid", "mixed"):   "hybrid",
        ("hybrid", ""):        "blp2017",
    }

    def __init__(self, output_dir: str = "Results/calibration/figures",
                 scheme: str = "hybrid", pricing_method: str = "euler"):
        self.scheme = scheme
        self.pricing_method = pricing_method
        # Tag used in all output filenames: e.g. "hybrid_mixed", "cholesky_euler"
        self.tag = f"{scheme}_{pricing_method}" if pricing_method else scheme

        # Resolve model subfolder from (scheme, pricing_method) mapping
        subfolder = self.MODEL_SUBFOLDER.get(
            (scheme, pricing_method),
            self.MODEL_SUBFOLDER.get((scheme, ""), scheme)
        )
        self.output_dir = Path(output_dir) / subfolder
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Also create tables directory (mirroring the same subfolder structure)
        self.tables_dir = Path(output_dir).parent / "tables" / subfolder
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"CalibrationVisualizer initialized: {self.output_dir} "
                     f"(scheme={scheme}, pricing_method={pricing_method}, "
                     f"model_subfolder={subfolder})")

    # =========================================================================
    # 1. PARAMETER STABILITY TIME SERIES
    # =========================================================================

    def plot_parameter_stability(
        self,
        results: pl.DataFrame,
        events: Optional[List[Dict]] = None,
        save_name: str = "parameter_stability_timeseries",
        show: bool = False
    ) -> plt.Figure:
        """
        Plot time series of calibrated parameters (H, eta, rho, xi).

        Parameters
        ----------
        results : pl.DataFrame
            Calibration results with columns: date, H, eta, rho, xi, rmse_pp
        events : list of dict, optional
            Market events to annotate [{date, name}, ...]
        save_name : str
            Filename (without extension)
        show : bool
            Whether to display the plot

        Returns
        -------
        plt.Figure
        """
        # Convert to pandas for easier plotting
        if isinstance(results, pl.DataFrame):
            df = results.to_pandas()
        else:
            df = results

        df['date'] = pd.to_datetime(df['date'], format='mixed')
        df = df.sort_values('date')

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

        params = [
            ('H', r'$H$ (Hurst)', COLORS['H'], (0, 0.5)),
            ('eta', r'$\eta$ (Vol-of-Vol)', COLORS['eta'], (0, 5)),
            ('rho', r'$\rho$ (Correlation)', COLORS['rho'], (-1, 0)),
            ('xi', r'$\xi$ (Forward Variance)', COLORS['xi'], (0, 1)),
        ]

        for ax, (param, label, color, ylim) in zip(axes.flat, params):
            if param not in df.columns:
                continue

            # Plot parameter values
            ax.plot(df['date'], df[param], color=color, linewidth=1.5, alpha=0.8)
            ax.fill_between(df['date'], df[param], alpha=0.2, color=color)

            # Add mean line
            mean_val = df[param].mean()
            ax.axhline(mean_val, color=color, linestyle='--', alpha=0.5,
                      label=f'Mean = {mean_val:.3f}')

            # Add events
            if events:
                for event in events:
                    event_date = pd.to_datetime(event['date'])
                    if df['date'].min() <= event_date <= df['date'].max():
                        ax.axvline(event_date, color='gray', linestyle=':', alpha=0.5)

            ax.set_ylabel(label)
            ax.set_ylim(ylim)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)

        # Format x-axis
        for ax in axes[1, :]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

        fig.autofmt_xdate()
        fig.suptitle('rBergomi Parameter Stability Over Time', fontsize=14, fontweight='bold')
        plt.tight_layout()

        self._save_figure(fig, save_name)

        plt.close(fig)

        return fig

    def plot_parameter_with_spot(
        self,
        results: pl.DataFrame,
        param: str = 'H',
        events: Optional[List[Dict]] = None,
        save_name: str = "param_vs_spot",
        show: bool = False
    ) -> plt.Figure:
        """
        Plot a single parameter alongside BTC spot price.

        Parameters
        ----------
        results : pl.DataFrame
            Calibration results with columns: date, param, spot
        param : str
            Parameter to plot ('H', 'eta', 'rho', 'xi')
        """
        if isinstance(results, pl.DataFrame):
            df = results.to_pandas()
        else:
            df = results

        df['date'] = pd.to_datetime(df['date'], format='mixed')
        df = df.sort_values('date')

        fig, ax1 = plt.subplots(figsize=(14, 6))

        # Plot parameter
        param_labels = {
            'H': r'$H$ (Hurst)',
            'eta': r'$\eta$ (Vol-of-Vol)',
            'rho': r'$\rho$ (Correlation)',
            'xi': r'$\xi$ (Forward Variance)'
        }

        color1 = COLORS.get(param, 'blue')
        ax1.plot(df['date'], df[param], color=color1, linewidth=1.5, label=param_labels.get(param, param))
        ax1.set_ylabel(param_labels.get(param, param), color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        # Plot spot on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(df['date'], df['spot'] / 1000, color=COLORS['spot'], linewidth=1, alpha=0.6, label='BTC Spot')
        ax2.set_ylabel('BTC Spot Price (thousands USD)', color=COLORS['spot'])
        ax2.tick_params(axis='y', labelcolor=COLORS['spot'])

        # Add events
        if events:
            for event in events:
                event_date = pd.to_datetime(event['date'])
                if df['date'].min() <= event_date <= df['date'].max():
                    ax1.axvline(event_date, color='red', linestyle='--', alpha=0.4)
                    ax1.annotate(
                        event['name'][:15],
                        xy=(event_date, ax1.get_ylim()[1]),
                        rotation=45, fontsize=8, ha='left'
                    )

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate()

        fig.suptitle(f'{param_labels.get(param, param)} vs BTC Spot Price', fontsize=14, fontweight='bold')
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))
        plt.tight_layout()

        self._save_figure(fig, f"{save_name}_{param}")

        plt.close(fig)

        return fig

    # =========================================================================
    # 2. SKEW REPRODUCTION
    # =========================================================================

    def plot_skew_reproduction(
        self,
        result,  # CalibrationResult
        market_data: Dict[str, Any],
        maturities_days: Optional[List[int]] = None,
        maturity_tolerance_days: int = 5,
        n_maturities: int = 3,
        save_name: str = "skew_reproduction_snapshot",
        show: bool = False
    ) -> plt.Figure:
        """
        Plot market IV vs model IV by moneyness for different maturities.

        Parameters
        ----------
        result : CalibrationResult
            Calibration result containing residuals
        market_data : dict
            Market data used for calibration
        maturities_days : list or None
            Target maturities to plot (in days). If None, auto-detect from data.
        maturity_tolerance_days : int
            Tolerance for matching maturities
        n_maturities : int
            Number of maturities to plot when auto-detecting (default 3)
        """
        from .pricer import rBergomiPricer
        from .calibrator import ForwardVarianceCurve
        from .utils import implied_vol_batch

        # Extract data
        S0 = market_data['S']
        K_arr = market_data['K']
        tau_arr = market_data['tau']
        iv_market = market_data['iv_market']
        moneyness = market_data['moneyness']

        tau_days = tau_arr * 365

        # Auto-detect maturities if not provided
        if maturities_days is None:
            # Get unique TTMs rounded to nearest day
            unique_ttm = np.unique(np.round(tau_days))

            if len(unique_ttm) <= n_maturities:
                # Use all available maturities
                maturities_days = sorted(unique_ttm.astype(int))
            else:
                # Pick evenly spaced maturities (short, medium, long)
                indices = np.linspace(0, len(unique_ttm) - 1, n_maturities, dtype=int)
                maturities_days = sorted(unique_ttm[indices].astype(int))

            logger.info(f"Auto-detected maturities: {maturities_days} days")

        # Reconstruct xi_0 curve from result if available; fall back to scalar
        xi_param = result.xi  # scalar fallback
        if 'xi0_curve' in result.details:
            curve_data = result.details['xi0_curve']
            xi_param = ForwardVarianceCurve(
                np.array(curve_data['maturities']),
                np.array(curve_data['xi_values'])
            )

        # Create pricer with calibrated parameters and xi_0 CURVE
        pricer = rBergomiPricer(
            H=result.H, eta=result.eta, rho=result.rho, xi=xi_param,
            n_paths=50_000, n_steps=100, seed=42
        )

        # Compute model IVs — batch by maturity (same as calibration)
        n_points = len(K_arr)
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
            except Exception:
                continue

        model_ivs = implied_vol_batch(
            model_prices, S0, K_arr, tau_arr, option_type='put', r=0.0
        )

        # Create figure with subplots for each maturity
        n_mats = len(maturities_days)
        fig, axes = plt.subplots(1, n_mats, figsize=(5 * n_mats, 5))
        if n_mats == 1:
            axes = [axes]

        tol = maturity_tolerance_days

        for ax, target_days in zip(axes, maturities_days):
            # Filter for this maturity
            mask = np.abs(tau_days - target_days) <= tol

            if mask.sum() < 3:
                ax.text(0.5, 0.5, f'No data for {target_days}d',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'T = {target_days} days')
                continue

            m_slice = moneyness[mask]
            iv_mkt_slice = iv_market[mask] * 100  # to percentage
            iv_mod_slice = model_ivs[mask] * 100

            # Sort by moneyness
            sort_idx = np.argsort(m_slice)
            m_sorted = m_slice[sort_idx]
            iv_mkt_sorted = iv_mkt_slice[sort_idx]
            iv_mod_sorted = iv_mod_slice[sort_idx]

            # Plot
            ax.scatter(m_sorted, iv_mkt_sorted, c=COLORS['market'], s=40,
                      alpha=0.7, label='Market IV', marker='o')
            ax.plot(m_sorted, iv_mod_sorted, c=COLORS['model'], linewidth=2,
                   label='Model IV', marker='x', markersize=6)

            # ATM line
            ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)

            ax.set_xlabel('Moneyness (K/S)')
            ax.set_ylabel('Implied Volatility (%)')
            ax.set_title(f'T = {target_days} days')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            # Calculate RMSE for this slice
            valid = ~np.isnan(iv_mod_sorted)
            if valid.sum() > 0:
                rmse = np.sqrt(np.mean((iv_mod_sorted[valid] - iv_mkt_sorted[valid])**2))
                ax.text(0.02, 0.98, f'RMSE = {rmse:.2f}pp', transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        date_str = result.date.strftime('%Y-%m-%d') if result.date else 'N/A'
        fig.suptitle(
            f'Skew Reproduction: Market vs Model IV\n'
            f'{date_str} | H={result.H:.3f}, eta={result.eta:.2f}, '
            f'rho={result.rho:.2f}, xi={result.xi:.4f}',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()

        self._save_figure(fig, save_name)

        plt.close(fig)

        return fig

    def plot_model_vs_market_scatter(
        self,
        result,  # CalibrationResult
        save_name: str = "model_vs_market_scatter",
        show: bool = False
    ) -> plt.Figure:
        """
        Scatter plot of model IV vs market IV with 45-degree line.
        """
        if 'residuals' not in result.details:
            logger.warning("No residuals in result, cannot plot scatter")
            return None

        residuals = result.details['residuals']
        iv_market = residuals['iv_market'] * 100
        iv_model = residuals['iv_model'] * 100

        valid = ~np.isnan(iv_model)
        iv_market = iv_market[valid]
        iv_model = iv_model[valid]

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(iv_market, iv_model, c=COLORS['market'], alpha=0.6, s=30)

        # 45-degree line
        lims = [min(iv_market.min(), iv_model.min()) - 5,
                max(iv_market.max(), iv_model.max()) + 5]
        ax.plot(lims, lims, 'k--', linewidth=1, label='Perfect fit')

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Market IV (%)')
        ax.set_ylabel('Model IV (%)')
        ax.set_title(f'Model vs Market IV\nRMSE = {result.rmse:.2f}pp')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        self._save_figure(fig, save_name)

        plt.close(fig)

        return fig

    # =========================================================================
    # 3. RESIDUALS HEATMAP
    # =========================================================================

    def plot_residuals_heatmap(
        self,
        result,  # CalibrationResult
        n_moneyness_bins: int = 10,
        n_ttm_bins: int = 8,
        save_name: str = "residuals_heatmap",
        show: bool = False
    ) -> plt.Figure:
        """
        Plot heatmap of model errors by (moneyness, maturity).

        Parameters
        ----------
        result : CalibrationResult
            Must contain residuals in details
        n_moneyness_bins : int
            Number of moneyness bins
        n_ttm_bins : int
            Number of time-to-maturity bins
        """
        if 'residuals' not in result.details:
            logger.warning("No residuals in result, cannot plot heatmap")
            return None

        residuals = result.details['residuals']
        moneyness = residuals['moneyness']
        tau = residuals['tau']
        errors = residuals['error_pp']

        # Filter valid points
        valid = ~np.isnan(errors)
        moneyness = moneyness[valid]
        tau = tau[valid]
        errors = errors[valid]

        if len(errors) < 10:
            logger.warning("Too few valid points for heatmap")
            return None

        # Create bins
        m_bins = np.linspace(moneyness.min(), moneyness.max(), n_moneyness_bins + 1)
        t_bins = np.linspace(tau.min(), tau.max(), n_ttm_bins + 1)

        # Compute mean error in each bin
        heatmap_data = np.full((n_ttm_bins, n_moneyness_bins), np.nan)

        for i in range(n_ttm_bins):
            for j in range(n_moneyness_bins):
                mask = (
                    (moneyness >= m_bins[j]) & (moneyness < m_bins[j + 1]) &
                    (tau >= t_bins[i]) & (tau < t_bins[i + 1])
                )
                if mask.sum() > 0:
                    heatmap_data[i, j] = np.mean(errors[mask])

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Diverging colormap centered at 0
        vmax = np.nanmax(np.abs(heatmap_data))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.imshow(
            heatmap_data,
            cmap='RdBu_r',
            norm=norm,
            aspect='auto',
            origin='lower'
        )

        # Labels
        m_labels = [f'{(m_bins[i] + m_bins[i+1])/2:.2f}' for i in range(n_moneyness_bins)]
        t_labels = [f'{(t_bins[i] + t_bins[i+1])/2 * 365:.0f}d' for i in range(n_ttm_bins)]

        ax.set_xticks(range(n_moneyness_bins))
        ax.set_xticklabels(m_labels, rotation=45)
        ax.set_yticks(range(n_ttm_bins))
        ax.set_yticklabels(t_labels)

        ax.set_xlabel('Moneyness (K/S)')
        ax.set_ylabel('Time to Maturity')

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Model Error (pp)')

        # Title
        date_str = result.date.strftime('%Y-%m-%d') if result.date else 'N/A'
        ax.set_title(
            f'Calibration Residuals Heatmap\n{date_str} | RMSE = {result.rmse:.2f}pp',
            fontsize=12, fontweight='bold'
        )

        plt.tight_layout()

        self._save_figure(fig, save_name)

        plt.close(fig)

        return fig

    def plot_residuals_histogram(
        self,
        result,  # CalibrationResult
        save_name: str = "residuals_histogram",
        show: bool = False
    ) -> plt.Figure:
        """
        Plot histogram of residuals (should be centered at 0).
        """
        if 'residuals' not in result.details:
            logger.warning("No residuals in result")
            return None

        errors = result.details['residuals']['error_pp']
        valid = ~np.isnan(errors)
        errors = errors[valid]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(errors, bins=30, color=COLORS['market'], alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
        ax.axvline(np.mean(errors), color='green', linestyle='-', linewidth=2,
                  label=f'Mean = {np.mean(errors):.2f}pp')

        ax.set_xlabel('Model Error (pp)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Residuals Distribution\nRMSE = {result.rmse:.2f}pp, Bias = {np.mean(errors):.2f}pp')
        ax.legend()
        ax.grid(True, alpha=0.3)

        self._save_figure(fig, save_name)

        plt.close(fig)

        return fig

    # =========================================================================
    # 4. LATEX TABLES
    # =========================================================================

    def generate_calibration_table(
        self,
        results: pl.DataFrame,
        save_name: str = "calibration_results_table",
        european_format: bool = True
    ) -> str:
        """
        Generate LaTeX-ready table of calibration results.

        Parameters
        ----------
        results : pl.DataFrame
            Calibration results
        save_name : str
            Filename (without extension)
        european_format : bool
            Use comma as decimal separator

        Returns
        -------
        str
            LaTeX table code
        """
        if isinstance(results, pl.DataFrame):
            df = results.to_pandas()
        else:
            df = results

        # Select columns for table
        cols = ['date', 'spot', 'H', 'eta', 'rho', 'xi', 'rmse_pp', 'n_points']
        available_cols = [c for c in cols if c in df.columns]
        df_table = df[available_cols].copy()

        # Format date
        if 'date' in df_table.columns:
            df_table['date'] = pd.to_datetime(df_table['date'], format='mixed').dt.strftime('%Y-%m-%d')

        # Format numbers
        if european_format:
            def fmt_european(val, decimals=4):
                if pd.isna(val):
                    return '-'
                formatted = f'{val:.{decimals}f}'
                return formatted.replace('.', ',')

            for col in ['H', 'eta', 'rho', 'xi', 'rmse_pp']:
                if col in df_table.columns:
                    decimals = 4 if col in ['H', 'xi'] else 2
                    df_table[col] = df_table[col].apply(lambda x: fmt_european(x, decimals))

            if 'spot' in df_table.columns:
                df_table['spot'] = df_table['spot'].apply(lambda x: f'{x:,.0f}'.replace(',', '.'))

        # Column headers
        header_map = {
            'date': 'Date',
            'spot': r'$S_0$ (USD)',
            'H': r'$H$',
            'eta': r'$\eta$',
            'rho': r'$\rho$',
            'xi': r'$\xi$',
            'rmse_pp': 'RMSE (pp)',
            'n_points': r'$N$'
        }
        df_table.columns = [header_map.get(c, c) for c in df_table.columns]

        # Generate LaTeX
        latex_table = df_table.to_latex(
            index=False,
            escape=False,
            column_format='l' + 'c' * (len(df_table.columns) - 1)
        )

        # Add caption and label
        latex_full = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{rBergomi Calibration Results}}
\\label{{tab:calibration_results}}
{latex_table}
\\end{{table}}"""

        # Save with scheme suffix
        tagged = f"{save_name}_{self.tag}"
        output_path = self.tables_dir / f"{tagged}.tex"
        with open(output_path, 'w') as f:
            f.write(latex_full)

        # Also save CSV
        csv_path = self.tables_dir / f"{tagged}.csv"
        df_table.to_csv(csv_path, index=False)

        logger.info(f"Table saved to {output_path} and {csv_path}")

        return latex_full

    def generate_summary_statistics(
        self,
        results: pl.DataFrame,
        save_name: str = "calibration_summary"
    ) -> str:
        """
        Generate summary statistics table for calibration results.
        """
        if isinstance(results, pl.DataFrame):
            df = results.to_pandas()
        else:
            df = results

        params = ['H', 'eta', 'rho', 'xi', 'rmse_pp']
        stats_rows = []

        for param in params:
            if param not in df.columns:
                continue
            values = df[param].dropna()
            stats_rows.append({
                'Parameter': param,
                'Mean': values.mean(),
                'Std': values.std(),
                'Min': values.min(),
                'Max': values.max(),
                'Median': values.median()
            })

        stats_df = pd.DataFrame(stats_rows)

        # Save with scheme suffix
        tagged = f"{save_name}_{self.tag}"
        csv_path = self.tables_dir / f"{tagged}.csv"
        stats_df.to_csv(csv_path, index=False, float_format='%.4f')

        logger.info(f"Summary statistics saved to {csv_path}")

        return stats_df.to_string()

    # =========================================================================
    # 5. COMPARISON PLOTS
    # =========================================================================

    def plot_model_comparison(
        self,
        bs_rmse: float,
        heston_rmse: Optional[float],
        rbergomi_rmse: float,
        save_name: str = "model_comparison",
        show: bool = False
    ) -> plt.Figure:
        """
        Bar chart comparing RMSE across models (BS, Heston, rBergomi).
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        models = ['Black-Scholes']
        rmses = [bs_rmse]
        colors_list = [PALETTE['primary']]

        if heston_rmse is not None:
            models.append('Heston')
            rmses.append(heston_rmse)
            colors_list.append(PALETTE['tertiary'])

        models.append('rBergomi')
        rmses.append(rbergomi_rmse)
        colors_list.append(PALETTE['quaternary'])

        bars = ax.bar(models, rmses, color=colors_list, alpha=0.8, edgecolor='black')

        # Add value labels
        for bar, rmse in zip(bars, rmses):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                   f'{rmse:.2f}', ha='center', va='bottom', fontsize=11)

        ax.set_ylabel('RMSE (pp)')
        ax.set_title('Model Comparison: Calibration Error')
        ax.grid(True, alpha=0.3, axis='y')

        self._save_figure(fig, save_name)

        plt.close(fig)

        return fig

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _save_figure(self, fig: plt.Figure, name: str):
        """Save figure in PNG and PDF formats, with scheme suffix."""
        tagged = f"{name}_{self.tag}"
        for ext in ['png']:
            path = self.output_dir / f"{tagged}.{ext}"
            fig.savefig(path, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved: {tagged}.png")

    def create_full_report(
        self,
        timeseries_results: pl.DataFrame,
        snapshot_result,
        market_data: Dict[str, Any],
        events: Optional[List[Dict]] = None
    ):
        """
        Generate all visualizations for a complete calibration report.

        Parameters
        ----------
        timeseries_results : pl.DataFrame
            Time series calibration results
        snapshot_result : CalibrationResult
            Single date calibration result
        market_data : dict
            Market data for the snapshot date
        events : list of dict, optional
            Market events
        """
        logger.info("Generating full calibration report...")

        # 1. Parameter stability
        if timeseries_results is not None and len(timeseries_results) > 0:
            self.plot_parameter_stability(timeseries_results, events)
            for param in ['H', 'eta', 'rho', 'xi']:
                if param in timeseries_results.columns:
                    self.plot_parameter_with_spot(timeseries_results, param, events)
            self.generate_calibration_table(timeseries_results)
            self.generate_summary_statistics(timeseries_results)

        # 2. Snapshot analysis
        if snapshot_result is not None:
            self.plot_skew_reproduction(snapshot_result, market_data)
            self.plot_model_vs_market_scatter(snapshot_result)
            self.plot_residuals_heatmap(snapshot_result)
            self.plot_residuals_histogram(snapshot_result)

        logger.info("Full report generated successfully!")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_visualizer(output_dir: str = "Results/calibration/figures") -> CalibrationVisualizer:
    """Create and return a CalibrationVisualizer instance."""
    return CalibrationVisualizer(output_dir)


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("CalibrationVisualizer - Module loaded successfully")
    print(f"Output directory: Results/calibration/figures")
