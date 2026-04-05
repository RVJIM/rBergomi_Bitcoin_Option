# btc_volatility_analysis.py
"""
Bitcoin Volatility Analysis Module
==================================

Unified module for Bitcoin volatility analysis:
1. Fat Tails & Kurtosis - Heavy tail analysis of returns
2. Volatility Smile - Implied volatility smile/skew patterns
3. Term Structure - Volatility term structure

NOTE: This module should be run ONLY via main_c.py

Available commands:
    python main_c.py --vol-kurtosis    # Fat tails and kurtosis analysis
    python main_c.py --vol-smile       # Volatility smile analysis
    python main_c.py --vol-term        # Term structure analysis
    python main_c.py --vol-all         # All analyses
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import datetime
import warnings
import sys
import logging
from src.config.plot_style import apply_thesis_style, PALETTE, SURFACE_CMAP

# Setup
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')
logger = logging.getLogger("BTCVolatilityAnalysis")
apply_thesis_style()

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

CONFIG = {
    "data_dir": Path("data/option"),
    "results_dir": Path("Results"),
    "index_file": Path("data/option_index.parquet"),

    # Kurtosis analysis
    "kurtosis": {
        "output_dir": "fat_tails_kurtosis",
        "resample_freq": "1min",
        "frequencies": [('5-minute', '5min'), ('Hourly', '1h'), ('Daily', '1d'), ('Weekly', '1W')]
    },

    # Smile analysis
    "smile": {
        "output_dir": "implied_volatility_smile",
        "max_options": 1000,
        "min_volume": 2.0,
        "min_trades": 3,
        "days_to_expiry_range": (1, 60),
        "moneyness_range": (0.75, 1.35),
        "iv_bounds": (10, 250),
    },

    # Term structure analysis
    "term_structure": {
        "output_dir": "volatility_term_structure",
        "sample_size": 2000
    },

    "dpi_save": 300,
    "dpi_display": 150
}


# =============================================================================
# 1. FAT TAILS & KURTOSIS ANALYSIS
# =============================================================================

class KurtosisAnalyzer:
    """Heavy tail and kurtosis analysis of Bitcoin returns."""

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else CONFIG["data_dir"]
        self.output_dir = CONFIG["results_dir"] / CONFIG["kurtosis"]["output_dir"]
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def load_perpetual_data(self) -> pd.DataFrame:
        """Load BTC-PERPETUAL data with resampling for efficiency."""
        perpetual_files = sorted(self.data_dir.glob("BTC-PERPETUAL_*.parquet"))
        logger.info(f"Found {len(perpetual_files)} BTC-PERPETUAL files")

        resampled_dfs = []
        resample_freq = CONFIG["kurtosis"]["resample_freq"]

        for i, f in enumerate(perpetual_files):
            df = pd.read_parquet(f, columns=['timestamp', 'mark_price'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            resampled = df.set_index('datetime')['mark_price'].resample(resample_freq).last().dropna()
            resampled_dfs.append(resampled)

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(perpetual_files)} files...")

        combined = pd.concat(resampled_dfs).sort_index()
        combined = combined[~combined.index.duplicated(keep='last')]

        result = combined.reset_index()
        result.columns = ['datetime', 'mark_price']

        logger.info(f"Loaded {len(result):,} observations from {result['datetime'].min()} to {result['datetime'].max()}")
        return result

    def compute_returns(self, df: pd.DataFrame, freq: str) -> pd.Series:
        """Compute log-returns at the specified frequency."""
        prices = df.set_index('datetime')['mark_price'].resample(freq).last().dropna()
        log_returns = np.log(prices / prices.shift(1)).dropna()
        return log_returns.replace([np.inf, -np.inf], np.nan).dropna()

    def compute_statistics(self, returns: pd.Series) -> dict:
        """Compute complete distributional statistics."""
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        return {
            'count': len(returns),
            'mean': returns.mean(),
            'std': returns.std(),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns, fisher=False),
            'excess_kurtosis': stats.kurtosis(returns, fisher=True),
            'jarque_bera_pvalue': jb_pvalue
        }

    def analyze_tail_events(self, returns: pd.Series, thresholds=[2, 3, 4, 5]) -> pd.DataFrame:
        """Compare extreme event frequency vs normal distribution."""
        standardized = (returns - returns.mean()) / returns.std()
        n = len(standardized)

        results = []
        for sigma in thresholds:
            observed = ((standardized < -sigma) | (standardized > sigma)).sum()
            expected_freq = 2 * (1 - norm.cdf(sigma))
            ratio = (observed / n) / expected_freq if expected_freq > 0 else np.inf

            results.append({
                'threshold_sigma': sigma,
                'observed_count': observed,
                'observed_freq': observed / n,
                'expected_freq_normal': expected_freq,
                'ratio': ratio
            })
        return pd.DataFrame(results)

    def plot_analysis(self, returns_dict: dict):
        """Generate complete kurtosis analysis visualizations in two separate files."""
        # Usa rendimenti orari per i plot principali
        returns = list(returns_dict.values())[1]
        standardized = (returns - returns.mean()) / returns.std()

        # Fit Student's t una volta sola
        df_t, loc_t, scale_t = stats.t.fit(standardized)
        x = np.linspace(-6, 6, 1000)

        # =====================================================================
        # FIGURA 1: Distribution Analysis (3 grafici)
        # =====================================================================
        fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
        fig1.suptitle('Bitcoin Returns: Distribution Analysis',
                      fontsize=14, fontweight='bold')

        # 1. Histogram vs Normal
        ax1 = axes1[0]
        ax1.hist(standardized, bins=100, density=True, alpha=0.7, color=PALETTE['primary'],
                 edgecolor='black', linewidth=0.5, label='BTC Returns')
        ax1.plot(x, norm.pdf(x), 'r-', linewidth=2, label='Normal Distribution')
        ax1.plot(x, stats.t.pdf(x, df_t, loc_t, scale_t), 'g--', linewidth=2,
                 label=f"Student's t (df={df_t:.1f})")
        ax1.set_xlabel('Standardized Returns')
        ax1.set_ylabel('Density')
        ax1.set_title('Return Distribution vs Normal')
        ax1.legend(fontsize=8)
        ax1.set_xlim(-6, 6)
        ax1.grid(True, alpha=0.3)

        # 2. Log-scale view
        ax2 = axes1[1]
        ax2.hist(standardized, bins=100, density=True, alpha=0.7, color=PALETTE['primary'],
                 edgecolor='black', linewidth=0.5)
        ax2.plot(x, norm.pdf(x), 'r-', linewidth=2, label='Normal')
        ax2.plot(x, stats.t.pdf(x, df_t, loc_t, scale_t), 'g--', linewidth=2, label="Student's t")
        ax2.set_yscale('log')
        ax2.set_xlabel('Standardized Returns')
        ax2.set_ylabel('Log Density')
        ax2.set_title('Log-Scale View (Fat Tails Visible)')
        ax2.legend(fontsize=8)
        ax2.set_xlim(-6, 6)
        ax2.set_ylim(1e-5, 1)
        ax2.grid(True, alpha=0.3)

        # 3. QQ-Plot
        ax3 = axes1[2]
        stats.probplot(standardized, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot vs Normal Distribution')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path1 = self.figures_dir / "btc_distribution_analysis.png"
        plt.savefig(output_path1, dpi=CONFIG["dpi_save"], bbox_inches='tight')
        logger.info(f"Plot saved: {output_path1}")
        plt.close(fig1)

        # =====================================================================
        # FIGURA 2: Kurtosis Analysis (3 grafici)
        # =====================================================================
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
        fig2.suptitle('Bitcoin Returns: Kurtosis and Extreme Events Analysis',
                      fontsize=14, fontweight='bold')

        # 4. Kurtosis by frequency
        ax4 = axes2[0]
        frequencies = list(returns_dict.keys())
        kurtosis_values = [stats.kurtosis(r, fisher=False) for r in returns_dict.values()]
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(frequencies)))

        ax4.bar(range(len(frequencies)), kurtosis_values, color=colors, edgecolor='black')
        ax4.axhline(y=3, color=PALETTE['tertiary'], linestyle='--', linewidth=2, label='Normal (k=3)')
        ax4.set_xticks(range(len(frequencies)))
        ax4.set_xticklabels(frequencies, rotation=45, ha='right')
        ax4.set_ylabel('Kurtosis')
        ax4.set_title('Kurtosis by Return Frequency')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')

        # 5. Extreme event frequency
        ax5 = axes2[1]
        tail_analysis = self.analyze_tail_events(returns)
        x_pos = np.arange(len(tail_analysis))
        width = 0.35

        observed = tail_analysis['observed_freq'] * 100
        expected = tail_analysis['expected_freq_normal'] * 100

        ax5.bar(x_pos - width/2, observed, width, label='Observed (BTC)', color=PALETTE['primary'])
        ax5.bar(x_pos + width/2, expected, width, label='Expected (Normal)', color=PALETTE['tertiary'])
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([f'>{s}s' for s in tail_analysis['threshold_sigma']])
        ax5.set_ylabel('Frequency (%)')
        ax5.set_title('Extreme Event Frequency')
        ax5.legend(fontsize=8)
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3, axis='y')

        # 6. Rolling kurtosis
        ax6 = axes2[2]
        daily_returns = list(returns_dict.values())[2]
        rolling_kurt = daily_returns.rolling(window=30).apply(
            lambda x: stats.kurtosis(x, fisher=False))

        ax6.plot(rolling_kurt.index, rolling_kurt.values, color=PALETTE['primary'], linewidth=0.8)
        ax6.axhline(y=3, color=PALETTE['tertiary'], linestyle='--', linewidth=2, label='Normal (k=3)')
        ax6.fill_between(rolling_kurt.index, 3, rolling_kurt.values,
                         where=rolling_kurt.values > 3, alpha=0.3, color=PALETTE['quaternary'])
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Rolling Kurtosis')
        ax6.set_title('30-Day Rolling Kurtosis')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path2 = self.figures_dir / "btc_kurtosis_analysis.png"
        plt.savefig(output_path2, dpi=CONFIG["dpi_save"], bbox_inches='tight')
        logger.info(f"Plot saved: {output_path2}")
        plt.close(fig2)

    def print_report(self, returns_dict: dict):
        """Print complete analysis report."""
        print("\n" + "="*80)
        print("BITCOIN RETURNS KURTOSIS ANALYSIS REPORT")
        print("="*80)

        print("\n" + "-"*80)
        print("DISTRIBUTION STATISTICS BY FREQUENCY")
        print("-"*80)

        print(f"\n{'Frequency':<12} {'N':>10} {'Mean':>12} {'Std':>12} {'Skew':>10} {'Kurt':>10} {'Excess':>10}")
        print("-"*80)

        for freq, returns in returns_dict.items():
            stat = self.compute_statistics(returns)
            print(f"{freq:<12} {stat['count']:>10,} {stat['mean']*100:>11.4f}% "
                  f"{stat['std']*100:>11.4f}% {stat['skewness']:>10.2f} "
                  f"{stat['kurtosis']:>10.2f} {stat['excess_kurtosis']:>10.2f}")

        # Tail events for hourly
        returns = list(returns_dict.values())[1]
        tail_df = self.analyze_tail_events(returns)

        print("\n" + "-"*80)
        print("EXTREME EVENT ANALYSIS (Hourly returns)")
        print("-"*80)

        for _, row in tail_df.iterrows():
            print(f"\n  Beyond {row['threshold_sigma']} std dev:")
            print(f"    Observed: {row['observed_count']:,} ({row['observed_freq']*100:.4f}%)")
            print(f"    Expected: {row['expected_freq_normal']*100:.4f}%")
            print(f"    Ratio: {row['ratio']:.1f}x more frequent than normal")

        print("\n" + "="*80)

    def run(self):
        """Run the complete analysis."""
        logger.info("Starting Fat Tails & Kurtosis analysis...")

        df = self.load_perpetual_data()

        returns_dict = {}
        for label, freq in CONFIG["kurtosis"]["frequencies"]:
            returns = self.compute_returns(df, freq)
            returns_dict[label] = returns
            logger.info(f"  {label}: {len(returns):,} observations")

        self.print_report(returns_dict)
        self.plot_analysis(returns_dict)

        logger.info(f"Analysis completed. Output: {self.output_dir}")


# =============================================================================
# 2. VOLATILITY SMILE ANALYSIS
# =============================================================================

class SmileAnalyzer:
    """Volatility smile analysis of Bitcoin options."""

    def __init__(self, data_dir: str = None, index_file: str = None):
        self.data_dir = Path(data_dir) if data_dir else CONFIG["data_dir"]
        self.index_file = Path(index_file) if index_file else CONFIG["index_file"]
        self.output_dir = CONFIG["results_dir"] / CONFIG["smile"]["output_dir"]
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = CONFIG["smile"]

    def load_option_data(self, max_options: int, sample_seed: int = 42) -> pd.DataFrame:
        """Load option data from the index."""
        logger.info(f"Loading option index from: {self.index_file}")
        index_df = pd.read_parquet(self.index_file)

        index_df['expiry'] = pd.to_datetime(index_df['expiry'])
        index_df['min_date'] = pd.to_datetime(index_df['min_ts'], unit='ms')
        index_df['days_to_expiry'] = (index_df['expiry'] - index_df['min_date']).dt.total_seconds() / 86400

        # Filter to a single snapshot date for a consistent cross-section
        snap = self.cfg.get('snapshot_date')
        if snap:
            snap_date = pd.to_datetime(snap).date()
            index_df['trade_date'] = index_df['min_date'].dt.date
            index_df = index_df[index_df['trade_date'] == snap_date]
            logger.info(f"Snapshot date {snap_date}: {len(index_df)} options")

        filtered = index_df[
            (index_df['total_volume'] >= self.cfg['min_volume']) &
            (index_df['n_trades'] >= self.cfg['min_trades']) &
            (index_df['days_to_expiry'] >= self.cfg['days_to_expiry_range'][0]) &
            (index_df['days_to_expiry'] <= self.cfg['days_to_expiry_range'][1])
        ].copy()

        logger.info(f"Options after filtering: {len(filtered)}")

        if len(filtered) > max_options:
            filtered = filtered.sample(n=max_options, random_state=sample_seed)

        data_points = []
        for _, row in filtered.iterrows():
            try:
                file_path = self.data_dir / row['file']
                if not file_path.exists():
                    continue

                opt_df = pd.read_parquet(file_path)
                if len(opt_df) == 0 or 'iv' not in opt_df.columns:
                    continue

                valid = opt_df[opt_df['iv'] > 0]
                if len(valid) == 0:
                    continue

                data_points.append({
                    'iv': valid['iv'].mean(),
                    'moneyness': row['strike'] / valid['index_price'].mean(),
                    'option_type': row['option_type'],
                    'volume': row['total_volume'],
                    'strike': row['strike'],
                    'spot': valid['index_price'].mean(),
                    'days_to_expiry': row['days_to_expiry']
                })
            except Exception:
                continue

        df = pd.DataFrame(data_points)

        # Hard bounds: moneyness and IV
        m_lo, m_hi = self.cfg['moneyness_range']
        iv_lo, iv_hi = self.cfg['iv_bounds']
        df = df[(df['moneyness'] >= m_lo) & (df['moneyness'] <= m_hi) &
                (df['iv'] >= iv_lo) & (df['iv'] <= iv_hi)]

        # Remove remaining outliers with quantile trim
        q01, q99 = df['iv'].quantile([0.01, 0.99])
        df = df[(df['iv'] >= q01) & (df['iv'] <= q99)]

        logger.info(f"Loaded {len(df)} valid options")
        return df

    def calculate_statistics(self, df: pd.DataFrame) -> dict:
        """Compute volatility smile statistics."""
        otm_puts = df[(df['option_type'] == 'P') & (df['moneyness'] < 0.95)]
        atm = df[(df['moneyness'] >= 0.95) & (df['moneyness'] <= 1.05)]
        otm_calls = df[(df['option_type'] == 'C') & (df['moneyness'] > 1.05)]

        stats_dict = {
            'total': len(df),
            'calls': len(df[df['option_type'] == 'C']),
            'puts': len(df[df['option_type'] == 'P']),
            'mean_iv': df['iv'].mean(),
            'std_iv': df['iv'].std(),
            'avg_spot': df['spot'].mean(),
            'otm_puts_iv': otm_puts['iv'].mean() if len(otm_puts) > 0 else np.nan,
            'atm_iv': atm['iv'].mean() if len(atm) > 0 else np.nan,
            'otm_calls_iv': otm_calls['iv'].mean() if len(otm_calls) > 0 else np.nan,
            'otm_puts_count': len(otm_puts),
            'atm_count': len(atm),
            'otm_calls_count': len(otm_calls)
        }

        if not np.isnan(stats_dict['otm_puts_iv']) and not np.isnan(stats_dict['atm_iv']):
            stats_dict['put_premium'] = stats_dict['otm_puts_iv'] - stats_dict['atm_iv']
        else:
            stats_dict['put_premium'] = np.nan

        return stats_dict

    def plot_2d_smile(self, df: pd.DataFrame, stats: dict):
        """Generate 2D volatility smile visualization."""
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(16, 10), dpi=CONFIG["dpi_display"])
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], hspace=0.35, wspace=0.3)

        calls = df[df['option_type'] == 'C']
        puts = df[df['option_type'] == 'P']

        # Main smile plot (top row, full width)
        ax1 = fig.add_subplot(gs[0, :])

        for data, color, label in [(calls, PALETTE['tertiary'], 'Calls'), (puts, PALETTE['primary'], 'Puts')]:
            if len(data) == 0:
                continue
            data_sorted = data.sort_values('moneyness')
            bins = pd.cut(data_sorted['moneyness'], bins=20)
            grouped = data_sorted.groupby(bins, observed=True).agg({
                'iv': 'mean', 'moneyness': 'mean', 'volume': 'sum'
            }).dropna()

            if len(grouped) > 0:
                ax1.plot(grouped['moneyness'], grouped['iv'], color=color,
                        linewidth=3, alpha=0.8, label=label)
                sizes = np.sqrt(grouped['volume']) * 20
                ax1.scatter(grouped['moneyness'], grouped['iv'], s=sizes,
                           color=color, alpha=0.6, edgecolors='black')

        ax1.axvline(x=1.0, color=PALETTE['neutral'], linestyle='--', linewidth=2.5, alpha=0.6, label='ATM')
        ax1.set_xlabel('Moneyness (Strike / Spot)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Implied Volatility (%)', fontsize=13, fontweight='bold')
        ax1.set_title('Bitcoin Options Volatility Smile', fontsize=15, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(df['moneyness'].quantile(0.02), df['moneyness'].quantile(0.98))

        # By strike (bottom row, 2/3 width)
        ax2 = fig.add_subplot(gs[1, :2])
        for data, color, label in [(calls, PALETTE['tertiary'], 'Calls'), (puts, PALETTE['primary'], 'Puts')]:
            if len(data) == 0:
                continue
            grouped = data.groupby('strike').agg({'iv': 'mean', 'volume': 'sum'}).reset_index()
            if len(grouped) > 2:
                ax2.plot(grouped['strike'], grouped['iv'], color=color, linewidth=2.5, label=label)
                ax2.scatter(grouped['strike'], grouped['iv'], s=np.sqrt(grouped['volume'])*20,
                           color=color, alpha=0.6, edgecolors='black')

        ax2.axvline(x=stats['avg_spot'], color=PALETTE['neutral'], linestyle='--', linewidth=2)
        ax2.set_xlabel('Strike Price ($)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Implied Volatility (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Volatility by Strike Price', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Statistics (bottom row, 1/3 width)
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.axis('off')

        stats_text = f"""
VOLATILITY SMILE STATISTICS

Sample Size: {stats['total']} options
Calls: {stats['calls']}  |  Puts: {stats['puts']}
Mean IV: {stats['mean_iv']:.2f}%

By Moneyness:
  OTM Puts (K/S < 0.95): {stats['otm_puts_iv']:.2f}%
  ATM (0.95-1.05): {stats['atm_iv']:.2f}%
  OTM Calls (K/S > 1.05): {stats['otm_calls_iv']:.2f}%

Smile Evidence:
  OTM Put Premium: +{stats['put_premium']:.2f} pp

Spot Price: ${stats['avg_spot']:,.2f}
"""
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()
        output_path = self.figures_dir / "volatility_smile_2d.png"
        fig.savefig(output_path, dpi=CONFIG["dpi_save"], bbox_inches='tight', facecolor='white')
        logger.info(f"Plot saved: {output_path}")
        plt.close()

    def plot_3d_surface(self, df: pd.DataFrame):
        """Generate 3D volatility surface visualization."""
        fig = plt.figure(figsize=(14, 10), dpi=CONFIG["dpi_display"])
        ax = fig.add_subplot(111, projection='3d')

        moneyness_range = np.linspace(df['moneyness'].min(), df['moneyness'].max(), 80)
        days_range = np.linspace(df['days_to_expiry'].min(), df['days_to_expiry'].max(), 80)
        moneyness_grid, days_grid = np.meshgrid(moneyness_range, days_range)

        points = df[['moneyness', 'days_to_expiry']].values
        values = df['iv'].values
        iv_grid = griddata(points, values, (moneyness_grid, days_grid), method='linear')

        # Clip interpolated values to data range as safety net
        iv_floor, iv_ceil = df['iv'].quantile(0.02), df['iv'].quantile(0.98)
        iv_grid = np.clip(iv_grid, iv_floor, iv_ceil)

        from matplotlib.colors import Normalize
        norm = Normalize(vmin=iv_floor, vmax=iv_ceil)
        surf = ax.plot_surface(moneyness_grid, days_grid, iv_grid, cmap='plasma',
                               norm=norm, alpha=0.95, antialiased=True, edgecolor='none')

        ax.set_xlabel('Moneyness (K/S)', fontsize=12, labelpad=10, fontweight='bold')
        ax.set_ylabel('Days to Expiry', fontsize=12, labelpad=10, fontweight='bold')
        ax.set_zlabel('Implied Volatility (%)', fontsize=12, labelpad=10, fontweight='bold')
        ax.set_title('Bitcoin Options Volatility Surface', fontsize=14, fontweight='bold')
        ax.set_zlim(iv_floor, iv_ceil)

        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
        cbar.set_label('Implied Volatility (%)', fontsize=11, fontweight='bold')

        ax.view_init(elev=25, azim=45)

        plt.tight_layout()
        output_path = self.figures_dir / "volatility_surface_3d.png"
        fig.savefig(output_path, dpi=CONFIG["dpi_save"], bbox_inches='tight', facecolor='white')
        logger.info(f"Plot saved: {output_path}")
        plt.close()

        # Interactive HTML version (rotatable)
        import plotly.graph_objects as go

        fig_html = go.Figure(data=[go.Surface(
            x=moneyness_grid, y=days_grid, z=iv_grid,
            colorscale='Plasma', cmin=iv_floor, cmax=iv_ceil,
            colorbar=dict(title='IV (%)'),
        )])
        fig_html.update_layout(
            title='Bitcoin Options Volatility Surface on Deribit',
            scene=dict(
                xaxis_title='Moneyness (K/S)',
                yaxis_title='Days to Expiry',
                zaxis_title='Implied Volatility (%)',
                zaxis=dict(range=[iv_floor, iv_ceil]),
            ),
            width=900, height=700,
        )
        html_path = self.figures_dir / "volatility_surface_3d.html"
        fig_html.write_html(str(html_path))
        logger.info(f"Interactive plot saved: {html_path}")

    def save_summary(self, stats: dict):
        """Save text report."""
        report = f"""
BITCOIN OPTIONS VOLATILITY SMILE ANALYSIS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

DATA OVERVIEW
Total Options: {stats['total']}
  - Calls: {stats['calls']}
  - Puts: {stats['puts']}

Average Spot: ${stats['avg_spot']:,.2f}

VOLATILITY BY MONEYNESS
  OTM Puts (K/S < 0.95): {stats['otm_puts_iv']:.2f}% ({stats['otm_puts_count']} options)
  ATM (0.95-1.05): {stats['atm_iv']:.2f}% ({stats['atm_count']} options)
  OTM Calls (K/S > 1.05): {stats['otm_calls_iv']:.2f}% ({stats['otm_calls_count']} options)

SMILE EVIDENCE
  OTM Put Premium: +{stats['put_premium']:.2f} percentage points

This confirms OTM puts trade at significantly higher IV than ATM,
reflecting demand for downside protection that Black-Scholes cannot capture.
"""
        output_path = self.tables_dir / "analysis_summary.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Report saved: {output_path}")

    def run(self):
        """Run the complete analysis."""
        logger.info("Starting Volatility Smile analysis...")

        # Single data load for both 2D and 3D
        df = self.load_option_data(max_options=self.cfg['max_options'])
        stats = self.calculate_statistics(df)
        self.plot_2d_smile(df, stats)
        self.plot_3d_surface(df)
        self.save_summary(stats)

        logger.info(f"Analysis completed. Output: {self.output_dir}")


# =============================================================================
# 3. TERM STRUCTURE ANALYSIS
# =============================================================================

class TermStructureAnalyzer:
    """Volatility term structure analysis."""

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else CONFIG["data_dir"]
        self.output_dir = CONFIG["results_dir"] / CONFIG["term_structure"]["output_dir"]
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def parse_instrument(self, name: str) -> dict:
        """Parse Deribit instrument name."""
        parts = name.split('-')
        if len(parts) != 4:
            return None

        try:
            expiry_date = datetime.strptime(parts[1], "%d%b%y")
        except ValueError:
            try:
                expiry_date = datetime.strptime(parts[1], "%d%b%Y")
            except ValueError:
                return None

        return {
            'expiry_date': expiry_date,
            'strike': float(parts[2]),
            'option_type': parts[3]
        }

    def load_option_data(self, sample_size: int = None) -> pd.DataFrame:
        """Load option data (excludes PERPETUAL)."""
        sample_size = sample_size or CONFIG["term_structure"]["sample_size"]
        option_files = [f for f in self.data_dir.glob("BTC-*.parquet")
                        if "PERPETUAL" not in f.name]

        logger.info(f"Found {len(option_files)} option files")

        if sample_size < len(option_files):
            option_files = sorted(option_files)
            indices = np.linspace(0, len(option_files)-1, sample_size, dtype=int)
            option_files = [option_files[i] for i in indices]

        all_data = []
        for i, f in enumerate(option_files):
            try:
                df = pd.read_parquet(f, columns=['timestamp', 'iv', 'instrument_name',
                                                  'index_price', 'expiration_timestamp'])
                if len(df) == 0 or df['iv'].isna().all():
                    continue

                info = self.parse_instrument(df['instrument_name'].iloc[0])
                if info is None:
                    continue

                df['strike'] = info['strike']
                df['option_type'] = info['option_type']
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['expiry_datetime'] = pd.to_datetime(df['expiration_timestamp'], unit='ms')
                df['ttm_days'] = (df['expiry_datetime'] - df['datetime']).dt.total_seconds() / 86400
                df['moneyness'] = df['strike'] / df['index_price']
                df['log_moneyness'] = np.log(df['moneyness'])

                all_data.append(df)

                if (i + 1) % 500 == 0:
                    logger.info(f"  Processed {i + 1} files...")
            except Exception:
                continue

        combined = pd.concat(all_data, ignore_index=True)
        combined = combined[(combined['iv'] > 0) & (combined['ttm_days'] > 0) &
                           (combined['ttm_days'] < 365)]

        logger.info(f"Loaded {len(combined):,} observations")
        return combined

    def categorize_maturity(self, ttm: float) -> str:
        """Categorize TTM into buckets."""
        if ttm <= 7:
            return "1. Short (0-7d)"
        elif ttm <= 30:
            return "2. Medium (7-30d)"
        elif ttm <= 90:
            return "3. Long (30-90d)"
        else:
            return "4. Very Long (90d+)"

    def compute_smile_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute smile metrics per maturity bucket."""
        df = df.copy()
        df['maturity_bucket'] = df['ttm_days'].apply(self.categorize_maturity)

        metrics = []
        for bucket in sorted(df['maturity_bucket'].unique()):
            bucket_data = df[(df['maturity_bucket'] == bucket) &
                            (df['moneyness'] >= 0.8) & (df['moneyness'] <= 1.2)]

            if len(bucket_data) < 10:
                continue

            atm = bucket_data[(bucket_data['moneyness'] >= 0.98) &
                             (bucket_data['moneyness'] <= 1.02)]
            otm_put = bucket_data[(bucket_data['moneyness'] >= 0.85) &
                                  (bucket_data['moneyness'] <= 0.95) &
                                  (bucket_data['option_type'] == 'P')]
            otm_call = bucket_data[(bucket_data['moneyness'] >= 1.05) &
                                   (bucket_data['moneyness'] <= 1.15) &
                                   (bucket_data['option_type'] == 'C')]

            atm_iv = atm['iv'].median() if len(atm) > 0 else np.nan
            otm_put_iv = otm_put['iv'].median() if len(otm_put) > 0 else np.nan
            otm_call_iv = otm_call['iv'].median() if len(otm_call) > 0 else np.nan

            skew = otm_put_iv - otm_call_iv if not (np.isnan(otm_put_iv) or np.isnan(otm_call_iv)) else np.nan
            curvature = ((otm_put_iv + otm_call_iv) / 2) - atm_iv if not any(np.isnan([atm_iv, otm_put_iv, otm_call_iv])) else np.nan

            try:
                slope, _, _, _, _ = stats.linregress(bucket_data['log_moneyness'], bucket_data['iv'])
            except:
                slope = np.nan

            metrics.append({
                'maturity_bucket': bucket,
                'n_observations': len(bucket_data),
                'avg_ttm_days': bucket_data['ttm_days'].mean(),
                'atm_iv': atm_iv,
                'skew': skew,
                'smile_curvature': curvature,
                'skew_coefficient': slope
            })

        return pd.DataFrame(metrics)

    def plot_analysis(self, df: pd.DataFrame, metrics: pd.DataFrame):
        """Generate term structure visualizations."""
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Bitcoin Options: Volatility Term Structure Analysis',
                     fontsize=14, fontweight='bold')

        colors = {'1. Short (0-7d)': PALETTE['tertiary'], '2. Medium (7-30d)': PALETTE['primary'],
                  '3. Long (30-90d)': PALETTE['quaternary'], '4. Very Long (90d+)': PALETTE['secondary']}

        df_filtered = df[(df['moneyness'] >= 0.8) & (df['moneyness'] <= 1.2)].copy()
        df_filtered['maturity_bucket'] = df_filtered['ttm_days'].apply(self.categorize_maturity)

        # 1. Smiles by maturity
        ax1 = fig.add_subplot(2, 2, 1)
        for bucket in sorted(df_filtered['maturity_bucket'].unique()):
            bucket_data = df_filtered[df_filtered['maturity_bucket'] == bucket].copy()
            bucket_data['moneyness_bin'] = pd.cut(bucket_data['moneyness'], bins=20)
            smile = bucket_data.groupby('moneyness_bin', observed=True).agg({
                'iv': 'median', 'moneyness': 'mean'
            }).dropna()

            if len(smile) > 3:
                ax1.plot(smile['moneyness'], smile['iv'], 'o-', color=colors.get(bucket, 'gray'),
                        label=bucket, linewidth=2, markersize=4, alpha=0.8)

        ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Moneyness (K/S)')
        ax1.set_ylabel('Implied Volatility (%)')
        ax1.set_title('Volatility Smile by Maturity')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. Skew coefficient by maturity
        ax2 = fig.add_subplot(2, 2, 2)
        valid = metrics.dropna(subset=['skew_coefficient'])
        if len(valid) > 0:
            x_pos = range(len(valid))
            ax2.bar(x_pos, valid['skew_coefficient'].abs(),
                   color=[colors.get(b, 'gray') for b in valid['maturity_bucket']])
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([b.split('(')[1].split(')')[0] for b in valid['maturity_bucket']],
                               rotation=45, ha='right')
            ax2.set_ylabel('|Skew Coefficient|')
            ax2.set_title('Smile Steepness by Maturity')
            ax2.grid(True, alpha=0.3, axis='y')

        # 3. Skew by maturity
        ax3 = fig.add_subplot(2, 2, 3)
        valid = metrics.dropna(subset=['skew'])
        if len(valid) > 0:
            x_pos = range(len(valid))
            ax3.bar(x_pos, valid['skew'],
                   color=[colors.get(b, 'gray') for b in valid['maturity_bucket']])
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([b.split('(')[1].split(')')[0] for b in valid['maturity_bucket']],
                               rotation=45, ha='right')
            ax3.set_ylabel('OTM Put IV - OTM Call IV (%)')
            ax3.set_title('Volatility Skew by Maturity')
            ax3.grid(True, alpha=0.3, axis='y')

        # 4. Summary table
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')

        summary = "TERM STRUCTURE SUMMARY\n" + "="*30 + "\n\n"
        for _, row in metrics.iterrows():
            bucket_short = row['maturity_bucket'].split('(')[1].split(')')[0]
            summary += f"{bucket_short}:\n"
            summary += f"  ATM IV: {row['atm_iv']:.1f}%\n" if not np.isnan(row['atm_iv']) else "  ATM IV: N/A\n"
            summary += f"  Skew: {row['skew']:.1f}%\n" if not np.isnan(row['skew']) else "  Skew: N/A\n"
            summary += f"  N obs: {row['n_observations']:,}\n\n"

        ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        plt.tight_layout()
        output_path = self.figures_dir / "btc_volatility_term_structure.png"
        plt.savefig(output_path, dpi=CONFIG["dpi_save"], bbox_inches='tight')
        logger.info(f"Plot saved: {output_path}")
        plt.close()

    def print_report(self, metrics: pd.DataFrame):
        """Print term structure report."""
        print("\n" + "="*80)
        print("BITCOIN OPTIONS: VOLATILITY TERM STRUCTURE ANALYSIS")
        print("="*80)

        print("\n" + "-"*80)
        print("SMILE METRICS BY MATURITY")
        print("-"*80)

        print(f"\n{'Maturity':<20} {'N Obs':>10} {'ATM IV':>10} {'Skew':>10} {'Skew Coef':>12}")
        print("-"*70)

        for _, row in metrics.iterrows():
            bucket = row['maturity_bucket'].split('.')[1].strip()
            atm = f"{row['atm_iv']:.1f}%" if not np.isnan(row['atm_iv']) else "N/A"
            skew = f"{row['skew']:.1f}%" if not np.isnan(row['skew']) else "N/A"
            coef = f"{row['skew_coefficient']:.2f}" if not np.isnan(row['skew_coefficient']) else "N/A"
            print(f"{bucket:<20} {row['n_observations']:>10,} {atm:>10} {skew:>10} {coef:>12}")

        print("\n" + "="*80)

    def run(self):
        """Run the complete analysis."""
        logger.info("Starting Term Structure analysis...")

        df = self.load_option_data()
        metrics = self.compute_smile_metrics(df)

        self.print_report(metrics)
        self.plot_analysis(df, metrics)

        logger.info(f"Analysis completed. Output: {self.output_dir}")


# =============================================================================
# 4. FRACTIONAL BROWNIAN MOTION SAMPLE PATHS
# =============================================================================

def plot_fbm_sample_paths():
    """Generate fBm sample paths for H=0.5 and H=0.1."""
    from scipy.linalg import cholesky

    output_dir = CONFIG["results_dir"] / "fbm_paths" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_steps = 500
    T = 1.0
    times = np.linspace(0, T, n_steps + 1)
    seed = 42

    def fbm_covariance(times, H):
        n = len(times)
        two_H = 2.0 * H
        cov = np.empty((n, n))
        for i in range(n):
            for j in range(i + 1):
                cov[i, j] = 0.5 * (times[i]**two_H + times[j]**two_H
                                    - abs(times[i] - times[j])**two_H)
                cov[j, i] = cov[i, j]
        return cov

    def generate_fbm(times, H, seed):
        # Skip t=0 for Cholesky (cov is singular at 0)
        cov = fbm_covariance(times[1:], H)
        # Add small regularization for numerical stability
        cov += np.eye(len(cov)) * 1e-10
        L = cholesky(cov, lower=True)
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(len(cov))
        path = L @ z
        return np.concatenate([[0.0], path])

    # Generate paths with same seed
    path_05 = generate_fbm(times, H=0.5, seed=seed)
    path_01 = generate_fbm(times, H=0.1, seed=seed)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Sample Paths of Fractional Brownian Motion',
                 fontsize=14, fontweight='bold')

    fbm_colors = [PALETTE['primary'], PALETTE['tertiary']]
    for idx, (ax, path, H_val) in enumerate([(axes[0], path_05, 0.5), (axes[1], path_01, 0.1)]):
        ax.plot(times, path, color=fbm_colors[idx], linewidth=0.8)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('$B^H_t$', fontsize=12)
        ax.set_title(f'$H = {H_val}$', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    output_path = output_dir / "fbm_sample_paths.png"
    fig.savefig(output_path, dpi=CONFIG["dpi_save"], bbox_inches='tight', facecolor='white')
    logger.info(f"Plot saved: {output_path}")
    plt.close()


# =============================================================================
# PUBLIC FUNCTIONS (called from main_c.py)
# =============================================================================

def run_kurtosis_analysis():
    """Run fat tails and kurtosis analysis."""
    analyzer = KurtosisAnalyzer()
    analyzer.run()

def run_smile_analysis():
    """Run volatility smile analysis."""
    analyzer = SmileAnalyzer()
    analyzer.run()

def run_term_structure_analysis():
    """Run term structure analysis."""
    analyzer = TermStructureAnalyzer()
    analyzer.run()

def run_fbm_paths():
    """Generate fBm sample paths plot."""
    plot_fbm_sample_paths()

def run_all_volatility_analysis():
    """Run all volatility analyses."""
    logger.info("="*60)
    logger.info("FULL BITCOIN VOLATILITY ANALYSIS")
    logger.info("="*60)

    run_kurtosis_analysis()
    run_smile_analysis()
    run_term_structure_analysis()
    run_fbm_paths()

    logger.info("="*60)
    logger.info("ALL ANALYSES COMPLETED")
    logger.info("="*60)

    # Output summary
    print("\nGenerated output:")
    for subdir in ["fat_tails_kurtosis", "implied_volatility_smile", "volatility_term_structure", "fbm_paths"]:
        fig_path = CONFIG["results_dir"] / subdir / "figures"
        tbl_path = CONFIG["results_dir"] / subdir / "tables"
        if fig_path.exists():
            figs = list(fig_path.glob("*"))
            print(f"  {subdir}/figures/: {len(figs)} files")
        if tbl_path.exists():
            tbls = list(tbl_path.glob("*"))
            if tbls:
                print(f"  {subdir}/tables/: {len(tbls)} files")

# NOTE: This module should be run only via main_c.py
# Example: python main_c.py --vol-kurtosis
