# iv_visualizer.py
"""
IV Surface and Calibrated Parameter Visualization.
Dynamic (animations) and static (events) plots for the thesis.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

from src.config.plot_style import apply_thesis_style, PALETTE, SURFACE_CMAP
from scipy.stats import norm
from src.rbergomi.utils import (
    bs_d1, bs_d2, bs_price_call_usd, bs_price_put_usd,
)

apply_thesis_style()


class IVSurfaceVisualizer:
    """
    Generate visualizations for IV surfaces and calibrated parameters.
    """
    
    def __init__(self, output_dir: str = "figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Consistent colors for the thesis (plasma-derived)
        self.colors = {
            'call': PALETTE['primary'],
            'put': PALETTE['tertiary'],
            'atm': PALETTE['quaternary'],
            'spot': PALETTE['secondary'],
            'grid': '#E8E8E8'
        }
    
    def plot_iv_surface_3d(
        self,
        surface: pl.DataFrame,
        title: str = "Implied Volatility Surface",
        save_name: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        3D plot of the implied volatility surface.
        """
        if surface.height < 10:
            print("Insufficient data for 3D surface")
            return None
        
        # Extract data
        K = surface["strike"].to_numpy()
        tau = surface["ttm"].to_numpy()
        iv = surface["iv"].to_numpy()
        spot = surface["spot"].mean()
        
        # Create grid for interpolation
        K_grid = np.linspace(K.min(), K.max(), 50)
        tau_grid = np.linspace(tau.min(), tau.max(), 30)
        K_mesh, tau_mesh = np.meshgrid(K_grid, tau_grid)
        
        # Interpolate IV on the grid
        iv_mesh = griddata((K, tau), iv, (K_mesh, tau_mesh), method='cubic')
        
        # Plot 3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Surface
        surf = ax.plot_surface(
            K_mesh / 1000,  # In thousands for readability
            tau_mesh * 365,  # In days
            iv_mesh,
            cmap=SURFACE_CMAP,
            alpha=0.8,
            edgecolor='none'
        )
        
        # Actual data points
        ax.scatter(
            K / 1000, tau * 365, iv,
            c='red', s=10, alpha=0.6, label='Market data'
        )
        
        # ATM line
        atm_mask = np.abs(K / spot - 1) < 0.05
        if atm_mask.sum() > 0:
            ax.scatter(
                K[atm_mask] / 1000,
                tau[atm_mask] * 365,
                iv[atm_mask],
                c=self.colors['atm'], s=30, marker='D', label='ATM'
            )
        
        ax.set_xlabel('Strike (thousands USD)')
        ax.set_ylabel('Time to Maturity (days)')
        ax.set_zlabel('Implied Volatility (%)')
        ax.set_title(f'{title}\nSpot: ${spot:,.0f}')
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='IV (%)')
        ax.legend()
        
        if save_name:
            fig.savefig(self.output_dir / f"{save_name}.png")
        
        plt.close(fig)
        return fig
    
    def plot_iv_smile(
        self,
        surface: pl.DataFrame,
        ttm_target: float = 0.0833,  # ~30 days
        ttm_tolerance: float = 0.02,
        title: str = "Volatility Smile",
        save_name: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        2D volatility smile plot for a specific maturity.
        """
        # Filter for target TTM
        slice_data = surface.filter(
            (pl.col("ttm") >= ttm_target - ttm_tolerance) &
            (pl.col("ttm") <= ttm_target + ttm_tolerance)
        ).sort("moneyness")
        
        if slice_data.height < 5:
            print(f"Insufficient data for TTM ~ {ttm_target:.3f}")
            return None
        
        moneyness = slice_data["moneyness"].to_numpy()
        iv = slice_data["iv"].to_numpy()
        actual_ttm = slice_data["ttm"].mean()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(moneyness, iv, c=self.colors['call'], s=50, alpha=0.7, label='Market IV')
        ax.plot(moneyness, iv, c=self.colors['call'], alpha=0.5, linewidth=1)
        
        # ATM line
        ax.axvline(x=1.0, color=self.colors['atm'], linestyle='--', 
                   alpha=0.7, label='ATM (K/S = 1)')
        
        ax.set_xlabel('Moneyness (K/S)')
        ax.set_ylabel('Implied Volatility (%)')
        ax.set_title(f'{title}\nTTM ~ {actual_ttm*365:.0f} days')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_name:
            fig.savefig(self.output_dir / f"{save_name}.png")
        
        plt.close(fig)
        return fig
    
    def plot_term_structure(
        self,
        surface: pl.DataFrame,
        moneyness_target: float = 1.0,  # ATM
        moneyness_tolerance: float = 0.05,
        title: str = "ATM Term Structure",
        save_name: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        ATM term structure plot.
        """
        # Filter for target moneyness
        atm_data = surface.filter(
            (pl.col("moneyness") >= moneyness_target - moneyness_tolerance) &
            (pl.col("moneyness") <= moneyness_target + moneyness_tolerance)
        ).sort("ttm")
        
        if atm_data.height < 3:
            print("Insufficient ATM data")
            return None
        
        ttm_days = atm_data["ttm"].to_numpy() * 365
        iv = atm_data["iv"].to_numpy()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(ttm_days, iv, c=self.colors['put'], s=50, alpha=0.7, label='ATM IV')
        ax.plot(ttm_days, iv, c=self.colors['put'], alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Time to Maturity (days)')
        ax.set_ylabel('Implied Volatility (%)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_name:
            fig.savefig(self.output_dir / f"{save_name}.png")
        
        plt.close(fig)
        return fig
    
    def plot_calibration_timeseries(
        self,
        results: pl.DataFrame,
        param: str = "iv_atm_mean",
        title: str = "Calibrated Parameters Over Time",
        save_name: Optional[str] = None,
        show: bool = True,
        events: Optional[List[Dict]] = None
    ) -> plt.Figure:
        """
        Time series of calibrated parameters with highlighted events.
        """
        dates = results["date"].to_numpy()
        values = results[param].to_numpy()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(dates, values, c=self.colors['call'], linewidth=1.5, alpha=0.8)
        ax.fill_between(dates, values, alpha=0.2, color=self.colors['call'])
        
        # Add events
        if events:
            for event in events:
                event_date = event["date"]
                if dates.min() <= np.datetime64(event_date) <= dates.max():
                    ax.axvline(
                        x=event_date, 
                        color=self.colors['spot'], 
                        linestyle='--', 
                        alpha=0.7,
                        linewidth=1
                    )
                    # Event label (rotated)
                    ax.annotate(
                        event["name"][:20],  # Truncated
                        xy=(event_date, values.max()),
                        xytext=(5, 0),
                        textcoords='offset points',
                        fontsize=8,
                        rotation=45,
                        ha='left'
                    )
        
        ax.set_xlabel('Date')
        ax.set_ylabel(param.replace('_', ' ').title())
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        fig.autofmt_xdate()
        
        if save_name:
            fig.savefig(self.output_dir / f"{save_name}.png")
        
        plt.close(fig)
        return fig
    
    def plot_event_comparison(
        self,
        event_data: pl.DataFrame,
        event_name: str,
        save_name: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Pre/during/post event comparison for a single event.
        """
        event_rows = event_data.filter(pl.col("event_name") == event_name)
        
        if event_rows.height == 0:
            print(f"No data for event: {event_name}")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        labels = ["pre", "event", "post"]
        titles = ["Day Before", "Event Day", "Day After"]
        
        for idx, (label, title) in enumerate(zip(labels, titles)):
            ax = axes[idx]
            row = event_rows.filter(pl.col("observation") == label)
            
            if row.height == 0:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                ax.set_title(title)
                continue
            
            row = row.row(0, named=True)
            strikes = np.array(row["strikes"])
            ivs = np.array(row["ivs"]) * 100  # In percentage
            spot = row["spot"]
            
            moneyness = strikes / spot
            
            ax.scatter(moneyness, ivs, c=self.colors['call'], s=30, alpha=0.7)
            ax.plot(moneyness, ivs, c=self.colors['call'], alpha=0.5)
            ax.axvline(x=1.0, color=self.colors['atm'], linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Moneyness (K/S)')
            ax.set_ylabel('IV (%)')
            ax.set_title(f'{title}\nSpot: ${spot:,.0f}, IV mean: {row["iv_mean"]*100:.1f}%')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'{event_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.output_dir / f"{save_name}.png")
        
        plt.close(fig)
        return fig
    
    def create_surface_animation(
        self,
        builder,  # IVSurfaceBuilder instance
        start_date: datetime,
        end_date: datetime,
        interval_days: int = 7,
        fps: int = 2,
        save_name: str = "iv_surface_evolution",
        show_scatter: bool = True
    ) -> str:
        """
        Create GIF animation of IV surface evolution.
        """
        from iv_surface_builder import IVSurfaceBuilder
        
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=interval_days)
        
        if len(dates) < 2:
            print("Need at least 2 dates for animation")
            return None
        
        print(f"Creating animation for {len(dates)} dates...")
        
        # Pre-compute all surfaces
        surfaces = []
        valid_dates = []
        
        for date in dates:
            surface = builder.get_iv_surface(
                target_time=date,
                window_hours=4.0,
                min_volume=0.05,
                moneyness_range=(0.7, 1.4),
                min_ttm_days=7,
                max_ttm_days=90
            )
            if surface.height >= 15:
                surfaces.append(surface)
                valid_dates.append(date)
        
        if len(surfaces) < 2:
            print("Insufficient valid surfaces for animation")
            return None
        
        # Find global ranges for consistent axes
        all_K = np.concatenate([s["strike"].to_numpy() for s in surfaces])
        all_tau = np.concatenate([s["ttm"].to_numpy() for s in surfaces])
        all_iv = np.concatenate([s["iv"].to_numpy() for s in surfaces])
        
        K_range = (all_K.min(), all_K.max())
        tau_range = (all_tau.min(), all_tau.max())
        iv_range = (all_iv.min() * 0.9, all_iv.max() * 1.1)
        
        # Setup figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            
            surface = surfaces[frame]
            date = valid_dates[frame]
            
            K = surface["strike"].to_numpy()
            tau = surface["ttm"].to_numpy()
            iv = surface["iv"].to_numpy()
            spot = surface["spot"].mean()
            
            # Grid for interpolation
            K_grid = np.linspace(K_range[0], K_range[1], 40)
            tau_grid = np.linspace(tau_range[0], tau_range[1], 25)
            K_mesh, tau_mesh = np.meshgrid(K_grid, tau_grid)
            
            try:
                iv_mesh = griddata((K, tau), iv, (K_mesh, tau_mesh), method='linear')
                
                ax.plot_surface(
                    K_mesh / 1000, tau_mesh * 365, iv_mesh,
                    cmap=SURFACE_CMAP, alpha=0.7, edgecolor='none'
                )
            except Exception:
                pass
            
            if show_scatter:
                ax.scatter(K / 1000, tau * 365, iv, c='red', s=8, alpha=0.5)
            
            ax.set_xlim(K_range[0] / 1000, K_range[1] / 1000)
            ax.set_ylim(tau_range[0] * 365, tau_range[1] * 365)
            ax.set_zlim(iv_range[0], iv_range[1])
            
            ax.set_xlabel('Strike (K USD)')
            ax.set_ylabel('TTM (days)')
            ax.set_zlabel('IV (%)')
            ax.set_title(f'IV Surface Evolution\n{date.strftime("%Y-%m-%d")} | Spot: ${spot:,.0f}')
            
            return ax,
        
        anim = FuncAnimation(fig, update, frames=len(surfaces), interval=1000//fps, blit=False)
        
        # Save as GIF
        output_path = self.output_dir / f"{save_name}.gif"
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        
        plt.close(fig)
        print(f"Animation saved to {output_path}")
        
        return str(output_path)


class DeltaComparisonPlotter:
    """
    Generate Delta comparison plots between inverse and traditional options.
    Required for the thesis.
    """
    
    def __init__(self, output_dir: str = "figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def _bs_put_price(S, K, T, r, sigma):
        """Traditional Black-Scholes put price."""
        return bs_price_put_usd(S, K, T, r, sigma)

    @staticmethod
    def _bs_call_price(S, K, T, r, sigma):
        """Traditional Black-Scholes call price."""
        return bs_price_call_usd(S, K, T, r, sigma)

    @staticmethod
    def _delta_traditional_put(S, K, T, r, sigma):
        """Traditional put delta."""
        return norm.cdf(bs_d1(S, K, T, r, sigma)) - 1

    @staticmethod
    def _delta_traditional_call(S, K, T, r, sigma):
        """Traditional call delta."""
        return norm.cdf(bs_d1(S, K, T, r, sigma))

    def _delta_inverse_put(self, S, K, T, r, sigma):
        """
        Inverse put delta (in BTC).
        d(V_put/S)/dS = (1/S) * [N(-d2) - V_put/S]
        """
        d2_val = bs_d2(S, K, T, r, sigma)
        V_put = self._bs_put_price(S, K, T, r, sigma)
        return (1/S) * (norm.cdf(-d2_val) - V_put/S)

    def _delta_inverse_call(self, S, K, T, r, sigma):
        """
        Inverse call delta (in BTC).
        d(V_call/S)/dS = (1/S) * [N(d1) - V_call/S]
        """
        d1_val = bs_d1(S, K, T, r, sigma)
        V_call = self._bs_call_price(S, K, T, r, sigma)
        return (1/S) * (norm.cdf(d1_val) - V_call/S)
    
    def plot_delta_comparison(
        self,
        K: float = 100_000,
        T: float = 30/365,
        r: float = 0.05,
        sigma: float = 0.60,
        S_range: Tuple[float, float] = (60_000, 140_000),
        option_type: str = "put",
        save_name: str = "delta_comparison",
        show: bool = True
    ) -> plt.Figure:
        """
        Generate the Delta comparison plot required for the thesis.
        
        "Illustration of the delta of an inverse put option versus a traditional 
        put option as a function of the underlying price. Should highlight 
        divergence as price moves away from strike."
        """
        S_array = np.linspace(S_range[0], S_range[1], 500)
        
        if option_type.lower() == "put":
            delta_trad = np.array([self._delta_traditional_put(S, K, T, r, sigma) for S in S_array])
            delta_inv = np.array([self._delta_inverse_put(S, K, T, r, sigma) for S in S_array])
            title = "Delta Comparison: Inverse vs Traditional Put Option"
        else:
            delta_trad = np.array([self._delta_traditional_call(S, K, T, r, sigma) for S in S_array])
            delta_inv = np.array([self._delta_inverse_call(S, K, T, r, sigma) for S in S_array])
            title = "Delta Comparison: Inverse vs Traditional Call Option"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(S_array / 1000, delta_trad, 'b-', linewidth=2, label='Traditional Delta')
        ax.plot(S_array / 1000, delta_inv, 'r--', linewidth=2, label='Inverse Delta')
        
        ax.axvline(K / 1000, color='gray', linestyle=':', alpha=0.7, 
                   label=f'Strike (K=${K/1000:.0f}k)')
        ax.axhline(0, color='black', linewidth=0.5)
        
        # Highlight divergence zone
        if option_type.lower() == "put":
            # For puts, greater divergence when S < K (ITM)
            ax.fill_between(
                S_array[S_array < K] / 1000,
                delta_trad[S_array < K],
                delta_inv[S_array < K],
                alpha=0.2, color='orange',
                label='Divergence (ITM)'
            )
        else:
            # For calls, greater divergence when S > K (ITM)
            ax.fill_between(
                S_array[S_array > K] / 1000,
                delta_trad[S_array > K],
                delta_inv[S_array > K],
                alpha=0.2, color='orange',
                label='Divergence (ITM)'
            )
        
        ax.set_xlabel('Underlying Price S (thousands USD)', fontsize=12)
        ax.set_ylabel('Delta', fontsize=12)
        ax.set_title(f'{title}\nK=${K/1000:.0f}k, T={T*365:.0f} days, σ={sigma*100:.0f}%', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.output_dir / f"{save_name}.png")
        
        plt.close(fig)
        return fig
    
    def plot_all_greeks_comparison(
        self,
        K: float = 100_000,
        T: float = 30/365,
        r: float = 0.05,
        sigma: float = 0.60,
        S_range: Tuple[float, float] = None,
        save_name: str = "greeks_comparison",
        show: bool = True
    ) -> plt.Figure:
        """
        Complete panel with all Greeks for calls and puts.
        """
        # If S_range not specified, use ±40% from K
        if S_range is None:
            S_range = (K * 0.6, K * 1.4)
        
        S_array = np.linspace(S_range[0], S_range[1], 300)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Put Delta
        ax = axes[0, 0]
        delta_trad_put = [self._delta_traditional_put(S, K, T, r, sigma) for S in S_array]
        delta_inv_put = [self._delta_inverse_put(S, K, T, r, sigma) for S in S_array]
        ax.plot(S_array / 1000, delta_trad_put, 'b-', linewidth=2, label='Traditional')
        ax.plot(S_array / 1000, delta_inv_put, 'r--', linewidth=2, label='Inverse')
        ax.axvline(K / 1000, color='gray', linestyle=':', alpha=0.7)
        ax.set_title('PUT Delta')
        ax.set_xlabel('S (k USD)')
        ax.set_ylabel('Delta')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Call Delta
        ax = axes[0, 1]
        delta_trad_call = [self._delta_traditional_call(S, K, T, r, sigma) for S in S_array]
        delta_inv_call = [self._delta_inverse_call(S, K, T, r, sigma) for S in S_array]
        ax.plot(S_array / 1000, delta_trad_call, 'b-', linewidth=2, label='Traditional')
        ax.plot(S_array / 1000, delta_inv_call, 'r--', linewidth=2, label='Inverse')
        ax.axvline(K / 1000, color='gray', linestyle=':', alpha=0.7)
        ax.set_title('CALL Delta')
        ax.set_xlabel('S (k USD)')
        ax.set_ylabel('Delta')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Put Prices
        ax = axes[1, 0]
        price_trad_put = [self._bs_put_price(S, K, T, r, sigma) for S in S_array]
        price_inv_put = [self._bs_put_price(S, K, T, r, sigma) / S for S in S_array]
        ax.plot(S_array / 1000, np.array(price_trad_put) / 1000, 'b-', linewidth=2, label='Traditional (k USD)')
        ax.plot(S_array / 1000, price_inv_put, 'r--', linewidth=2, label='Inverse (BTC)')
        ax.axvline(K / 1000, color='gray', linestyle=':', alpha=0.7)
        ax.set_title('PUT Price')
        ax.set_xlabel('S (k USD)')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Call Prices
        ax = axes[1, 1]
        price_trad_call = [self._bs_call_price(S, K, T, r, sigma) for S in S_array]
        price_inv_call = [self._bs_call_price(S, K, T, r, sigma) / S for S in S_array]
        ax.plot(S_array / 1000, np.array(price_trad_call) / 1000, 'b-', linewidth=2, label='Traditional (k USD)')
        ax.plot(S_array / 1000, price_inv_call, 'r--', linewidth=2, label='Inverse (BTC)')
        ax.axvline(K / 1000, color='gray', linestyle=':', alpha=0.7)
        ax.set_title('CALL Price')
        ax.set_xlabel('S (k USD)')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'Greeks Comparison: Inverse vs Traditional Options\nK=${K/1000:.0f}k, T={T*365:.0f}d, σ={sigma*100:.0f}%', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            fig.savefig(self.output_dir / f"{save_name}.png")
        
        plt.close(fig)
        return fig


# NOTE: This module should be run ONLY via main_c.py
# Example: python main_c.py --plots
