import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter
from datetime import datetime, timedelta
from src.config.plot_style import apply_thesis_style, PALETTE

apply_thesis_style()

# Try to import yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("="*70)
    print("ERROR: yfinance module not available")
    print("Please install it with: pip install yfinance")
    print("="*70)


class VolatilityComparison:
    def __init__(self, window=30):
        """
        Initialize volatility comparison analyzer
        
        Parameters:
        -----------
        window : int
            Rolling window size in days for volatility calculation
        """
        self.window = window
        self.data = None
        self.stats = None
        
    def fetch_bitcoin_data(self, start_date='2015-01-01', end_date='2024-12-31'):
        """
        Fetch Bitcoin historical data from Yahoo Finance (BTC-USD)
        """
        if not HAS_YFINANCE:
            print("[FAIL] yfinance module not available")
            return None
            
        try:
            print(f"Fetching Bitcoin data from Yahoo Finance ({start_date} to {end_date})...")
            
            btc = yf.Ticker("BTC-USD")
            df = btc.history(start=start_date, end=end_date, auto_adjust=True)
            
            if df.empty:
                print("[FAIL] No Bitcoin data returned")
                return None
            
            prices = df[['Close']].rename(columns={'Close': 'price'})
            prices.index = prices.index.tz_localize(None)
            
            print(f"[OK] Bitcoin data fetched: {len(prices)} observations")
            print(f"  Date range: {prices.index.min().date()} to {prices.index.max().date()}")
            
            return prices
            
        except Exception as e:
            print(f"[FAIL] Error fetching Bitcoin data: {e}")
            return None
        
    def fetch_spy_data(self, start_date='2015-01-01', end_date='2024-12-31'):
        """
        Fetch S&P 500 (SPY) data from Yahoo Finance
        """
        if not HAS_YFINANCE:
            print("[FAIL] yfinance module not available")
            return None
            
        try:
            print(f"Fetching S&P 500 (SPY) data from Yahoo Finance ({start_date} to {end_date})...")
            
            spy = yf.Ticker("SPY")
            df = spy.history(start=start_date, end=end_date, auto_adjust=True)
            
            if df.empty:
                print("[FAIL] No S&P 500 data returned")
                return None
            
            prices = df[['Close']].rename(columns={'Close': 'price'})
            prices.index = prices.index.tz_localize(None)
            
            print(f"[OK] S&P 500 data fetched: {len(prices)} observations")
            print(f"  Date range: {prices.index.min().date()} to {prices.index.max().date()}")
            
            return prices
            
        except Exception as e:
            print(f"[FAIL] Error fetching S&P 500 data: {e}")
            return None
    
    def calculate_returns(self, prices):
        """Calculate log returns from price series"""
        returns = np.log(prices / prices.shift(1))
        return returns.dropna()
    
    def calculate_rolling_volatility(self, returns, window):
        """Calculate rolling annualized volatility"""
        rolling_std = returns.rolling(window=window).std()
        annualized_vol = rolling_std * np.sqrt(252) * 100
        return annualized_vol.dropna()
    
    def process_data(self, start_date='2015-01-01', end_date='2024-12-31'):
        """Fetch and process data to calculate volatilities"""
        print("\n" + "="*70)
        print("FETCHING DATA")
        print("="*70)
        
        btc_prices = self.fetch_bitcoin_data(start_date=start_date, end_date=end_date)
        if btc_prices is None:
            print("\n[FAIL] Failed to fetch Bitcoin data")
            return False
        
        print()
        
        spy_prices = self.fetch_spy_data(start_date=start_date, end_date=end_date)
        if spy_prices is None:
            print("\n[FAIL] Failed to fetch S&P 500 data")
            return False
        
        print(f"\nAligning data on common dates...")
        btc_start = btc_prices.index.min()
        btc_end = btc_prices.index.max()
        spy_start = spy_prices.index.min()
        spy_end = spy_prices.index.max()
        
        print(f"  BTC range: {btc_start.date()} to {btc_end.date()}")
        print(f"  SPY range: {spy_start.date()} to {spy_end.date()}")
        
        common_start = max(btc_start, spy_start)
        common_end = min(btc_end, spy_end)
        
        btc_prices = btc_prices[(btc_prices.index >= common_start) & (btc_prices.index <= common_end)]
        spy_prices = spy_prices[(spy_prices.index >= common_start) & (spy_prices.index <= common_end)]
        
        if len(btc_prices) == 0 or len(spy_prices) == 0:
            print("[FAIL] No data available after filtering")
            return False
        
        print(f"[OK] Common date range: {common_start.date()} to {common_end.date()}")
        
        print(f"\nCalculating returns...")
        btc_returns = self.calculate_returns(btc_prices['price'])
        spy_returns = self.calculate_returns(spy_prices['price'])
        
        print(f"  BTC returns: {len(btc_returns)} observations")
        print(f"  SPY returns: {len(spy_returns)} observations")
        
        print(f"\nCalculating rolling volatility (window: {self.window} days)...")
        btc_vol = self.calculate_rolling_volatility(btc_returns, self.window)
        spy_vol = self.calculate_rolling_volatility(spy_returns, self.window)
        
        self.data = pd.DataFrame({
            'BTC': btc_vol,
            'SPX': spy_vol
        }).dropna()
        
        if len(self.data) == 0:
            print("[FAIL] No overlapping data after processing")
            return False
        
        self.calculate_statistics()
        
        print(f"[OK] Data processed: {len(self.data)} observations")
        print("="*70 + "\n")
        
        return True
        
    def calculate_statistics(self):
        """Calculate summary statistics"""
        if self.data is None or len(self.data) == 0:
            return
        
        self.stats = {
            'btc_mean': self.data['BTC'].mean(),
            'btc_std': self.data['BTC'].std(),
            'btc_min': self.data['BTC'].min(),
            'btc_max': self.data['BTC'].max(),
            'btc_median': self.data['BTC'].median(),
            'spx_mean': self.data['SPX'].mean(),
            'spx_std': self.data['SPX'].std(),
            'spx_min': self.data['SPX'].min(),
            'spx_max': self.data['SPX'].max(),
            'spx_median': self.data['SPX'].median(),
            'correlation': self.data['BTC'].corr(self.data['SPX']),
        }
    
    def plot_volatility(self, save_path=''):
        """Create high-quality figure for publication"""
        if self.data is None or len(self.data) == 0:
            print("No data to plot")
            return None, None
        
        # Prepare data with Date column
        df = self.data.copy()
        df['Date'] = df.index
        
        # Create high-quality figure for publication
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        
        # Plot with distinct colors
        ax.plot(df['Date'], df['BTC'], linewidth=1.5, color=PALETTE['quaternary'], label='Bitcoin', alpha=0.9)
        ax.plot(df['Date'], df['SPX'], linewidth=1.5, color=PALETTE['primary'], label='S&P 500', alpha=0.9)
        
        # Formatting
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Annualized Realized Volatility (%)', fontsize=12, fontweight='bold')
        
        # Get year range from data
        start_year = df['Date'].min().year
        end_year = df['Date'].max().year
        ax.set_title(f'Comparison of Annualized Realized Volatility: Bitcoin vs S&P 500 ({start_year}-{end_year})', 
                     fontsize=13, fontweight='bold', pad=15)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Format x-axis
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        
        # Legend
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
        
        # Set y-axis limits
        ax.set_ylim(0, 200)
        
        # Tight layout
        plt.tight_layout()
            
        png_path = f'btc_vs_rv_b.png'

        plt.savefig(png_path, dpi=300, bbox_inches='tight')

        print("Figure saved successfully!")
        print(f"  - {png_path}")

        plt.close(fig)

        return png_path
    
    def print_statistics(self):
        """Print summary statistics to console"""
        if self.stats is None:
            print("No statistics available")
            return
        
        start_year = self.data.index.min().year
        end_year = self.data.index.max().year
        
        print(f"\nSummary Statistics ({start_year}-{end_year}):")
        print(f"\nBitcoin:")
        print(f"  Mean: {self.stats['btc_mean']:.2f}%")
        print(f"  Std Dev: {self.stats['btc_std']:.2f}%")
        print(f"  Min: {self.stats['btc_min']:.2f}%")
        print(f"  Max: {self.stats['btc_max']:.2f}%")
        
        print(f"\nS&P 500:")
        print(f"  Mean: {self.stats['spx_mean']:.2f}%")
        print(f"  Std Dev: {self.stats['spx_std']:.2f}%")
        print(f"  Min: {self.stats['spx_min']:.2f}%")
        print(f"  Max: {self.stats['spx_max']:.2f}%")
        
        ratio = self.stats['btc_mean'] / self.stats['spx_mean']
        print(f"\nVolatility Ratio (BTC/SPX): {ratio:.2f}x")
    
    def export_data(self, filepath='volatility_data.csv'):
        """Export processed data to CSV"""
        if self.data is None:
            print("No data to export")
            return
        
        self.data.to_csv(filepath)
        print(f"[OK] Data exported to: {filepath}")


def run_btc_spy_comparison():
    """Funzione pubblica per il confronto BTC vs SPY."""
    print("\n" + "="*70)
    print("Bitcoin vs S&P 500 Realized Volatility Comparison")
    print("="*70)
    print("\nData Source: Yahoo Finance (yfinance)")
    print("  - Bitcoin: BTC-USD")
    print("  - S&P 500: SPY ETF")
    print("\nNo API keys required!")
    print("="*70)

    if not HAS_YFINANCE:
        print("\n[X] Please install yfinance: pip install yfinance")
        return

    # Create analyzer with 30-day rolling window
    analyzer = VolatilityComparison(window=30)

    # Process data from 2015 to 2024
    success = analyzer.process_data(
        start_date='2015-01-01',
        end_date='2024-12-31'
    )

    if not success:
        print("\n[X] Failed to process data.")
        print("\nTroubleshooting:")
        print("  - Check your internet connection")
        print("  - Try: pip install --upgrade yfinance")
        return

    # Create plot
    analyzer.plot_volatility()

    # Print statistics
    analyzer.print_statistics()

    # Export data
    analyzer.export_data('volatility_data_2015_2024.csv')


# NOTA: Questo modulo deve essere eseguito SOLO tramite main_c.py
# Esempio: python main_c.py --btc-spy