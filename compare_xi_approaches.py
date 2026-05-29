"""
compare_xi_approaches.py
========================
Confronto tra due approcci per la curva di varianza forward xi_0(t)
nell'ambito della calibrazione del modello rBergomi.

===========================================================================
SPIEGAZIONE: cos'è xi_0(t) e perché è diversa da eta
===========================================================================

Nel modello rBergomi il processo di varianza è:

    V_t = xi_0(t) · exp( eta · Z^H_t  −  0.5 · eta² · t^{2H} )

Dove:
  - xi_0(t)  [FORWARD VARIANCE CURVE]  è il "livello atteso" della varianza
              ad ogni istante t. È una funzione deterministica, calibrata dai
              dati di mercato (ATM implied vol). NON è un parametro libero
              dell'ottimizzatore.
  - eta      [VOL-OF-VOL]  controlla quanto la varianza si muove attorno al
              suo livello atteso xi_0(t). È un parametro scalare, ottimizzato.
  - H        [HURST]  controlla la rugosità (roughness) della varianza.
              H < 0.5 → varianza rough (path di V_t non differenziabili).
  - rho      [CORRELAZIONE spot-vol]  determina il segno e la magnitudo
              dello skew della superficie di IV.
  - Z^H_t    è un moto Browniano frazionario con Hurst H.

La proprietà chiave di xi_0(t) è:
    E_Q[ V_t ] = xi_0(t)
    (il termine exp(...) ha media 1 sotto Q grazie alla correzione -0.5·eta²·t^{2H})

Quindi xi_0(t) fissa il LIVELLO MEDIO della varianza ad ogni t.
eta, H, rho controllano la FORMA della superficie di IV (skew, term structure).

===========================================================================
I DUE APPROCCI A CONFRONTO
===========================================================================

APPROCCIO A — Naive (corrente, SBAGLIATO):
    Imposta:  xi_0(t) = sigma_ATM(T_k)^2  per t ∈ [T_{k-1}, T_k]

    Problema: la varianza integrata che il modello "vede" per la maturity T_k è
        E[IV²(T_k)] = (1/T_k) · ∫₀^{T_k} xi_0(t) dt
                    = media pesata di sigma_ATM(T_1)², ..., sigma_ATM(T_k)²
                    ≠ sigma_ATM(T_k)²   (salvo term structure piatta)

    Per BTC (term structure decrescente: IV breve > IV lunga), il modello
    "accumula" troppa varianza dai segmenti più brevi → IVs traslate in ALTO.

APPROCCIO B — Forward Variance Bootstrap (CORRETTO):
    Risolve l'equazione per ogni segmento in modo che:
        E[IV²(T_k)] = sigma_ATM(T_k)^2   per ogni k (per costruzione)

    Formula (bootstrap ricorsivo):
        f_k = ( sigma_k² · T_k  −  sigma_{k-1}² · T_{k-1} ) / ( T_k − T_{k-1} )

    Con T_0 = 0, sigma_0 = 0. Il primo segmento dà f_1 = sigma_1² (uguale al
    naive), ma tutti i successivi sono corretti. Se f_k < 0 (calendar spread
    arbitrage nei dati) si applica un floor a 1e-6.

===========================================================================
OUTPUT
===========================================================================
  Results/xi_comparison/
    tables/
      comparison_{method}.csv          — per-date stats per ogni metodo
      summary_all_methods.csv          — media aggregata per metodo
    figures/
      term_structure_{date}_{method}.png  — ATM term structure + xi curves
      scatter_{date}_{method}.png         — scatter model vs market IV
      bias_by_maturity_{method}.png       — bias medio per maturity bucket
      summary_barplot.png                 — RMSE/bias confronto cross-method

Usage:
    python compare_xi_approaches.py                          # tutti i metodi, tutte le date
    python compare_xi_approaches.py --method hybrid_euler   # un solo metodo
    python compare_xi_approaches.py --date 20241204         # una data, tutti i metodi
    python compare_xi_approaches.py --n-paths 5000          # più preciso (più lento)
"""

import argparse
import logging
import sys
import warnings
import time
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── Logging ───────────────────────────────────────────────────────────────────
OUT_DIR = Path("Results/xi_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "tables").mkdir(exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUT_DIR / "compare_xi.log", mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("XiCompare")

# ── Project imports ───────────────────────────────────────────────────────────
from iv_surface_builder import IVSurfaceBuilder
from src.rbergomi.calibrator import Calibrator, ForwardVarianceCurve
from src.rbergomi.pricer import rBergomiPricer
from src.rbergomi.utils import implied_vol_batch
from src.config.surfaces import SURFACE_CFG
from src.config.methods import METHODS, METHOD_LABELS
from src.config.plot_style import apply_thesis_style, PALETTE

apply_thesis_style()

DATA_DIR   = Path("data/option")
TABLES_DIR = Path("Results/calibration/tables")
ALL_METHODS = ["cholesky_euler", "hybrid_euler", "hybrid_mixed"]

# Colori dedicati ai due approcci
COLOR_NAIVE = PALETTE['tertiary']   # magenta-pink  → approccio A (sbagliato)
COLOR_CORR  = PALETTE['primary']    # blu scuro      → approccio B (corretto)
COLOR_MKT   = PALETTE['neutral']    # grigio         → mercato


# =============================================================================
# BOOTSTRAP FORWARD VARIANCE
# =============================================================================

def bootstrap_forward_variance(
    maturities: np.ndarray,
    atm_ivs: np.ndarray,
    floor: float = 1e-6,
) -> np.ndarray:
    """
    Calcola la varianza forward piecewise-constant dal bootstrap:

        f_k = (sigma_k² · T_k − sigma_{k-1}² · T_{k-1}) / (T_k − T_{k-1})

    Parameters
    ----------
    maturities : np.ndarray  — T_1 < T_2 < ... < T_n  (anni)
    atm_ivs    : np.ndarray  — sigma_ATM al corrispondente T_k
    floor      : float       — valore minimo per f_k (evita valori negativi)

    Returns
    -------
    np.ndarray — varianze forward f_1, ..., f_n
    """
    total_var = atm_ivs**2 * maturities
    T_prev  = np.concatenate([[0.0], maturities[:-1]])
    tv_prev = np.concatenate([[0.0], total_var[:-1]])
    fwd     = (total_var - tv_prev) / (maturities - T_prev)

    neg = fwd < 0
    if neg.any():
        for i in np.where(neg)[0]:
            log.debug(
                f"  f[{i}] < 0 (T={maturities[i]*252:.0f}d): "
                f"total_var decrescente ({tv_prev[i]:.5f} → {total_var[i]:.5f}) — floored"
            )
    return np.maximum(fwd, floor)


def expected_model_atm_sq(
    forward_vars: np.ndarray,
    maturities: np.ndarray,
) -> np.ndarray:
    """
    Calcola E[IV²(T_k)] = (1/T_k) · Σ f_i · ΔT_i
    per entrambe le versioni della curva xi (naive o corretta).
    """
    T_prev = np.concatenate([[0.0], maturities[:-1]])
    dT = maturities - T_prev
    result = np.array([
        np.sum(forward_vars[:k+1] * dT[:k+1]) / maturities[k]
        for k in range(len(maturities))
    ])
    return result


# =============================================================================
# CARICAMENTO DATI
# =============================================================================

def load_calibration_row(date_str: str, method: str) -> Optional[dict]:
    """Carica parametri calibrati (H, eta, rho) + metriche di velocità dal CSV."""
    csv = TABLES_DIR / method / f"calibration_{date_str}_{method}.csv"
    if not csv.exists():
        return None
    row = pl.read_csv(csv).row(0, named=True)
    return {
        "H":            float(row["H"]),
        "eta":          float(row["eta"]),
        "rho":          float(row["rho"]),
        "calib_sec":    float(row.get("time_seconds", np.nan)),
        "n_evals":      int(row.get("n_evals", 0)),
        "n_points":     int(row.get("n_points", 0)),
    }


def load_market_data(
    dt: datetime,
    builder: IVSurfaceBuilder,
    moneyness_range: Tuple[float, float] = (0.80, 1.20),
) -> Optional[dict]:
    """Carica e filtra la superficie di mercato."""
    try:
        surface     = builder.get_iv_surface(target_time=dt, **SURFACE_CFG)
        market_data = builder.export_for_rbergomi(surface)
    except Exception as e:
        log.warning(f"  Impossibile caricare dati di mercato: {e}")
        return None

    S0   = market_data["S"]
    mono = market_data["K"] / S0
    m_lo, m_hi = moneyness_range
    keep = (mono >= m_lo) & (mono <= m_hi)

    if keep.sum() < 5:
        return None

    return {
        "S":         S0,
        "K":         market_data["K"][keep],
        "tau":       market_data["tau"][keep],
        "iv_market": market_data["iv_market"][keep],
        "n_points":  int(keep.sum()),
    }


def extract_atm_ivs(market_data: dict, calibrator: Calibrator) -> Tuple[np.ndarray, np.ndarray]:
    """Restituisce (maturities, atm_ivs) usando il finder interno del calibratore."""
    calibrator._xi0_curve = None
    _, atm_dict = calibrator._estimate_forward_variance_curve(market_data)
    mats = np.array(sorted(atm_dict.keys()))
    ivs  = np.array([atm_dict[T] for T in mats])
    return mats, ivs


# =============================================================================
# PRICING
# =============================================================================

def price_surface_with_xi(
    H: float, eta: float, rho: float,
    xi_curve: ForwardVarianceCurve,
    market_data: dict,
    method_cfg: dict,
    n_paths: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Prezza la superficie con i parametri dati e la curva xi specificata.

    Returns (model_ivs_full, iv_mkt_full, tau_full, elapsed_seconds).
    Gli array hanno la stessa lunghezza dei dati di mercato in input;
    i punti dove l'inversione IV fallisce restano NaN in model_ivs_full.
    La maschera NaN viene applicata dal chiamante su entrambe le curve
    congiuntamente, garantendo array allineati per plot e statistiche.
    """
    t0 = time.perf_counter()

    pricer = rBergomiPricer(
        H=H, eta=eta, rho=rho, xi=xi_curve,
        n_paths=n_paths,
        n_steps=method_cfg.get("n_steps", 100),
        scheme=method_cfg.get("scheme", "hybrid"),
        kappa=method_cfg.get("kappa", 6),
        seed=method_cfg.get("seed", 42),
        antithetic=method_cfg.get("antithetic", False),
        pricing_method=method_cfg.get("pricing_method", "euler"),
    )

    S0      = market_data["S"]
    K_arr   = market_data["K"]
    tau_arr = market_data["tau"]
    iv_mkt  = market_data["iv_market"]

    model_prices = np.full(len(K_arr), np.nan)
    for T in np.unique(tau_arr):
        mask = tau_arr == T
        idx  = np.where(mask)[0]
        try:
            prices, _ = pricer.price_multiple_strikes(
                S0, K_arr[mask], T, option_type='put'
            )
            model_prices[idx] = prices
        except Exception:
            for j, ii in enumerate(idx):
                try:
                    p, _ = pricer.price_inverse_put(S0, K_arr[ii], T)
                    model_prices[ii] = p
                except Exception:
                    pass

    model_ivs = implied_vol_batch(model_prices, S0, K_arr, tau_arr,
                                  option_type='put', r=0.0)
    elapsed = time.perf_counter() - t0
    # Restituisce array COMPLETI (stessa dimensione del mercato) — NaN dove fallisce
    return model_ivs, iv_mkt, tau_arr, elapsed


def error_stats(model_ivs: np.ndarray, market_ivs: np.ndarray) -> dict:
    """
    Statistiche di errore in percentage points.
    Filtra automaticamente i NaN prima del calcolo.
    """
    valid = ~np.isnan(model_ivs) & ~np.isnan(market_ivs)
    err = (model_ivs[valid] - market_ivs[valid]) * 100
    return {
        "bias_pp": float(np.mean(err)),
        "rmse_pp": float(np.sqrt(np.mean(err**2))),
        "mae_pp":  float(np.mean(np.abs(err))),
        "std_pp":  float(np.std(err, ddof=1)),
        "n":       int(valid.sum()),
    }


# =============================================================================
# GRAFICI
# =============================================================================

def plot_term_structure(
    maturities: np.ndarray,
    atm_ivs: np.ndarray,
    date_str: str,
    method: str,
) -> None:
    """
    Due pannelli:
      Sinistra: sigma_ATM(T) di mercato + E[IV(T)] stimata sotto i due approcci
      Destra:   xi_0(t) piecewise-constant per i due approcci
    """
    naive_xi  = atm_ivs**2
    corr_xi   = bootstrap_forward_variance(maturities, atm_ivs)

    # E[IV²(T_k)] sotto i due approcci
    e_naive = expected_model_atm_sq(naive_xi, maturities)
    e_corr  = expected_model_atm_sq(corr_xi,  maturities)

    T_days = maturities * 252
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Pannello 1: ATM IV recovery ────────────────────────────────────────
    ax = axes[0]
    ax.plot(T_days, atm_ivs * 100, 'o-', color=COLOR_MKT, lw=2, ms=7,
            label='Mercato $\\sigma_{\\mathrm{ATM}}(T)$', zorder=3)
    ax.plot(T_days, np.sqrt(e_naive) * 100, 's--', color=COLOR_NAIVE,
            lw=1.8, ms=6, label='Modello (A — naive): $\\sqrt{E[IV^2(T)]}$')
    ax.plot(T_days, np.sqrt(e_corr) * 100,  '^-',  color=COLOR_CORR,
            lw=1.8, ms=6, label='Modello (B — corretto): $\\sqrt{E[IV^2(T)]}$')

    # Annotazioni errore sul naive
    for k, (Td, iv_mkt, iv_naive) in enumerate(
            zip(T_days, atm_ivs*100, np.sqrt(e_naive)*100)):
        diff = iv_naive - iv_mkt
        if abs(diff) > 0.5:
            ax.annotate(f'{diff:+.1f}pp', xy=(Td, iv_naive),
                        xytext=(5, 4), textcoords='offset points',
                        fontsize=7.5, color=COLOR_NAIVE)

    ax.set_xlabel("Maturity (giorni)")
    ax.set_ylabel("ATM Implied Vol (%)")
    ax.set_title(f"Recupero ATM IV — {date_str}")
    ax.legend(fontsize=8.5)

    # ── Pannello 2: xi_0(t) piecewise-constant ────────────────────────────
    ax = axes[1]
    # Disegna la step function su un range esteso
    T_prev = np.concatenate([[0.0], maturities[:-1]])
    for k in range(len(maturities)):
        t0, t1 = T_prev[k] * 252, maturities[k] * 252
        ax.hlines(naive_xi[k], t0, t1, colors=COLOR_NAIVE, lw=2.5,
                  label='(A) naive $\\xi_0 = \\sigma_{\\mathrm{ATM}}^2$' if k == 0 else "")
        ax.hlines(corr_xi[k],  t0, t1, colors=COLOR_CORR,  lw=2.5,
                  label='(B) corr. $\\xi_0 = f_k$ (bootstrap)' if k == 0 else "")
        ax.vlines(t1, min(naive_xi[k], corr_xi[k]),
                  max(naive_xi[k], corr_xi[k]),
                  colors=COLOR_CORR, lw=1, ls=':')

    ax.plot(T_days, atm_ivs**2, 'o', color=COLOR_MKT, ms=6, zorder=3,
            label='$\\sigma_{\\mathrm{ATM}}^2(T_k)$  (target)')
    ax.set_xlabel("Maturity (giorni)")
    ax.set_ylabel("Varianza forward $\\xi_0(t)$")
    ax.set_title("Curva $\\xi_0(t)$: naive vs bootstrap")
    ax.legend(fontsize=8.5)

    fig.suptitle(
        f"{METHOD_LABELS.get(method, method)} — {date_str}",
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    path = OUT_DIR / "figures" / f"term_structure_{date_str}_{method}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"    Figure: {path.name}")


def plot_scatter(
    iv_naive: np.ndarray, iv_corr: np.ndarray,
    iv_mkt: np.ndarray,   tau: np.ndarray,
    date_str: str, method: str,
    stats_naive: dict, stats_corr: dict,
) -> None:
    """Scatter model IV vs market IV per entrambi gli approcci."""
    n_naive = iv_naive * 100
    n_corr  = iv_corr  * 100
    n_mkt   = iv_mkt   * 100
    T_days  = tau * 252

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, model_iv, color, label, stats in zip(
        axes,
        [n_naive, n_corr],
        [COLOR_NAIVE, COLOR_CORR],
        ["(A) Naive: $\\xi_0 = \\sigma_{\\mathrm{ATM}}^2$",
         "(B) Corretto: $\\xi_0 = f_k$  (bootstrap)"],
        [stats_naive, stats_corr],
    ):
        sc = ax.scatter(n_mkt, model_iv, c=T_days, cmap='plasma',
                        alpha=0.75, s=28, zorder=2)
        all_vals = np.concatenate([n_mkt, model_iv])
        lim = [all_vals.min() - 3, all_vals.max() + 3]
        ax.plot(lim, lim, '--', color=COLOR_MKT, lw=1.2, zorder=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Market IV (%)")
        ax.set_ylabel("Model IV (%)")
        ax.set_title(label, fontsize=10, color=color)

        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("Maturity (giorni)", fontsize=8)

        txt = (f"Bias = {stats['bias_pp']:+.2f} pp\n"
               f"RMSE = {stats['rmse_pp']:.2f} pp\n"
               f"MAE  = {stats['mae_pp']:.2f} pp\n"
               f"N    = {stats['n']}")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top',
                fontsize=8.5, family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=color, alpha=0.9))

    fig.suptitle(
        f"{METHOD_LABELS.get(method, method)} — {date_str}\n"
        f"Model IV vs Market IV",
        fontsize=11, fontweight='bold'
    )
    fig.tight_layout()
    path = OUT_DIR / "figures" / f"scatter_{date_str}_{method}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"    Figure: {path.name}")


def plot_bias_by_maturity(
    all_rows: List[dict], method: str
) -> None:
    """
    Boxplot del bias (pp) per bucket di maturity, naive vs corretto.
    Aggrega tutti i punti di tutte le date per un dato metodo.
    """
    if not all_rows:
        return

    # Raccoglie residui per bucket di maturity
    buckets = {"≤14d": (0, 14), "15–30d": (14, 30), "31–60d": (30, 60), ">60d": (60, 999)}
    data_naive = {b: [] for b in buckets}
    data_corr  = {b: [] for b in buckets}

    for row in all_rows:
        for T_days, err_n, err_c in zip(
            row["tau_days"], row["err_naive_pp"], row["err_corr_pp"]
        ):
            for bname, (lo, hi) in buckets.items():
                if lo < T_days <= hi:
                    data_naive[bname].append(err_n)
                    data_corr[bname].append(err_c)

    bnames = list(buckets.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, data, color, title in zip(
        axes,
        [data_naive, data_corr],
        [COLOR_NAIVE, COLOR_CORR],
        ["(A) Naive: $\\xi_0 = \\sigma_{\\mathrm{ATM}}^2$",
         "(B) Corretto: $\\xi_0 = f_k$ (bootstrap)"],
    ):
        vals = [data[b] for b in bnames]
        bp = ax.boxplot(vals, labels=bnames, patch_artist=True,
                        medianprops=dict(color='black', lw=1.5))
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.axhline(0, color=COLOR_MKT, lw=1.2, ls='--')
        ax.set_xlabel("Bucket di maturity")
        ax.set_ylabel("Residuo modello − mercato (pp)")
        ax.set_title(title, color=color, fontsize=10)

        # Media per bucket
        for i, b in enumerate(bnames, start=1):
            if data[b]:
                m = np.mean(data[b])
                ax.text(i, ax.get_ylim()[1] * 0.92, f"μ={m:+.1f}",
                        ha='center', fontsize=8, color=color)

    fig.suptitle(
        f"{METHOD_LABELS.get(method, method)}\n"
        f"Bias per bucket di maturity (tutte le date)",
        fontsize=11, fontweight='bold'
    )
    fig.tight_layout()
    path = OUT_DIR / "figures" / f"bias_by_maturity_{method}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"  Figure: {path.name}")


def plot_summary_barplot(summary_df: pl.DataFrame) -> None:
    """
    Bar chart 2×3: bias e RMSE medi (naive vs corretto) per i tre metodi.
    """
    methods = summary_df["method"].to_list()
    labels  = [METHOD_LABELS.get(m, m) for m in methods]

    bias_n = summary_df["mean_bias_naive_pp"].to_list()
    bias_c = summary_df["mean_bias_corr_pp"].to_list()
    rmse_n = summary_df["mean_rmse_naive_pp"].to_list()
    rmse_c = summary_df["mean_rmse_corr_pp"].to_list()

    x = np.arange(len(methods))
    w = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, n_vals, c_vals, ylabel, title in zip(
        axes,
        [bias_n, rmse_n],
        [bias_c, rmse_c],
        ["Bias medio (pp)", "RMSE medio (pp)"],
        ["Bias medio: naive vs corretto", "RMSE medio: naive vs corretto"],
    ):
        bars_n = ax.bar(x - w/2, n_vals, w, color=COLOR_NAIVE, alpha=0.85,
                        label="(A) Naive $\\xi_0 = \\sigma_{\\mathrm{ATM}}^2$")
        bars_c = ax.bar(x + w/2, c_vals, w, color=COLOR_CORR,  alpha=0.85,
                        label="(B) Corretto $\\xi_0 = f_k$")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color=COLOR_MKT, lw=1, ls='--')
        ax.legend(fontsize=9)

        for bar in list(bars_n) + list(bars_c):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (0.4 if h >= 0 else -1.2),
                f"{h:+.1f}", ha='center', va='bottom', fontsize=8.5,
                fontweight='bold'
            )

    fig.suptitle("Confronto approcci xi_0: tutti i metodi", fontsize=12, fontweight='bold')
    fig.tight_layout()
    path = OUT_DIR / "figures" / "summary_barplot.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"  Figure: {path.name}")


def plot_bias_timeline(all_method_dfs: Dict[str, pl.DataFrame]) -> None:
    """
    Timeline del bias medio giornaliero (naive vs corretto) per ogni metodo.
    Un subplot per metodo.
    """
    n_methods = len(all_method_dfs)
    if n_methods == 0:
        return

    fig, axes = plt.subplots(n_methods, 1, figsize=(14, 3.5 * n_methods), sharex=True)
    if n_methods == 1:
        axes = [axes]

    for ax, (method, df) in zip(axes, all_method_dfs.items()):
        if len(df) == 0:
            continue
        dates = [datetime.strptime(d, "%Y%m%d") for d in df["date"].to_list()]
        bias_n = df["bias_naive_pp"].to_list()
        bias_c = df["bias_corr_pp"].to_list()

        ax.plot(dates, bias_n, 'o--', color=COLOR_NAIVE, lw=1.6, ms=5,
                label="(A) Naive", alpha=0.85)
        ax.plot(dates, bias_c, 's-',  color=COLOR_CORR,  lw=1.6, ms=5,
                label="(B) Corretto", alpha=0.85)
        ax.axhline(0, color=COLOR_MKT, lw=1, ls='--')
        ax.set_ylabel("Bias medio (pp)", fontsize=9)
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=10)
        ax.legend(fontsize=8.5, loc='upper right')

        # Evidenzia eventi principali
        events = {
            datetime(2022, 5, 9):  "LUNA",
            datetime(2022, 11, 8): "FTX",
            datetime(2023, 3, 10): "SVB",
            datetime(2024, 1, 10): "ETF",
            datetime(2024, 4, 20): "Halving",
            datetime(2024, 12, 5): "$100k",
            datetime(2025, 1, 20): "Trump",
        }
        ylim = ax.get_ylim()
        for ev_dt, ev_name in events.items():
            ax.axvline(ev_dt, color='#aaaaaa', lw=0.8, ls=':')
            ax.text(ev_dt, ylim[1] * 0.92, ev_name, fontsize=7,
                    color='#666666', ha='center')

    fig.suptitle("Bias timeline: naive vs corretto (tutte le date)", fontsize=12)
    fig.tight_layout()
    path = OUT_DIR / "figures" / "bias_timeline.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"  Figure: {path.name}")


# =============================================================================
# GRAFICI DI VELOCITÀ
# =============================================================================

def plot_speed_xi(all_dfs: Dict[str, pl.DataFrame]) -> None:
    """
    Confronto velocità re-pricing: naive vs corretto, per ogni metodo.

    Due pannelli:
      Sinistra: tempo medio di re-pricing (secondi) per approccio e metodo
      Destra:   scatter RMSE vs tempo — tradeoff accuratezza/velocità
    """
    methods = [m for m in ALL_METHODS if m in all_dfs and len(all_dfs[m]) > 0]
    if not methods:
        return

    labels = [METHOD_LABELS.get(m, m) for m in methods]
    mean_t_naive = [float(all_dfs[m]["time_naive_s"].mean()) for m in methods]
    mean_t_corr  = [float(all_dfs[m]["time_corr_s"].mean())  for m in methods]

    x = np.arange(len(methods))
    w = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Pannello 1: tempo medio per approccio xi ──────────────────────────────
    ax = axes[0]
    bars_n = ax.bar(x - w/2, mean_t_naive, w, color=COLOR_NAIVE, alpha=0.85,
                    label="(A) Naive $\\xi_0 = \\sigma^2_{\\mathrm{ATM}}$")
    bars_c = ax.bar(x + w/2, mean_t_corr,  w, color=COLOR_CORR,  alpha=0.85,
                    label="(B) Corretto $\\xi_0 = f_k$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Tempo medio re-pricing (s)")
    ax.set_title("Velocità re-pricing: naive vs corretto")
    ax.legend(fontsize=9)

    for bar in list(bars_n) + list(bars_c):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                f"{h:.1f}s", ha='center', va='bottom', fontsize=8.5)

    note = ("Nota: i due approcci usano lo stesso numero di\n"
            "paths MC — la differenza di tempo è solo nel\n"
            "calcolo di xi_0(t) (trascurabile vs simulazione).")
    ax.text(0.97, 0.97, note, transform=ax.transAxes, va='top', ha='right',
            fontsize=7.5, color='#555555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f8f8', alpha=0.8))

    # ── Pannello 2: scatter RMSE vs tempo (per punto = una data) ─────────────
    ax = axes[1]
    markers = {'cholesky_euler': 'o', 'hybrid_euler': 's', 'hybrid_mixed': '^'}
    for method in methods:
        df = all_dfs[method]
        t_n  = df["time_naive_s"].to_list()
        r_n  = df["rmse_naive_pp"].to_list()
        t_c  = df["time_corr_s"].to_list()
        r_c  = df["rmse_corr_pp"].to_list()
        mk   = markers.get(method, 'o')
        lbl  = METHOD_LABELS.get(method, method)

        ax.scatter(t_n, r_n, marker=mk, color=COLOR_NAIVE, alpha=0.6, s=40,
                   label=f"{lbl} (A)" if method == methods[0] else "")
        ax.scatter(t_c, r_c, marker=mk, color=COLOR_CORR,  alpha=0.6, s=40,
                   label=f"{lbl} (B)" if method == methods[0] else "")
        # linea che collega naive→corretto per stessa data
        for tn, rn, tc, rc in zip(t_n, r_n, t_c, r_c):
            ax.annotate("", xy=(tc, rc), xytext=(tn, rn),
                        arrowprops=dict(arrowstyle='->', color='#aaaaaa',
                                        lw=0.7, mutation_scale=8))

    # Aggiungi label metodo per ogni cluster
    for method in methods:
        df = all_dfs[method]
        mean_t = float(df[["time_naive_s","time_corr_s"]].mean_horizontal().mean())
        mean_r = float(df[["rmse_naive_pp","rmse_corr_pp"]].mean_horizontal().mean())
        ax.text(mean_t, mean_r, METHOD_LABELS.get(method, method),
                fontsize=7.5, ha='center', va='bottom', color='#333333',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))

    ax.set_xlabel("Tempo re-pricing (s)")
    ax.set_ylabel("RMSE (pp)")
    ax.set_title("Tradeoff accuratezza vs velocità\n(frecce: naive → corretto)")
    # Legenda manuale per approcci
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR_NAIVE,
               markersize=8, label="(A) Naive"),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOR_CORR,
               markersize=8, label="(B) Corretto"),
    ] + [
        Line2D([0],[0], marker=markers.get(m,'o'), color='#888888',
               markersize=7, label=METHOD_LABELS.get(m,m), linestyle='None')
        for m in methods
    ]
    ax.legend(handles=legend_els, fontsize=8, loc='upper right')

    fig.suptitle("Velocità re-pricing: confronto approcci xi_0 e metodi",
                 fontsize=11, fontweight='bold')
    fig.tight_layout()
    path = OUT_DIR / "figures" / "speed_xi_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"  Figure: {path.name}")


def plot_speed_methods(all_dfs: Dict[str, pl.DataFrame]) -> None:
    """
    Confronto velocità di calibrazione tra i tre metodi, letta dai CSV salvati.

    Tre pannelli:
      1. Boxplot tempo di calibrazione per metodo
      2. Timeline del tempo di calibrazione
      3. Scatter tempo calibrazione vs RMSE finale (quality vs cost)
    """
    methods = [m for m in ALL_METHODS if m in all_dfs and len(all_dfs[m]) > 0]
    if not methods:
        return

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Pannello 1: Boxplot tempo calibrazione ────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    data_calib = [all_dfs[m]["calib_sec"].drop_nulls().to_list() for m in methods]
    labels_s   = [METHOD_LABELS.get(m, m) for m in methods]
    colors_m   = [PALETTE['primary'], PALETTE['tertiary'], PALETTE['quaternary']]

    bp = ax1.boxplot(data_calib, labels=labels_s, patch_artist=True,
                     medianprops=dict(color='black', lw=2))
    for patch, col in zip(bp['boxes'], colors_m):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    # Aggiungi media
    for i, (vals, col) in enumerate(zip(data_calib, colors_m), start=1):
        if vals:
            m_val = np.mean(vals)
            ax1.plot(i, m_val, 'D', color=col, ms=7, zorder=3)
            ax1.text(i + 0.12, m_val, f"μ={m_val:.0f}s",
                     va='center', fontsize=8, color=col)

    ax1.set_ylabel("Tempo calibrazione (s)")
    ax1.set_title("Distribuzione tempi\ndi calibrazione")
    ax1.tick_params(axis='x', labelsize=8)

    # ── Pannello 2: Timeline tempo calibrazione ───────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    for method, col in zip(methods, colors_m):
        df = all_dfs[method]
        dates = [datetime.strptime(d, "%Y%m%d") for d in df["date"].to_list()]
        t_sec = df["calib_sec"].to_list()
        ax2.plot(dates, t_sec, 'o-', color=col, lw=1.5, ms=4, alpha=0.85,
                 label=METHOD_LABELS.get(method, method))

    ax2.set_ylabel("Tempo calibrazione (s)")
    ax2.set_title("Timeline tempi di calibrazione")
    ax2.legend(fontsize=8)
    ax2.tick_params(axis='x', labelrotation=25, labelsize=7)

    # Linee eventi
    events_dt = [datetime(2022,5,9), datetime(2022,11,8), datetime(2023,3,10),
                 datetime(2024,1,10), datetime(2024,4,20),
                 datetime(2024,12,5), datetime(2025,1,20)]
    for ev in events_dt:
        ax2.axvline(ev, color='#cccccc', lw=0.8, ls=':')

    # ── Pannello 3: Scatter tempo vs RMSE (quality-cost frontier) ─────────────
    ax3 = fig.add_subplot(gs[2])
    for method, col in zip(methods, colors_m):
        df = all_dfs[method]
        t_sec  = df["calib_sec"].to_list()
        rmse   = df["rmse_naive_pp"].to_list()   # RMSE con xi naive (come era calibrato)
        ax3.scatter(t_sec, rmse, color=col, alpha=0.7, s=35,
                    label=METHOD_LABELS.get(method, method))

    ax3.set_xlabel("Tempo calibrazione (s)")
    ax3.set_ylabel("RMSE finale (pp)")
    ax3.set_title("Tradeoff qualità vs costo\n(per data di calibrazione)")
    ax3.legend(fontsize=8)

    # Annotazione quadranti
    xlim, ylim = ax3.get_xlim(), ax3.get_ylim()
    xm, ym = np.mean(xlim), np.mean(ylim)
    ax3.text(xlim[0]*1.02, ym*1.05, "Veloce\nma\nimpreciso",
             fontsize=7, color='#999999', va='center')
    ax3.text(xm, ylim[0]*1.15, "Lento\nma\npreciso",
             fontsize=7, color='#999999', ha='center')

    fig.suptitle("Velocità di calibrazione: confronto tra metodi",
                 fontsize=12, fontweight='bold')
    path = OUT_DIR / "figures" / "speed_methods_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"  Figure: {path.name}")


def print_speed_table(all_dfs: Dict[str, pl.DataFrame]) -> None:
    """Stampa tabella riepilogativa tempi in log."""
    methods = [m for m in ALL_METHODS if m in all_dfs and len(all_dfs[m]) > 0]
    log.info("\n" + "="*75)
    log.info("  TABELLA VELOCITÀ")
    log.info("="*75)
    log.info(f"  {'Metodo':<22}  {'T_calib (s)':>12}  {'N_evals':>8}  "
             f"{'T_naive (s)':>12}  {'T_corr (s)':>11}  {'ΔT (ms)':>9}")
    log.info(f"  {'─'*75}")
    for method in methods:
        df  = all_dfs[method]
        tc  = float(df["calib_sec"].mean())
        ne  = float(df["n_evals"].mean())
        tn  = float(df["time_naive_s"].mean())
        tco = float(df["time_corr_s"].mean())
        dt_ms = (tco - tn) * 1000
        log.info(
            f"  {METHOD_LABELS.get(method,method):<22}  {tc:>12.1f}  {ne:>8.0f}  "
            f"{tn:>12.2f}  {tco:>11.2f}  {dt_ms:>+9.1f}"
        )
    log.info(f"  {'─'*75}")
    log.info("  ΔT = tempo(B) − tempo(A): differenza tra approccio corretto e naive")
    log.info("  (attesa vicina a zero — il calcolo di xi_0(t) è trascurabile vs MC)")


# =============================================================================
# ANALISI PER METODO
# =============================================================================

def analyse_method(
    method: str,
    dates: List[datetime],
    builder: IVSurfaceBuilder,
    n_paths: int,
) -> pl.DataFrame:
    """
    Esegue il confronto naive vs corretto per tutte le date di un metodo.
    Restituisce un DataFrame con una riga per data.
    """
    log.info(f"\n{'#'*70}")
    log.info(f"#  METODO: {METHOD_LABELS.get(method, method)}  ({len(dates)} date)")
    log.info(f"{'#'*70}")

    # Configurazione pricer
    cfg = {k: v for k, v in METHODS[method].items() if k not in ("label", "maxiter")}
    calibrator  = Calibrator(**cfg)
    pricer_cfg  = {k: cfg[k] for k in
                   ("n_steps", "scheme", "kappa", "seed", "antithetic", "pricing_method")
                   if k in cfg}

    rows       = []
    point_rows = []   # per il plot bias by maturity

    for dt in dates:
        date_str = dt.strftime("%Y%m%d")
        log.info(f"\n  {date_str}")

        # Carica parametri calibrati
        params = load_calibration_row(date_str, method)
        if params is None:
            log.warning(f"    CSV calibrazione non trovato — skip")
            continue

        H, eta, rho       = params["H"], params["eta"], params["rho"]
        calib_sec         = params["calib_sec"]
        n_evals           = params["n_evals"]
        n_points_calib    = params["n_points"]
        log.info(f"    H={H:.4f}  eta={eta:.3f}  rho={rho:.3f}  "
                 f"calib={calib_sec:.1f}s  evals={n_evals}")

        # Carica mercato
        market_data = load_market_data(dt, builder,
                                       moneyness_range=cfg.get("moneyness_range", (0.80, 1.20)))
        if market_data is None:
            log.warning(f"    Dati di mercato non disponibili — skip")
            continue

        # Estrai ATM IVs
        mats, atm_ivs = extract_atm_ivs(market_data, calibrator)
        n_mats = len(mats)
        log.info(f"    {n_mats} maturities: " +
                 ", ".join(f"T={T*252:.0f}d σ={iv*100:.1f}%" for T, iv in zip(mats, atm_ivs)))

        # Costruisci le due curve xi
        xi_naive = ForwardVarianceCurve(mats, atm_ivs**2)
        xi_corr  = ForwardVarianceCurve(mats, bootstrap_forward_variance(mats, atm_ivs))

        # Calcola forward variance bootstrap e mostra differenze
        fwd_vars = bootstrap_forward_variance(mats, atm_ivs)
        log.info(f"    {'T(d)':>6}  {'sigma_ATM':>10}  {'xi naive':>10}  "
                 f"{'xi corretto':>12}  {'diff %':>8}")
        for T, sig, fv in zip(mats, atm_ivs, fwd_vars):
            naive_v = sig**2
            log.info(f"    {T*252:>6.0f}  {sig*100:>9.2f}%  {naive_v:>10.5f}  "
                     f"{fv:>12.5f}  {(fv-naive_v)/naive_v*100:>+8.2f}%")

        # Prezzi con approccio A (naive)
        log.info(f"    Pricing (A) naive — n_paths={n_paths:,} ...")
        try:
            iv_model_n, iv_mkt_n, tau_n, t_naive = price_surface_with_xi(
                H, eta, rho, xi_naive, market_data, pricer_cfg, n_paths)
            stats_n = error_stats(iv_model_n, iv_mkt_n)
        except Exception as e:
            log.warning(f"    Pricing (A) fallito: {e}")
            continue

        # Prezzi con approccio B (corretto)
        log.info(f"    Pricing (B) corretto — n_paths={n_paths:,} ...")
        try:
            iv_model_c, iv_mkt_c, tau_c, t_corr = price_surface_with_xi(
                H, eta, rho, xi_corr, market_data, pricer_cfg, n_paths)
            stats_c = error_stats(iv_model_c, iv_mkt_c)
        except Exception as e:
            log.warning(f"    Pricing (B) fallito: {e}")
            continue

        # Maschera comune: valido in ENTRAMBI i pricing
        joint_valid = ~np.isnan(iv_model_n) & ~np.isnan(iv_model_c)
        iv_n_clean  = iv_model_n[joint_valid]
        iv_c_clean  = iv_model_c[joint_valid]
        iv_mk_clean = iv_mkt_n[joint_valid]     # iv_mkt_n == iv_mkt_c (stesso mercato)
        tau_clean   = tau_n[joint_valid]

        log.info(f"    (A) Bias={stats_n['bias_pp']:+.2f} pp  RMSE={stats_n['rmse_pp']:.2f} pp  "
                 f"t={t_naive:.2f}s  N={stats_n['n']}")
        log.info(f"    (B) Bias={stats_c['bias_pp']:+.2f} pp  RMSE={stats_c['rmse_pp']:.2f} pp  "
                 f"t={t_corr:.2f}s  N={stats_c['n']}")
        log.info(f"    Miglioramento RMSE: {stats_c['rmse_pp']-stats_n['rmse_pp']:+.2f} pp  "
                 f"ΔT={(t_corr-t_naive)*1000:+.1f}ms  N comune={joint_valid.sum()}")

        # Plot per questa data (usa array allineati)
        plot_term_structure(mats, atm_ivs, date_str, method)
        plot_scatter(iv_n_clean, iv_c_clean, iv_mk_clean, tau_clean,
                     date_str, method, stats_n, stats_c)

        # Raccogli dati per bias_by_maturity
        point_rows.append({
            "tau_days":     tau_clean * 252,
            "err_naive_pp": (iv_n_clean - iv_mk_clean) * 100,
            "err_corr_pp":  (iv_c_clean - iv_mk_clean) * 100,
        })

        rows.append({
            "date":             date_str,
            "method":           method,
            "H": H, "eta": eta, "rho": rho,
            "n_maturities":     n_mats,
            # Velocità calibrazione (dai CSV salvati)
            "calib_sec":        calib_sec,
            "n_evals":          n_evals,
            "n_points":         n_points_calib,
            "sec_per_eval":     calib_sec / max(n_evals, 1),
            # Velocità re-pricing
            "time_naive_s":     t_naive,
            "time_corr_s":      t_corr,
            "delta_time_ms":    (t_corr - t_naive) * 1000,
            # Approccio A
            "bias_naive_pp":    stats_n["bias_pp"],
            "rmse_naive_pp":    stats_n["rmse_pp"],
            "mae_naive_pp":     stats_n["mae_pp"],
            "n_naive":          stats_n["n"],
            # Approccio B
            "bias_corr_pp":     stats_c["bias_pp"],
            "rmse_corr_pp":     stats_c["rmse_pp"],
            "mae_corr_pp":      stats_c["mae_pp"],
            "n_corr":           stats_c["n"],
            # Delta accuratezza
            "delta_bias_pp":    stats_c["bias_pp"]  - stats_n["bias_pp"],
            "delta_rmse_pp":    stats_c["rmse_pp"]  - stats_n["rmse_pp"],
            "delta_mae_pp":     stats_c["mae_pp"]   - stats_n["mae_pp"],
        })

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)

    # Salva CSV per-metodo
    csv_path = OUT_DIR / "tables" / f"comparison_{method}.csv"
    df.write_csv(csv_path)
    log.info(f"\n  CSV salvato: {csv_path}")

    # Plot bias per maturity bucket
    plot_bias_by_maturity(point_rows, method)

    # Stampa tabella riepilogativa per metodo
    log.info(f"\n  {'─'*90}")
    log.info(f"  RIEPILOGO  {METHOD_LABELS.get(method, method)}")
    log.info(f"  {'Date':>10}  {'T_cal(s)':>9}  {'BiasA':>8}  {'RMSE_A':>8}  "
             f"{'BiasB':>8}  {'RMSE_B':>8}  {'ΔRMSE':>8}  {'T_A(s)':>7}  {'T_B(s)':>7}")
    log.info(f"  {'─'*90}")
    for row in df.iter_rows(named=True):
        log.info(
            f"  {row['date']:>10}  {row['calib_sec']:>9.1f}  "
            f"{row['bias_naive_pp']:>+8.2f}  {row['rmse_naive_pp']:>8.2f}  "
            f"{row['bias_corr_pp']:>+8.2f}  {row['rmse_corr_pp']:>8.2f}  "
            f"{row['delta_rmse_pp']:>+8.2f}  "
            f"{row['time_naive_s']:>7.2f}  {row['time_corr_s']:>7.2f}"
        )
    log.info(f"  {'─'*90}")
    log.info(
        f"  {'MEDIA':>10}  "
        f"{float(df['calib_sec'].mean()):>9.1f}  "
        f"{float(df['bias_naive_pp'].mean()):>+8.2f}  "
        f"{float(df['rmse_naive_pp'].mean()):>8.2f}  "
        f"{float(df['bias_corr_pp'].mean()):>+8.2f}  "
        f"{float(df['rmse_corr_pp'].mean()):>8.2f}  "
        f"{float(df['delta_rmse_pp'].mean()):>+8.2f}  "
        f"{float(df['time_naive_s'].mean()):>7.2f}  "
        f"{float(df['time_corr_s'].mean()):>7.2f}"
    )

    return df


# =============================================================================
# MAIN
# =============================================================================

def collect_dates(method: str) -> List[datetime]:
    """Raccoglie le date dai CSV di calibrazione per un dato metodo."""
    method_dir = TABLES_DIR / method
    if not method_dir.exists():
        return []
    dates = []
    for p in sorted(method_dir.glob(f"calibration_*_{method}.csv")):
        ds = p.stem.split("_")[1]
        try:
            dates.append(datetime.strptime(ds, "%Y%m%d").replace(hour=12))
        except ValueError:
            pass
    return dates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Confronto approcci xi_0: naive vs forward variance bootstrap"
    )
    parser.add_argument(
        "--method", action="append", dest="methods", choices=ALL_METHODS,
        help="Metodo da analizzare (default: tutti e tre)"
    )
    parser.add_argument(
        "--date", default=None,
        help="Data singola (YYYYMMDD); se omesso usa tutte le date disponibili"
    )
    parser.add_argument(
        "--n-paths", type=int, default=3_000,
        help="Paths MC per il re-pricing (default=3000; 5000+ per risultati più stabili)"
    )
    args = parser.parse_args()

    methods = args.methods or ALL_METHODS

    log.info("=" * 70)
    log.info("  CONFRONTO APPROCCI xi_0(t)")
    log.info("  (A) Naive:    xi_0(T_k) = sigma_ATM(T_k)^2")
    log.info("  (B) Corretto: xi_0 = varianza forward bootstrap f_k")
    log.info(f"  Metodi: {methods}")
    log.info(f"  n_paths re-pricing: {args.n_paths:,}")
    log.info("=" * 70)

    builder = IVSurfaceBuilder(str(DATA_DIR))
    builder.build_index()

    all_dfs: Dict[str, pl.DataFrame] = {}

    for method in methods:
        if args.date:
            dates = [datetime.strptime(args.date, "%Y%m%d").replace(hour=12)]
        else:
            dates = collect_dates(method)
            if not dates:
                log.warning(f"  Nessuna data trovata per {method} — skip")
                continue

        df = analyse_method(method, dates, builder, n_paths=args.n_paths)
        if len(df) > 0:
            all_dfs[method] = df

    if not all_dfs:
        log.error("Nessun risultato prodotto.")
        sys.exit(1)

    # ── Summary aggregato cross-method ────────────────────────────────────────
    summary_rows = []
    for method, df in all_dfs.items():
        summary_rows.append({
            "method":                method,
            "label":                 METHOD_LABELS.get(method, method),
            "n_dates":               len(df),
            # Accuratezza
            "mean_bias_naive_pp":    float(df["bias_naive_pp"].mean()),
            "mean_rmse_naive_pp":    float(df["rmse_naive_pp"].mean()),
            "mean_mae_naive_pp":     float(df["mae_naive_pp"].mean()),
            "mean_bias_corr_pp":     float(df["bias_corr_pp"].mean()),
            "mean_rmse_corr_pp":     float(df["rmse_corr_pp"].mean()),
            "mean_mae_corr_pp":      float(df["mae_corr_pp"].mean()),
            "mean_delta_rmse_pp":    float(df["delta_rmse_pp"].mean()),
            "mean_delta_bias_pp":    float(df["delta_bias_pp"].mean()),
            # Velocità calibrazione
            "mean_calib_sec":        float(df["calib_sec"].mean()),
            "mean_n_evals":          float(df["n_evals"].mean()),
            "mean_sec_per_eval":     float(df["sec_per_eval"].mean()),
            # Velocità re-pricing
            "mean_time_naive_s":     float(df["time_naive_s"].mean()),
            "mean_time_corr_s":      float(df["time_corr_s"].mean()),
            "mean_delta_time_ms":    float(df["delta_time_ms"].mean()),
        })

    summary_df = pl.DataFrame(summary_rows)
    summary_path = OUT_DIR / "tables" / "summary_all_methods.csv"
    summary_df.write_csv(summary_path)

    # Stampa tabella accuratezza
    log.info("\n" + "=" * 90)
    log.info("  RIEPILOGO FINALE — Accuratezza")
    log.info("=" * 90)
    log.info(f"  {'Metodo':<22}  {'N':>4}  "
             f"{'BiasA':>8}  {'RMSE_A':>8}  "
             f"{'BiasB':>8}  {'RMSE_B':>8}  "
             f"{'ΔRMSE':>8}  {'ΔBias':>8}")
    log.info(f"  {'─'*90}")
    for row in summary_df.iter_rows(named=True):
        log.info(
            f"  {row['label']:<22}  {row['n_dates']:>4}  "
            f"{row['mean_bias_naive_pp']:>+8.2f}  {row['mean_rmse_naive_pp']:>8.2f}  "
            f"{row['mean_bias_corr_pp']:>+8.2f}  {row['mean_rmse_corr_pp']:>8.2f}  "
            f"{row['mean_delta_rmse_pp']:>+8.2f}  {row['mean_delta_bias_pp']:>+8.2f}"
        )

    # Stampa tabella velocità
    log.info("\n" + "=" * 90)
    log.info("  RIEPILOGO FINALE — Velocità")
    log.info("=" * 90)
    log.info(f"  {'Metodo':<22}  {'T_calib(s)':>11}  {'N_evals':>8}  "
             f"{'s/eval':>7}  {'T_naive(s)':>11}  {'T_corr(s)':>10}  {'ΔT(ms)':>8}")
    log.info(f"  {'─'*90}")
    for row in summary_df.iter_rows(named=True):
        log.info(
            f"  {row['label']:<22}  {row['mean_calib_sec']:>11.1f}  "
            f"{row['mean_n_evals']:>8.0f}  {row['mean_sec_per_eval']:>7.3f}  "
            f"{row['mean_time_naive_s']:>11.2f}  {row['mean_time_corr_s']:>10.2f}  "
            f"{row['mean_delta_time_ms']:>+8.1f}"
        )

    # ── Figure aggregate ──────────────────────────────────────────────────────
    plot_summary_barplot(summary_df)
    plot_bias_timeline(all_dfs)
    plot_speed_xi(all_dfs)
    plot_speed_methods(all_dfs)
    print_speed_table(all_dfs)

    # CSV combinato
    combined = pl.concat(list(all_dfs.values()))
    combined.write_csv(OUT_DIR / "tables" / "comparison_all_methods.csv")

    # Verdetto
    overall_delta = float(summary_df["mean_delta_rmse_pp"].mean())
    log.info(f"\n  Delta RMSE medio complessivo: {overall_delta:+.2f} pp")
    if overall_delta < -1.0:
        log.info("  => Il bootstrap corretto migliora significativamente il fit")
    elif overall_delta < 0:
        log.info("  => Miglioramento modesto — amplificabile con ricalibrazione di (H,eta,rho)")
    else:
        log.info("  => Nessun miglioramento netto — la fonte del bias è altrove")

    log.info(f"\n  Output salvati in: {OUT_DIR.resolve()}")
