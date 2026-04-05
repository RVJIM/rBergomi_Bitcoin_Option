# Approccio 1 — rBergomi (Codice Tesi)

## Modello

Il rough Bergomi (rBergomi) è il modello base della tesi. La dinamica della varianza è guidata da un moto browniano frazionario (fBm) con parametro di Hurst $H \in (0, 0.5)$:

$$V_t = \xi_0(t) \exp\!\Big(\eta\, Z^H_t - \tfrac{1}{2}\eta^2\, t^{2H}\Big)$$

dove $Z^H_t$ è il fBm di Riemann-Liouville e lo spot evolve come:

$$\frac{dS_t}{S_t} = \sqrt{V_t}\,\big(\rho\,dW^1_t + \sqrt{1-\rho^2}\,dW^2_t\big)$$

## Simulazione

Lo schema implementato nel codice della tesi (`src/rbergomi/pricer.py`) usa lo **Hybrid Scheme** di Bennedsen, Lunde & Pakkanen (2017) con le seguenti caratteristiche:

- **Decomposizione del kernel**: il kernel di Volterra $(t-s)^{H-1/2}$ è diviso in componente *recente* (variance-matched) e *distante* (integral-averaged), con parametro di troncamento $\kappa$.
- **Discretizzazione Euler-Maruyama** per il processo spot (log-Euler per stabilità numerica).
- **Numba JIT compilation**: i kernel sono ottimizzati con `@numba.njit` per prestazioni native.
- **Schema Cholesky alternativo**: disponibile come fallback ma più lento ($O(n^2)$ vs $O(n \log n)$).

## Opzioni Inverse BTC

I payoff delle opzioni inverse (Deribit) sono denominati in BTC:

- **Put inversa**: $\max(K/S_T - 1, 0)$
- **Call inversa**: $\max(1 - K/S_T, 0)$

Il pricing avviene tramite Monte Carlo standard senza variance reduction.

## Calibrazione

La pipeline di calibrazione (`src/rbergomi/calibrator.py`) ottimizza $(H, \eta, \rho)$ congiuntamente:

1. **Forward variance** $\xi_0(t)$: stimata dalle IV ATM delta-neutrali e fissata
2. **Differential Evolution** (globale): esplora lo spazio dei parametri
3. **Nelder-Mead** (locale): raffina la soluzione
4. **Loss function**: RMSE pesato per maturità delle IV (in pp)

### Bounds

| Parametro | Range |
|-----------|-------|
| $H$       | $[0.01, 0.49]$ |
| $\eta$    | $[0.3, 4.0]$ |
| $\rho$    | $[-0.99, 0.0]$ |

### Quality flags

| RMSE (pp)  | Flag        |
|------------|-------------|
| $< 10$     | good        |
| $< 20$     | acceptable  |
| $< 30$     | borderline  |
| $\geq 30$  | poor        |

## Punti di forza

- **Implementazione completa e testata** su dati reali BTC Deribit
- **Numba JIT** per buone prestazioni (~14s per superficie)
- **Pipeline modulare**: forward variance → ottimizzazione → diagnostica

## Limitazioni

- **Nessuna variance reduction**: errore MC relativamente alto (SE ~ $10^{-3}$)
- **Instabilità numerica** con parametri estremi ($\eta > 4$, $H \to 0$): possibili overflow
- **Velocità intermedia**: troppo lento per DE estese, troppo impreciso per pricing finale

## Risultati empirici (media su 7 date)

| Metrica           | Valore     |
|-------------------|------------|
| RMSE medio        | 23.3pp     |
| Tempo medio       | 8.6s       |
| SE medio          | ~$1.5 \times 10^{-3}$ |
| Vittorie (best RMSE) | 2/7    |

## File

```
src/rbergomi/
├── __init__.py
├── pricer.py           # rBergomiPricer (Hybrid + Cholesky)
├── calibrator.py       # Calibrator (DE + NM, quality flags)
├── utils.py            # BS helpers, IV inversion, validation
└── mixed_estimator.py  # (Mixed estimator, non usato qui)
```
