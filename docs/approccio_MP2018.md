# Approccio 3 — MP2018 (McCrickerd & Pakkanen, 2018)

## Riferimento

> McCrickerd, R. & Pakkanen, M.S. (2018).
> *Turbocharging Monte Carlo pricing for the rough Bergomi model.*
> Quantitative Finance, 18(11), 1877–1886. arXiv:1708.02563.

## Idea centrale

Il paper propone una composizione di tecniche di **variance reduction** per il pricing MC sotto il rough Bergomi, ottenendo una riduzione della varianza di ~20× a parità di costo computazionale. Le tecniche sono organizzate in una gerarchia (Tabella 1 del paper):

| Estimator     | X (signal)                    | Y (control)                   |
|---------------|-------------------------------|-------------------------------|
| Base          | $g(S_T)$                      | —                             |
| Conditional   | $\text{BS}_{\text{inv}}(F_m, K, \sigma_c)$ | —              |
| Controlled    | $g(S_T)$                      | $\text{BS}_{\text{inv}}(Q - I_V)$ |
| **Mixed**     | $\text{BS}_{\text{inv}}(F_m, K, \sigma_c)$ | $\text{BS}_{\text{inv}}(\rho^2(Q - I_V))$ |

## Parametrizzazione normalizzata

MP2018 usa il processo di Volterra normalizzato $W^\alpha_t$ con $\alpha = H - 1/2$:

$$W^\alpha_t = \sqrt{2\alpha + 1}\int_0^t (t-s)^\alpha\,dW^1_s$$

dove $\text{Var}(W^\alpha_t) = t^{2H}$ esattamente. La varianza diventa:

$$V_t = \xi_0(t)\,\exp\!\Big(\eta\,W^\alpha_t - \tfrac{1}{2}\eta^2\,t^{2H}\Big)$$

## Conditional MC (Romano-Touzi)

Condizionando sul cammino della varianza $\{V_t\}$, il log-return di $S_T/S_0$ è gaussiano:

$$\log(S_T/S_0)\,\big|\,\{V_t\} \;\sim\; \mathcal{N}(\mu_c,\;\sigma_c^2)$$

dove:

$$\mu_c = -\tfrac{1}{2}I_V + \rho\,I_{\text{corr}}, \qquad \sigma_c = \sqrt{(1-\rho^2)\,I_V}$$

con:
- $I_V = \int_0^T V_t\,dt$ (varianza integrata)
- $I_{\text{corr}} = \int_0^T \sqrt{V_t}\,dW^1_t$ (integrale stocastico spot-vol)

Il forward condizionale è $F_m = S_0\,\exp(-\tfrac{1}{2}\rho^2 I_V + \rho\,I_{\text{corr}})$ e il prezzo per-cammino è dato dalla formula di Black-Scholes applicata al payoff inverso.

Questo **elimina tutto il rumore MC legato a $W^2$**, riducendo la varianza di ~5×.

## Timer-option control variate

Il control variate usa la timer option (opzione il cui payoff dipende dalla varianza integrata):

$$Y_i = \text{BS}_{\text{inv}}\!\Big(\rho^2(Q - I_V^{(i)})\,;\; F_m^{(i)},\,K\Big)$$

dove $Q = \max_i I_V^{(i)}$ è la massima varianza integrata nel campione. Il coefficiente ottimale $\alpha^*$ è calcolato post-hoc dalla covarianza campionaria:

$$\alpha^* = -\frac{\text{Cov}(X, Y)}{\text{Var}(Y)}$$

## Antithetic sampling

Sfrutta la simmetria $(W^1, W^2) \to (-W^1, -W^2)$: si generano $n/2$ cammini base e $n/2$ cammini specchiati, dimezzando la varianza.

## Mixed Estimator (composizione finale)

La composizione di conditional MC + control variate + antithetic dà il **Mixed estimator**:

$$\hat{P}_n = \frac{1}{n}\sum_{i=1}^n \Big[X_i + \alpha^*(Y_i - \bar{Y})\Big]$$

dove $X_i$ è il prezzo condizionale (BS per-cammino) e $Y_i$ il control variate.

**Risultato**: riduzione della varianza runtime-adjusted di ~20× (Tabella 2 del paper).

## Implementazione (`src/mp2018/`)

### `volterra.py`

- `_compute_mp_weights()`: pesi del primo ordine ($\kappa = 1$) per il Hybrid Scheme
- `NormalisedVolterra`: genera cammini $W^\alpha_t$ via convoluzione
- `generate_antithetic_paths()`: sampling antitetico

### `estimators.py`

- `compute_integrated_quantities()`: calcola $I_V$ e $I_{\text{corr}}$ dai cammini
- `BaseEstimator`: MC standard $\hat{P} = \frac{1}{n}\sum g(S_T^{(i)})$
- `ConditionalEstimator`: BS condizionale per-cammino (Romano-Touzi)
- `MixedEstimator`: condizionale + control variate con $\alpha^*$ ottimale

### `pricer.py`

- `rBergomiPricerMP`: pricer unificato con estimator selezionabile (`'base'`, `'conditional'`, `'mixed'`)
- Supporto antithetic, multi-strike, implied vol

### `calibrator.py`

- `CalibratorMP`: calibrazione con estimator MP2018 (DE + NM)

## Punti di forza

- **Massima accuratezza**: RMSE medio più basso (23.2pp su 7 date)
- **SE ~10× inferiore**: da $\sim 10^{-3}$ (base) a $\sim 10^{-4}$ (mixed)
- **Stabilità numerica superiore**: nessun overflow anche con $\eta > 4$, $H \to 0$
- **Eccellente in regimi estremi**: FTX collapse (42pp vs 81–91pp), bull run 2025 (7pp vs 33–69pp)

## Limitazioni

- **Lento**: ~40s per superficie (overhead del conditional BS per-cammino via `scipy.stats.norm`)
- **Non adatto alla calibrazione**: 300 eval × 40s = ~3.3 ore per una singola calibrazione DE
- **Overhead Python**: le funzioni BS vettorizzate con `np.vectorize` non sfruttano la parallelizzazione

## Risultati empirici (media su 7 date)

| Metrica           | Valore     |
|-------------------|------------|
| RMSE medio        | 18.9pp     |
| Tempo medio       | 39.6s      |
| SE medio          | ~$3.2 \times 10^{-4}$ |
| Vittorie (best RMSE) | 3/7    |

## File

```
src/mp2018/
├── __init__.py
├── volterra.py      # NormalisedVolterra (κ=1 Hybrid Scheme)
├── estimators.py    # Base, Conditional, Mixed estimators
├── pricer.py        # rBergomiPricerMP (estimator selezionabile)
└── calibrator.py    # CalibratorMP (DE + NM con Mixed estimator)
```
