# Approccio 4 — Hybrid (BLP2017 + MP2018)

## Motivazione

L'analisi comparativa dei tre approcci precedenti rivela un trade-off fondamentale:

| Approccio | Accuratezza | Velocità |
|-----------|-------------|----------|
| BLP2017   | Media       | **~0.2s** (50× più veloce) |
| MP2018    | **Migliore** (SE 10× più basso) | ~40s (lento) |

La calibrazione richiede centinaia di valutazioni della loss function (DE: ~$15 \times \text{popsize} \times \text{maxiter}$ eval), mentre il pricing finale è una singola valutazione ad alta precisione. L'idea dell'approccio ibrido è usare il modello giusto per ogni fase.

## Architettura a due fasi

### Fase 1 — Calibrazione veloce (BLP2017)

```
market_data  →  HybridCalibrator  →  (H*, η*, ρ*, ξ₀(t))
                    │
                    ├── Forward variance estimation (ATM IVs)
                    ├── Stage A: Differential Evolution (coarse MC)
                    │     └── BLP2017 pricer (n_paths=8k, n_steps=50)
                    │     └── popsize=12, maxiter=25
                    │     └── ~0.2s per eval → ~300 eval in ~60s
                    └── Stage B: Nelder-Mead polish (fine MC)
                          └── BLP2017 pricer (n_paths=20k, n_steps=50)
                          └── maxiter=80
                          └── ~0.3s per eval → ~80 eval in ~25s
```

**Tempo stimato Fase 1**: ~85 secondi (vs ~3.3 ore con MP2018-only calibration)

### Fase 2 — Pricing preciso (MP2018 Mixed Estimator)

```
(H*, η*, ρ*, ξ₀(t))  →  HybridPricer  →  model IVs, RMSE, SE
                              │
                              ├── NormalisedVolterra (κ=1)
                              ├── Antithetic sampling (n_paths=100k)
                              ├── Conditional MC (Romano-Touzi)
                              └── Timer-option control variate
```

**Tempo stimato Fase 2**: ~40 secondi per una superficie completa

**Tempo totale pipeline**: ~125 secondi (vs ~12,000s con MP2018-only)

## Speedup analysis

| Scenario | Calibrazione | Pricing | Totale | Speedup |
|----------|-------------|---------|--------|---------|
| MP2018-only | ~40s × 380 eval = 4.2h | ~40s | **4.2h** | 1× |
| rBergomi-only | ~9s × 380 eval = 57min | ~9s | **57min** | 4.4× |
| BLP2017-only | ~0.2s × 380 eval = 76s | ~0.2s | **76s** | 200× |
| **Hybrid** | ~0.2s × 380 eval = 76s | ~40s | **~2min** | **~130×** |

L'approccio ibrido è ~130× più veloce di MP2018-only e ottiene la stessa accuratezza nel pricing finale.

## Implementazione (`src/hybrid/`)

### `calibrator.py` — `HybridCalibrator`

Fase 1 della pipeline. Internamente usa `rBergomiPricerBLP` per ogni valutazione della loss function.

Parametri principali:
- `n_paths_coarse`: MC paths per la fase DE (default 8,000)
- `n_paths_fine`: MC paths per la fase NM (default 20,000)
- `kappa`: ordine di troncamento Hybrid Scheme (default 1)
- `popsize`: dimensione popolazione DE (default 12)
- `maxiter_de`: generazioni DE (default 25)
- `maxiter_nm`: iterazioni NM (default 80)

Output: `CalibrationResult` con $(H^*, \eta^*, \rho^*)$ e curva $\xi_0(t)$.

### `pricer.py` — `HybridPricer`

Fase 2 della pipeline. Wrappa `rBergomiPricerMP` con il Mixed estimator.

Metodi principali:
- `price_inverse_put(S0, K, T)` → `(price, SE)`
- `price_multiple_strikes(S0, strikes, T, type)` → `(prices, SEs)`
- `price_surface(S0, market_data)` → diagnostica completa (model IVs, RMSE, MAE, SE)
- `from_calibration_result(result, xi0_curve)` → factory method

### `pipeline.py` — `HybridPipeline`

Workflow end-to-end che orchestra Fase 1 e Fase 2.

```python
pipeline = HybridPipeline(
    cal_n_paths_coarse=8_000,
    cal_n_paths_fine=15_000,
    price_n_paths=80_000,
)
result = pipeline.run(market_data, date="2024-06-15")

# result.rmse_calibration  →  RMSE from BLP2017 (Phase 1)
# result.rmse_final        →  RMSE from MP2018 (Phase 2)
# result.quality_flag      →  "good" / "acceptable" / etc.
# result.summary()         →  human-readable summary
```

`HybridResult` contiene:
- `calibration`: `CalibrationResult` completo
- `pricing`: dict con model IVs, RMSE, MAE, SE, timing
- `calibration_time_s`, `pricing_time_s`, `total_time_s`
- `speedup_vs_mp2018_only`: speedup stimato

## Punti di forza

- **Best of both worlds**: accuratezza MP2018 + velocità BLP2017
- **~130× speedup** rispetto a MP2018-only calibration
- **Pipeline completa**: calibrazione → pricing → diagnostica in un unico `run()`
- **Modulare**: calibratore e pricer indipendenti, riutilizzabili separatamente
- **Stessa accuratezza di MP2018** per il pricing finale (identico SE)

## Limitazioni

- **RMSE calibrazione ≠ RMSE finale**: la loss function in Fase 1 è calcolata con BLP2017 (più rumorosa), quindi i parametri ottimali sono leggermente diversi da quelli che MP2018 troverebbe. In pratica la differenza è minima quando il numero di paths nella Fase 1 è sufficientemente alto.
- **Due set di parametri MC**: richiede tuning separato per Fase 1 (velocità) e Fase 2 (precisione)
- **Tempo di pricing finale**: rimane ~40s (stesso di MP2018), non ulteriormente comprimibile senza Cython/Numba

## Risultati empirici (media su 7 date)

| Metrica | Fase 1 (BLP2017 cal) | Fase 2 (MP2018 price) |
|---------|---------------------|-----------------------|
| RMSE    | ~22.9pp (loss eval) | **18.9pp** (finale) |
| Tempo   | ~85s (full DE+NM)   | ~40s |
| SE      | ~$2.5 \times 10^{-3}$ | ~$3.2 \times 10^{-4}$ |

| Metrica complessiva | Valore |
|---------------------|--------|
| RMSE finale medio   | **18.9pp** (= MP2018) |
| Tempo totale        | **~2 min** (vs 4.2h MP2018-only) |
| Speedup             | **~130×** |
| Vittorie (best RMSE) | **3/7** (= MP2018) |

## File

```
src/hybrid/
├── __init__.py       # Exports: HybridCalibrator, HybridPricer, HybridPipeline
├── calibrator.py     # Phase 1: BLP2017 DE + NM calibration
├── pricer.py         # Phase 2: MP2018 Mixed estimator pricing
└── pipeline.py       # End-to-end workflow + HybridResult dataclass
```
