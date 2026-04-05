# Approccio 2 — BLP2017 (Bennedsen, Lunde & Pakkanen, 2017)

## Riferimento

> Bennedsen, M., Lunde, A. & Pakkanen, M.S. (2017).
> *Hybrid scheme for Brownian semistationary processes.*
> Finance and Stochastics, 21(4), 931–965. arXiv:1507.03004.

## Idea centrale

Il paper introduce uno schema di simulazione ibrido per processi semi-stazionari browniani (BSS), di cui il rough Bergomi è un caso speciale. L'innovazione chiave è la **decomposizione del kernel di Volterra** in due componenti trattate con metodi diversi:

$$\hat{Z}^H_{t_i} = \underbrace{\int_{t_{i-\kappa}}^{t_i} (t_i - s)^{H-1/2}\,dW^1_s}_{\text{Recente: variance-matched}} + \underbrace{\sum_{l=\kappa+1}^{i} b_l\,(W^1_{t_{i-l+1}} - W^1_{t_{i-l}})}_{\text{Distante: integral-averaged}}$$

### Componente recente ($l \leq \kappa$)

Per i passi temporali più vicini, il kernel singolare viene approssimato con pesi **variance-matched**: il peso $c_l$ è scelto in modo che la varianza marginale del contributo recente sia esatta:

$$c_l = \sqrt{\mathbb{E}\Big[\Big(\int_{t_{i-l}}^{t_{i-l+1}} (t_i - s)^{H-1/2}\,dW^1_s\Big)^2\Big]}$$

Questo cattura la singolarità del kernel $(t-s)^{H-1/2}$ quando $s \to t$.

### Componente distante ($l > \kappa$)

Per i passi lontani, il kernel è approssimato con la media integrale:

$$b_l = \frac{1}{\Delta}\int_{(l-1)\Delta}^{l\Delta} (l\Delta - u + \Delta)^{H-1/2}\,du$$

Il parametro $\kappa$ controlla il confine tra le due regioni. Per $\kappa = 1$ (usato in questo lavoro, come raccomandato da MP2018), solo il passo più recente usa il trattamento esatto.

## Proprietà dello schema

- **Convergenza forte**: ordine $H + 1/2 - \epsilon$ (BLP2017, Teorema 3.3)
- **Complessità**: $O(n \log n)$ tramite FFT per la convoluzione distante
- **Varianza marginale esatta**: $\text{Var}(\hat{Z}^H_t) = t^{2H}$ per costruzione

## Implementazione (`src/blp2017/`)

### `hybrid_scheme.py`

- `compute_hybrid_weights()`: pre-calcola i pesi recenti e distanti
- `VolterraFBM`: genera i cammini del fBm di Volterra via convoluzione
- `HybridSchemeBLP`: pipeline completa fBm → varianza → spot

### `pricer.py`

- `rBergomiPricerBLP`: pricer MC puro (nessuna variance reduction)
- Metodi: `price_inverse_put`, `price_inverse_call`, `price_multiple_strikes`
- Payoff inversi BTC: $\max(K/S_T - 1, 0)$ e $\max(1 - K/S_T, 0)$

### `calibrator.py`

- `CalibratorBLP`: ri-usa l'infrastruttura di calibrazione del codice tesi
- Forward variance da ATM IV, DE + Nelder-Mead, quality flags

## Punti di forza

- **Velocità eccezionale**: ~0.2s per superficie (~50× più veloce del codice tesi)
- L'assenza di overhead JIT (puro NumPy) rende il cold-start immediato
- **Ideale per la calibrazione**: permette centinaia di valutazioni della loss function in pochi secondi
- **Fedele al paper originale**: implementazione diretta delle equazioni BLP2017

## Limitazioni

- **Nessuna variance reduction**: SE comparabile al codice tesi (~$10^{-3}$)
- **Instabilità con parametri estremi**: overflow con $\eta > 4$ (stesso problema del codice base)
- **Meno preciso per il pricing finale**: l'errore MC puro limita l'accuratezza raggiungibile

## Risultati empirici (media su 7 date)

| Metrica           | Valore     |
|-------------------|------------|
| RMSE medio        | 22.9pp     |
| Tempo medio       | 0.2s       |
| SE medio          | ~$2.5 \times 10^{-3}$ |
| Vittorie (best RMSE) | 2/7    |

## File

```
src/blp2017/
├── __init__.py
├── hybrid_scheme.py   # VolterraFBM, HybridSchemeBLP, convergence_analysis
├── pricer.py          # rBergomiPricerBLP (MC puro, no VR)
└── calibrator.py      # CalibratorBLP (DE + NM)
```
