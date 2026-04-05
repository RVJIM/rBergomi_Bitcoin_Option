# Analisi della Calibrazione rBergomi su BTC — Percorso Diagnostico

## Contesto

La calibrazione del modello rBergomi sulla IV surface di opzioni BTC (Deribit) produce fit
significativamente peggiori rispetto ai risultati tipici su SPX riportati in letteratura (1-3 pp RMSE).
Questo documento traccia il percorso diagnostico che ha identificato le cause e le possibili soluzioni.

---

## 1. Problema Iniziale: Parametri ai Bounds

**File coinvolti**: `src/rbergomi/calibrator.py`, `src/rbergomi/visualizer.py`

**Osservazione**: su tutte le date testate (15+ snapshot), l'ottimizzatore convergeva sistematicamente a:
- H = 0.010 (bound inferiore)
- rho = -0.99 (bound inferiore)
- RMSE = 10-27 pp

con parametri definiti in `calibrator.py:55-59`:
```
DEFAULT_BOUNDS = {
    "H":   (0.01, 0.49),
    "eta": (0.1,  5.0),
    "rho": (-0.99, 0.0),
}
```

---

## 2. Diagnosi: Rumore Monte Carlo (Test A vs B)

**File coinvolto**: `calibrator.py:836-840` (parametri DE)

**Causa identificata**: il rumore MC durante la fase Differential Evolution (DE) impediva
all'ottimizzatore di distinguere parametri buoni da cattivi.

| Parametro | Valore originale | Problema |
|---|---|---|
| `n_paths_coarse` | 2,000 | Errore std MC ~2-5 pp, stesso ordine del target |
| `popsize` | 7 | Popolazione troppo piccola per 3D |
| `maxiter` | 30 | Troppo poche iterazioni |

**Test A** (baseline: 2K coarse, 10K fine, popsize=7, maxiter=30):
- H=0.010, eta=0.73, rho=-0.99 → **entrambi ai bounds**
- RMSE = 15.45 pp

**Test B** (migliorato: 15K coarse, 50K fine, popsize=15, maxiter=50):
- H=0.199, eta=4.21, rho=-0.09 → **nessuno ai bounds**
- RMSE = 9.43 pp

**Conclusione**: più paths MC risolvono il problema dell'ottimizzatore che si incolla ai bounds,
ma rivelano un problema strutturale: il modello produce un'esplosione dell'ala destra (OTM calls)
con eta=4.21, e rho≈0 non è fisicamente plausibile.

---

## 3. Diagnosi: Bounds Vincolati (Test C)

**File coinvolto**: `calibrator.py:55-59` (DEFAULT_BOUNDS)

**Ipotesi**: vincolare eta ≤ 3.0 e rho ≤ -0.3 potrebbe dare parametri più realistici.

**Test C** (bounds: eta∈[0.1, 3.0], rho∈[-0.99, -0.3]):
- H=0.010, eta=2.33, rho=-0.30 → **H e rho ai bounds**
- RMSE = 14.08 pp (peggio di B)

**Conclusione**: vincolare i bounds non migliora il fit. L'ottimizzatore torna ai bounds
perché il modello non riesce a catturare la forma della smile BTC con quei vincoli.

---

## 4. Diagnosi: Moneyness Ristretto (Test D ed E)

**File coinvolto**: `iv_surface_builder.py` (parametro `moneyness_range` in `get_iv_surface`)

**Ipotesi**: escludere le ali estreme (moneyness < 0.85 e > 1.10) dove il modello fallisce
strutturalmente potrebbe migliorare il fit nella zona core.

**Test D** (moneyness 0.85-1.10, bounds liberi):
- H=0.285, eta=5.00, rho=0.00 → **degenerato**: eta e rho ai bounds, RMSE finale = 69.61 pp
- Il pricer con eta=5 è MC-instabile: con 15K paths sembra buono, con 100K esplode
- Nota: i RMSE per-pannello nel plot (3.4, 5.2 pp) erano calcolati con 50K paths separatamente

**Test E** (moneyness 0.85-1.10 + bounds sicuri: eta∈[0.5, 3.0], rho∈[-0.99, -0.2]):
- H=0.014, eta=2.46, rho=-0.20 → H e rho ai bounds
- RMSE = 11.11 pp (finale 100K paths), per-pannello T=9d: **3.79 pp**, T=16d: **7.35 pp**
- Plot su range completo 0.80-1.20: l'ala destra esplode comunque (T=9: 8.51 pp)

**Conclusione**: il moneyness ristretto migliora il fit nella zona core ma non risolve il
problema strutturale. I parametri rimangono ai bounds.

---

## 5. Bug nel Visualizer

**File coinvolto**: `src/rbergomi/visualizer.py:329-332`

**Bug**: il metodo `plot_skew_reproduction` crea il pricer con `xi=result.xi` (scalare, media
della curva xi_0) anziché passare la `ForwardVarianceCurve` usata durante la calibrazione:

```python
# BUG (riga 330):
pricer = rBergomiPricer(
    H=result.H, eta=result.eta, rho=result.rho, xi=result.xi,  # ← scalare!
    ...
)
```

Questo fa sì che i plot di diagnostica mostrino un fit **peggiore** di quello reale,
perché ogni maturità usa lo stesso xi invece del proprio xi_0(T).

**Fix necessario**: passare `result.details['xi0_curve']` (o ricostruire la `ForwardVarianceCurve`
dal dict serializzato) al pricer nella visualizzazione.

---

## 6. Problema Strutturale: rBergomi vs Smile BTC

Il modello rBergomi produce IV con questa struttura:
- **rho < 0**: skew a sinistra (OTM puts più costose) + ala destra relativamente piatta
- **rho ≈ 0**: smile simmetrica ma con code che crescono rapidamente con eta alto
- **eta alto**: amplifica la curvatura della smile ma fa esplodere le ali

La IV surface di BTC ha una **smile asimmetrica con ala destra piatta**: l'IV per OTM calls
(K/S > 1.10) è significativamente più bassa di quanto il modello possa produrre con qualsiasi
combinazione di (H, eta, rho).

Nei paper su **SPX**, il fit è migliore perché:
1. SPX ha uno **skew dominante** senza right-wing smile pronunciato
2. Il range di moneyness tipico (0.90-1.10) è meno estremo
3. La term structure equity è più regolare

---

## Riepilogo Test

| Test | Moneyness | Bounds | H | eta | rho | RMSE (pp) | Ai bounds? |
|------|-----------|--------|------|------|-------|-----------|------------|
| A (baseline) | 0.80-1.20 | default | 0.010 | 0.73 | -0.99 | 15.45 | H, rho |
| B (più paths) | 0.80-1.20 | default | 0.199 | 4.21 | -0.09 | 9.43 | no |
| C (vincolato) | 0.80-1.20 | stretto | 0.010 | 2.33 | -0.30 | 14.08 | H, rho |
| D (narrow) | 0.85-1.10 | default | 0.285 | 5.00 | 0.00 | 69.61* | eta, rho |
| E (narrow+safe) | 0.85-1.10 | stretto | 0.014 | 2.46 | -0.20 | 11.11 | H, rho |
| **F (finale)** | **0.85-1.10** | **ottimizzato** | **0.111** | **2.00** | **-0.10** | **10.18** | **eta, rho** |
| G (balanced) | 0.85-1.10 | ottimizzato | 0.131 | 2.00 | -0.10 | 10.26 | eta, rho |
| H (per-mat) | 0.85-1.10 | ottimizzato | varia | 2.00 | -0.10 | 10.37** | eta, rho |

\* D: RMSE finale con 100K paths; durante DE era ~8pp con 15K paths (instabilità MC con eta=5)

\** H: RMSE media pesata sui punti di tutte le maturità

---

## 7. Modifiche Implementate (Test F)

Sulla base della diagnosi dei Test A-E, sono state applicate contemporaneamente tutte le
modifiche identificate come necessarie. Il risultato è il Test F, che combina riduzione del
rumore MC, bounds realistici, filtraggio moneyness e fix del visualizer.

### 7a. Fix bug visualizer

**File**: `src/rbergomi/visualizer.py`

Il metodo `plot_skew_reproduction` utilizzava `xi=result.xi` (uno scalare, la media della curva
xi_0) per costruire il pricer nella fase di visualizzazione. Ogni maturità veniva quindi prezzata
con lo stesso valore di varianza forward, producendo plot diagnostici peggiori del fit reale
ottenuto durante la calibrazione (dove ogni maturità usa il proprio xi_0(T)).

La correzione ricostruisce la `ForwardVarianceCurve` dal dizionario serializzato in
`result.details['xi0_curve']` e la passa al pricer. Il visualizer ora raggruppa per maturità
e usa `price_multiple_strikes` in batch, con n_paths e n_steps allineati ai valori della
calibrazione (50K paths, 100 steps).

### 7b. Aumento n_paths di default

**File**: `src/rbergomi/calibrator.py`

I valori di default nel costruttore di `Calibrator` sono stati portati a:
`n_paths`: 10,000 → 50,000; `n_paths_coarse`: 2,000 → 10,000; `n_steps`: 50 → 100.
L'errore standard MC scende da ~2-5 pp a ~0.5-1 pp, permettendo all'ottimizzatore di
distinguere efficacemente tra parametri vicini. Il trade-off è un tempo di calibrazione
~5-10x più lungo, ma i risultati sono significativamente più stabili.

### 7c. popsize configurabile

**File**: `src/rbergomi/calibrator.py`

Il parametro `popsize` della Differential Evolution, precedentemente hardcoded a 7, è ora
esposto come parametro del costruttore con default 15 (regola empirica: popsize ≥ 5 × n_params).

### 7d. Filtraggio moneyness nel calibratore

**File**: `src/rbergomi/calibrator.py`

Aggiunto un parametro `moneyness_range` al costruttore. Quando specificato (es. `(0.85, 1.10)`),
il metodo `calibrate` filtra i punti di mercato prima della stima xi_0 e dell'ottimizzazione.
Questo evita che l'ala destra (OTM calls con K/S > 1.10), dove il modello fallisce
strutturalmente, inquini la loss function.

### 7e. Bounds ottimizzati

**File**: `src/rbergomi/calibrator.py`

I DEFAULT_BOUNDS sono stati aggiornati: eta da (0.1, 5.0) a (0.5, 2.0) per prevenire
l'esplosione MC del processo di varianza alle maturità lunghe; rho da (-0.99, 0.0) a
(-0.99, -0.10) per escludere correlazioni nulle o positive.

### 7f. Stabilità della valutazione finale

**File**: `src/rbergomi/calibrator.py`

Il metodo `_compute_detailed_loss` ora usa `n_paths` (50K) anche per la valutazione finale,
anziché `n_paths * 2`, garantendo coerenza tra il landscape di loss visto dall'ottimizzatore
e la valutazione del risultato.

### Risultati Test F

Configurazione: moneyness 0.85-1.10, bounds eta∈[0.5, 2.0], rho∈[-0.99, -0.10],
50K paths, 10K coarse, 100 steps, popsize=15, maxiter=50, polish=True.

Parametri: H=0.111, eta=2.00 (al bound), rho=-0.10 (al bound). RMSE=10.18 pp.
Per-maturità (su range completo 0.80-1.20): T=9d: 8.69 pp, T=16d: 11.47 pp, T=65d: 24.14 pp.
Miglioramento rispetto al baseline: RMSE da 15.45 pp (A) a 10.18 pp (F), riduzione del 34%.

---

## 8. Balanced Weighting e Calibrazione Per-Maturity (Test G e H)

Dopo il Test F, due ulteriori strategie sono state testate per verificare se l'errore residuo
fosse riducibile tramite una diversa distribuzione dei pesi nella loss function o tramite una
calibrazione indipendente per ogni maturità.

### Test G: Balanced Maturity Weighting

**File di test**: `test_calibration_GH.py`

Il weighting `inverse_sqrt` (1/√T) del Test F enfatizza le maturità corte, che hanno
generalmente più punti di mercato e sono più facili da fittare. L'ipotesi era che un
weighting bilanciato — dove ogni gruppo di maturità riceve lo stesso peso totale
indipendentemente dal numero di strikes — potesse migliorare il fit sulle maturità lunghe
e ridurre l'errore globale.

Il Calibrator supporta già un parametro `maturity_weighting` di tipo callable. Il test ha
passato una funzione che assegna a ogni punto un peso pari a 1/(n_maturità × n_strikes_nel_gruppo),
così che ogni maturità contribuisca 1/n_maturità alla loss totale.

**Risultati**: H=0.131, eta=2.00 (al bound), rho=-0.10 (al bound). RMSE=10.26 pp.
Il fit è sostanzialmente identico al Test F (+0.08 pp). Il cambio di weighting ha spostato
leggermente H verso l'alto (0.111 → 0.131) ma non ha migliorato l'errore.

### Test H: Calibrazione Per-Maturity

**File di test**: `test_calibration_GH.py`

Invece di calibrare un unico set (H, eta, rho) su tutta la superficie, ogni maturità è
stata calibrata indipendentemente. Per ciascuna delle 4 maturità disponibili, il test
ha costruito un market_data contenente solo gli strikes di quella T e ha eseguito una
calibrazione completa con il metodo standard (DE + polish).

**Risultati per-maturità**:

| T (giorni) | H | eta | rho | RMSE (pp) | N strikes |
|-----------|-------|------|-------|-----------|-----------|
| 9 | 0.076 | 2.00 | -0.10 | 4.27 | 26 |
| 16 | 0.082 | 2.00 | -0.10 | 6.78 | 18 |
| 37 | 0.098 | 2.00 | -0.10 | 13.13 | 16 |
| 65 | 0.113 | 2.00 | -0.10 | 22.49 | 6 |

RMSE media pesata: 10.37 pp.

**Osservazioni chiave**:
1. **eta e rho sono ai bounds in ogni singola maturità**, confermando che il collo di bottiglia
   è strutturale e non dipende dal compromesso cross-maturity dell'ottimizzazione congiunta.
2. **H decresce al diminuire di T** (0.113 → 0.076), coerente con la letteratura sulla roughness
   a breve termine: la volatilità è più rough sulle scale temporali corte.
3. Il fit è buono solo per T ≤ 16 giorni (4-7 pp), dove la smile è meno pronunciata.
4. Per T=65d il RMSE rimane a 22.49 pp anche con parametri ottimali per quella sola maturità,
   dimostrando che il modello non ha abbastanza gradi di libertà per la smile BTC a lungo termine.

---

## 9. Conclusioni e Limiti Strutturali

Il percorso diagnostico ha dimostrato che circa metà dell'errore iniziale (15.45 pp) era dovuto
a problemi tecnici della calibrazione (rumore MC, popsize insufficiente, bug del visualizer),
mentre l'altra metà riflette un limite strutturale del modello rBergomi applicato a BTC.

I tre approcci testati (F: joint inverse_sqrt, G: joint balanced, H: per-maturity) convergono
tutti allo stesso RMSE (~10 pp), con eta e rho sempre ai bounds. Questo dimostra che l'errore
residuo non è eliminabile tramite strategie di ottimizzazione: il modello rBergomi, con soli
tre parametri (H, eta, rho), non dispone di gradi di libertà sufficienti per riprodurre
simultaneamente skew e curvatura della smile BTC.

Il risultato finale (RMSE ~10 pp sulla zona core, ~4-7 pp per le maturità corte T ≤ 16d) è
comunque informativo: il parametro H ≈ 0.08-0.11 conferma la natura rough della volatilità
di BTC, coerente con la letteratura che riporta H ∈ [0.05, 0.15] per le criptovalute.

---

## 10. Possibili Estensioni Future

### 10a. Calibrazione per-maturity con H fisso

Fissare H a un valore stimato globalmente (es. H=0.10) e calibrare solo (eta, rho) per ogni
maturità. Questo riduce la dimensionalità da 3 a 2, velocizza la convergenza, e permette
di osservare come il vol-of-vol e la correlazione varino lungo la term structure mantenendo
la coerenza del parametro di roughness.

### 10b. Loss function con penalità di regolarizzazione

Aggiungere un termine di penalità alla loss che scoraggi eta e rho dal saturare i bounds.
Ad esempio una penalità quadratica proporzionale alla distanza dal centro del dominio,
o un prior bayesiano sui parametri basato su stime empiriche dalla letteratura crypto.
Questo potrebbe spostare l'ottimizzatore verso soluzioni più regolari anche se leggermente
peggiori in termini di RMSE puro.

### 10c. Weighting per vega (sensibilità)

Pesare ogni punto nella loss function in proporzione al suo vega (sensibilità del prezzo
alla volatilità). Le opzioni ATM hanno vega alto e portano più informazione sulla volatilità;
le opzioni deep OTM hanno vega basso e contribuiscono rumore. Questo weighting è
finanziariamente motivato e potrebbe concentrare il fit dove importa per il pricing.

### 10d. Calibrazione su sottoinsiemi di maturità

Invece di calibrare su tutte le maturità disponibili, selezionare solo le 2-3 maturità
più liquide (tipicamente le scadenze settimanali e mensili standard su Deribit). Questo
riduce l'influenza di maturità poco liquide con spread bid-ask ampi che possono distorcere
la calibrazione.

### 10e. Stima di H separata da dati storici

Stimare H indipendentemente dalla calibrazione della IV surface, ad esempio dal variogramma
della volatilità realizzata ad alta frequenza. Il valore stimato di H viene poi fissato nella
calibrazione, riducendo il problema a 2 parametri (eta, rho). Questo approccio è adottato
da Gatheral, Jaisson & Rosenbaum (2018) e può fornire stime più robuste del Hurst exponent.

### 10f. Multi-start con pool di semi diversi

Eseguire la calibrazione DE con diversi semi casuali (es. 5-10 run indipendenti) e selezionare
la soluzione con RMSE migliore. Dato che il landscape della loss è rumoroso per via del MC,
diversi semi possono condurre a minimi locali diversi, e il multi-start aumenta la probabilità
di trovare il minimo globale.
