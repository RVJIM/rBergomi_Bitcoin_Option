# LaTeX Context: Chapter 1 — The Crypto Derivatives Market and Inverse Option Mechanics
**File**: LaTeX/Chapter_1.tex (450 lines) | **Status**: COMPLETE DRAFT

## Structure

### 1.1 Bitcoin as an Emerging Asset Class
**§1.1.1 Genesis and Early Development** — COMPLETE
Bitcoin conceived during 2008 crisis [nakamoto2008]. Genesis block Jan 3 2009 with Times headline. First years: negligible value, Mt. Gox established 2010, pizza transaction. Early adopters: cypherpunks, libertarians. By 2013 reached $1000, attracted mainstream attention.

**§1.1.2 From Peer-to-Peer Cash to Digital Gold** — COMPLETE
Narrative shift from payment system to store of value. Volatility prevents medium-of-exchange use [yermack2014]. "Digital gold" framing gained traction 2016+. Institutional adoption: MicroStrategy, Tesla (2020-21). Scarcity (21M cap) + halving cycle parallels gold mining.

**§1.1.3 Institutional Adoption and Market Maturation** — COMPLETE
Three-phase institutional adoption: (1) 2017-2020 infrastructure (custody, Bakkt, CME futures); (2) 2020-2023 corporate treasuries + Grayscale; (3) 2024-2025 spot ETFs + sovereign interest. Spot ETF approval Jan 10 2024 [coinglass2025]. Market cap >$1T. BTC volatility declining over time [wainwright2024, borri2025].

**§1.1.4 The Derivatives Market** — COMPLETE
Crypto derivatives surpass spot volume [elad2025]. Deribit dominates BTC options (85-90% market share). Open interest >$20B. Inverse margining system: both margin and settlement in BTC. Perpetual swaps as key hedging tool. CME offers cash-settled alternatives but lower volume.

### 1.2 Mechanics of Inverse Options
**§1.2.1 Payoff Structure** — COMPLETE
Inverse call: max(S_T - K, 0)/S_T = max(1 - K/S_T, 0) in BTC.
Inverse put: max(K - S_T, 0)/S_T = max(K/S_T - 1, 0) in BTC.
Key derivation with r=0 convention (no BTC money market) [alexander2023, lucic2024].

**§1.2.2 The Convexity Problem** — COMPLETE
Numerical example table: K=$50k, payoffs at S_T from $10k to $60k. BTC put payoff accelerates as S→0 (∂²P^BTC/∂S² = 2K/S³ > 0). Call has negative convexity (capped at 1 BTC). Double risk for put sellers: liability grows in BTC + collateral shrinks in USD.

**§1.2.3 Risk-Neutral Pricing** — COMPLETE
C₀^BTC = E^Q[max(S_T-K,0)/S_T]. Decomposition: inverse call = K·max(1/S_T - 1/K, 0). Fig: decomposition_call.png. Deribit IV convention: C₀^USD = S₀·C₀^BTC → invert BS formula.

**§1.2.4 Greeks: Delta and Gamma** — COMPLETE
Δ^BTC = Δ^USD/S - C^USD/S². Inverse put delta unbounded as S→0. Γ^BTC = Γ^USD/S - 2Δ^USD/S² + 2C^USD/S³. Fig: fig3_delta_put_BS.png.

**§1.2.5 Hedging and Double Risk** — COMPLETE
Cross-gamma risk unique to inverse. Collateral (BTC) depreciates as liability (BTC) increases. Lucic et al. (2024) minimum-variance hedge differs from BS delta.

**§1.2.6 Inverse vs Quanto Options** — COMPLETE
Quanto: fixed Q factor, no convexity. Inverse: stochastic 1/S_T conversion, convexity present. Table comparing features. Alexander et al. (2023) unified framework.

**§1.2.7 Implications for Thesis Contribution** — COMPLETE
rBergomi must compute E[payoff/S_T] not just E[payoff]. S_T>0 a.s. guaranteed by geometric dynamics. Division by small S_T increases MC variance for puts. Gap: rough vol + inverse options integration.

### 1.3 Market Data
**§1.3.1 Data Source** — COMPLETE
Deribit 85-90% market share. Dataset: Jan 2022 – Dec 2025, ~74,500 parquet files. Four years cover: 2022 correction ($47k→$16k), 2023 recovery, 2024 ETF bull market, 2025 institutional adoption.

**§1.3.2 Data Structure** — COMPLETE
Table of fields: timestamp, trade_id, price (BTC), mark_price, iv, index_price, contracts, amount, direction, instrument_name, expiration_timestamp. Naming convention: "BTC-28MAR25-60000-C".

**§1.3.3 Mark Price and IV** — COMPLETE
Deribit mark price: composite (VWAP, EMA smoothing, term structure bounds). IV computed from mark price via BS inversion after USD conversion.

**§1.3.4 Data Quality** — COMPLETE
Issues: price-mark deviations, extreme IVs (<10% or >300%), microstructure noise, moneyness extremes. Filtering detailed in Section 3.1.

## Key Equations
- Inverse put payoff: P^BTC = max(K/S_T - 1, 0)
- Inverse call payoff: C^BTC = max(1 - K/S_T, 0)
- BTC delta: Δ^BTC = Δ^USD/S - C^USD/S² [eq:delta_btc]
- BTC gamma: Γ^BTC = Γ^USD/S - 2Δ^USD/S² + 2C^USD/S³ [eq:gamma_btc]
- Inverse pricing: C₀^BTC = E^Q[max(S_T-K,0)/S_T] [eq:inverse_pricing]
- Quanto payoff: C^quanto = Q·max(S_T-K, 0) [eq:quanto_call]

## Figures & Tables
- Table 1.1: Payoff analysis K=$50k [tab:convexity_example]
- Table 1.2: Dataset fields [tab:dataset_fields]
- Table 1.3: Inverse vs Quanto comparison [tab:inverse_vs_quanto]
- Fig 1.1: Inverse call decomposition [fig:inverse_call_decomposition]
- Fig 1.2: Delta comparison put [fig:inverse_delta]

## Cross-References
- Labels: ch:crypto_derivatives, sec:bitcoin_asset_class, subsec:convexity, subsec:rn_pricing, subsec:greeks, subsec:hedging, subsec:inverse_vs_quanto, sec:market_data
- Forward refs: Chapter 2 (rBergomi model eq:rbergomi_S, eq:rbergomi_v), Section 3.1 (filtering)

## Citations Used
[bis2018, nakamoto2008, yermack2014, wainwright2024, borri2025, elad2025, coinglass2025, coinmarketcap2026, alexander2023, lucic2024, bayer2016, takaishi2020, deribit_mark]
