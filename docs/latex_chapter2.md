# LaTeX Context: Chapter 2 — Theoretical Foundations: From Stochastic to Rough Volatility
**File**: LaTeX/Chapter_2.tex (545 lines) | **Status**: COMPLETE DRAFT

## Structure

### 2.1 Limitations of Classical Models
**§2.1.1 The Black-Scholes Framework and Its Assumptions** — COMPLETE
GBM: dS_t = μS_t dt + σS_t dW_t. Constant σ implies flat IV surface, normal returns. Empirical contradictions: leptokurtosis, smile/skew, amplified in BTC.

**§2.1.2 Empirical Failures in Bitcoin Markets** — COMPLETE
Three failures documented with figures:
- Fat tails: BTC kurtosis ~125 (5-min), ~17 (hourly), >5 (daily). Student's t with ν≈2,2. [Fig: distribution_analysis, kurtosis_analysis]
- IV smile: OTM puts ~21pp above ATM. [Fig: volatility_smile_2d, volatility_surface_3d]
- Term structure: Smile steepness 65 (0-7d) → 3 (90+d). ATM IV rises from 45,9% to 57,6%. Power-law decay ψ(T) ∝ T^{H-1/2}. [Fig: vol_structure_analysis]

**§2.1.3 The Heston Stochastic Volatility Model** — COMPLETE
dS_t = μS_t dt + √v_t S_t dW_t^S; dv_t = κ(θ−v_t)dt + σ_v √v_t dW_t^v; corr = ρ. Advantages: fat tails, skew via ρ, mean reversion, semi-analytical CF solution.

**§2.1.4 Limitations of Heston for Bitcoin** — COMPLETE
Three failures: (1) Insufficient short-term skew — Heston skew → ρσ_v as T→0 (finite), vs empirical T^{H-1/2} (explosive). (2) Smile flattening at short T — Heston predicts convergence to √v_0, market shows opposite. (3) Parameter instability across dates.

**§2.1.5 The Volatility Path Regularity Problem** — COMPLETE
Root cause: BM has Hölder regularity ~1/2. Gatheral et al. (2018) showed H≈0,1 for equities. Takaishi (2020) confirmed for BTC. Rough vol generates: steeper short-term skews, more pronounced short-T smiles, fat tails via high-freq clustering.

### 2.2 The Rough Volatility Revolution
**§2.2.1 Fractional Brownian Motion and the Hurst Parameter** — COMPLETE
Definition 2.0.1 (fBm): E[W_t^H W_s^H] = ½(|t|^{2H} + |s|^{2H} − |t−s|^{2H}). H=0,5 → standard BM. Three properties: path regularity (Hölder < H), self-similarity (W_{ct}^H =^d c^H W_t^H), increment correlation (anti-persistence for H<0,5).

**§2.2.2 Empirical Evidence: Volatility is Rough** — COMPLETE
Scaling method: E[|X_{t+Δ} − X_t|^q] ∝ Δ^{qH} → log-log regression. ACF: ρ(Δ) ≈ 1 − cΔ^{2H}. Gatheral et al. (2018): H ∈ [0,08; 0,14] for equities.

**§2.2.3 Rough Volatility in Bitcoin Markets** — COMPLETE
Takaishi (2020): H ∈ [0,05; 0,12] scaling regression. Takaishi (2021): H ∈ [0,08; 0,15] MF-DFA. Takaishi (2025): H ∈ [0,07; 0,14] wavelet analysis. Table 2.1 [tab:hurst_estimates]. Detailed methodology comparison (scaling regression, MF-DFA, wavelet).

**§2.2.4 Statistical Properties of Rough Volatility** — COMPLETE
- Volatility clustering: from stochastic exponential + forward variance curve, H shapes short-scale dynamics
- Leverage effect: ρ<0 + rough dynamics → skew explodes as T^{H-1/2} at short T (vs finite in Heston)
- Leptokurtosis: from stochastic vol + rough high-freq fluctuations
- Term structure: ψ(T) ∝ T^{H-1/2} (formalised in Prop 2.0.2)

**§2.2.5 Visual Comparison: Smooth vs Rough Paths** — COMPLETE
[Fig: fbm_sample_paths] H=0,5 vs H=0,1. Cholesky factorisation, n=500. Rough path: rapid oscillations, frequent reversals.

**§2.2.6 Implications for Option Pricing** — COMPLETE
H governs roughness like σ governs level. H stable across time/assets (0,05–0,15). Rough vol resolves short-term skew explosion and pronounced smiles.

### 2.3 The Rough Bergomi Model
**§2.3.1 Model Specification** — COMPLETE
Core equations:
- dS_t/S_t = √v_t dW_t [eq:rbergomi_S]
- v_t = ξ_0(t) E(η ∫₀ᵗ(t−s)^{H−1/2} dW_s^⊥) [eq:rbergomi_v]
- Stochastic exponential: E(X_t) = exp(X_t − ½⟨X⟩_t)
- Volterra integral: W̃_t = ∫₀ᵗ(t−s)^{H−1/2} dW_s^⊥ (kernel singular at s=t for H<1/2, L²-integrable)
- Hölder regularity: γ < H
- Remark 2.1 [rmk:non_semimartingale]: W̃_t not a semimartingale → E(ηW̃_t) = exp(ηW̃_t − η²/2 Var(W̃_t))
- Mandelbrot-Van Ness representation: full fBm vs Riemann-Liouville (Type II) used in rBergomi
- Forward variance: ξ_0(t) = E^Q[v_t | F_0], consistency: E^Q[v_t] = ξ_0(t)·1 = ξ_0(t)
- Explicit variance: v_t = ξ_0(t) exp(ηW̃_t − η²t^{2H}/(4H))
- Remark 2.2 [rmk:variance_convention]: Normalised kernel √(2H)(t−s)^{H−1/2} gives Var=t^{2H}, η̂=η/√(2H). McCrickerd convention.
- Proposition 2.0.1 [prop:logvar_variance]: Var(log v_t) = η²t^{2H}/(2H). Proof via Itô isometry.

**§2.3.2 Parameters and Correlation Structure** — COMPLETE
Decomposition: W_t^⊥ = ρW_t + √(1−ρ²)W_t^{⊥⊥}. Leverage effect via ρ<0. Separation of roles: ρ controls skew magnitude, H controls term structure.
- Proposition 2.0.2 [prop:atm_skew]: ψ(T) ~ ρηC_H T^{H−1/2} as T→0. (Bayer et al. 2016, Thm 3.1)
- Heston comparison: ψ_Heston(T) → ρσ_v/(2√v_0)
- Table 2.2 [tab:rbergomi_params]: H, η, ρ effects on IV surface
- Heston needs 5 params (κ,θ,σ_v,ρ,v_0), SABR 4, rBergomi only 3 + ξ_0 from data
- Typical BTC ranges: H ∈ [0,05; 0,15], η ∈ [1,5; 3,5], ρ ∈ [−0,9; −0,5]

**§2.3.3 The Hybrid Scheme** — COMPLETE
Computational challenge: direct Cholesky O(n³) + O(n²) storage, impractical for n≥1000.
Hybrid Scheme (Bennedsen et al. 2017): W̃_{t_i} = R_i + D_i
- Recent R_i: exact, kernel weights w_l = [(lΔ)^{H+1/2} − ((l−1)Δ)^{H+1/2}]/(H+1/2), for l=1..κ
- Distant D_i: Riemann sum approximation, O(n²) total or O(n) with incremental update
- Remark 2.3 [rmk:distant_approximation]: D_i ← D_{i-1} + (t_i−t_{i−κ})^{H−1/2} ΔW_{i−κ}^⊥
- Algorithm 1 [alg:hybrid]: Full pseudocode (preprocessing + main loop + correlated BM + log-Euler)
- Note on variance convention (normalised vs un-normalised kernel)
- κ choice: trade-off accuracy vs cost. Bennedsen: κ=3. This work: discussed in Ch3.
- Proposition 2.0.3 [prop:convergence]: Strong convergence O(n^{−H}). Slower than classical n^{−1/2}.

**§2.3.4 Pricing Inverse Options** — COMPLETE
MC estimators: Ĉ₀^BTC = (1/N)Σ max(S_T^(m)−K,0)/S_T^(m); P̂₀^BTC analogous.
Complication: 1/S_T amplifies small S_T paths → high variance for puts.
Variance reduction: antithetic variates, control variates (BS as control), importance sampling. Combined: up to 10× error reduction.

## Key Equations
- GBM: dS_t = μS_t dt + σS_t dW_t
- Heston: dv_t = κ(θ−v_t)dt + σ_v√v_t dW_t^v [eq:heston]
- Skew scaling: ∂σ_imp/∂k|_{k=0} ~ T^{H−1/2} [eq:skew_behaviour]
- fBm covariance: E[W_t^H W_s^H] = ½(|t|^{2H}+|s|^{2H}−|t−s|^{2H})
- rBergomi: dS_t/S_t = √v_t dW_t [eq:rbergomi_S]
- rBergomi variance: v_t = ξ_0(t)E(η∫(t−s)^{H−1/2}dW_s^⊥) [eq:rbergomi_v]
- Explicit: v_t = ξ_0(t)exp(ηW̃_t − η²t^{2H}/(4H))
- ATM skew: ψ(T) ~ ρηC_H T^{H−1/2}
- Heston skew limit: ρσ_v/(2√v_0)
- Convergence: O(n^{−H})
- Kernel weights: w_l = [(lΔ)^{H+1/2}−((l−1)Δ)^{H+1/2}]/(H+1/2)

## Definitions, Propositions, Remarks
- Definition 2.0.1: Fractional Brownian Motion [def:fbm]
- Definition 2.0.2: Hybrid Scheme Decomposition (R_i + D_i)
- Remark 2.1: Non-Semimartingale Nature [rmk:non_semimartingale]
- Remark 2.2: Variance Convention (normalised vs un-normalised) [rmk:variance_convention]
- Remark 2.3: Incremental Distant Component [rmk:distant_approximation]
- Proposition 2.0.1: Var(log v_t) = η²t^{2H}/(2H) [prop:logvar_variance]
- Proposition 2.0.2: ATM Skew Asymptotics [prop:atm_skew]
- Proposition 2.0.3: Strong Convergence O(n^{−H}) [prop:convergence]

## Figures & Tables
- Fig 2.1: Distribution analysis [fig:distribution_analysis]
- Fig 2.2: Kurtosis analysis [fig:kurtosis_analysis]
- Fig 2.3: Volatility smile 2D [fig:volatility_smile]
- Fig 2.4: Volatility surface 3D [fig:volatility_surface]
- Fig 2.5: Term structure analysis [fig:vol_structure_analysis]
- Fig 2.6: Smooth vs rough fBm paths [fig:smooth_vs_rough]
- Table 2.1: Hurst estimates across studies [tab:hurst_estimates]
- Table 2.2: rBergomi parameter effects [tab:rbergomi_params]
- Algorithm 1: Hybrid Scheme [alg:hybrid]

## Cross-References
- Labels: ch:theoretical_foundations, sec:classical_limitations, sec:rough_volatility, sec:rbergomi, subsec:limitations_heston
- Back-refs: Section 1.2 (sec:inverse_mechanics for pricing inverse options)
- Forward-refs: Chapter 3 (ch:empirical_analysis), Section 3.1 (subsec:forward_variance_estimation)

## Citations Used
[black1973, merton1973, heston1993, gatheral2018, gatheral2011, bayer2016, takaishi2020, takaishi2021, takaishi2025, mandelbrot1968, nualart2006, biagini2008, decreusefond1999, bennedsen2017, mccrickerd2018, lucic2024, forde2009]
