# LaTeX Context: Chapter 3 — Empirical Analysis and Calibration
**File**: LaTeX/Chapter_3.tex (655 lines) | **Status**: COMPLETE DRAFT

## Structure

### Chapter Introduction (lines 1-8)
Calibration of rBergomi to BTC inverse options on Deribit. Three shape params (H, η, ρ) optimised; ξ_0(·) estimated separately from ATM IVs. H freely optimised per snapshot. Internal consistency: both realised vol (P-measure H) and IV surface (Q-measure H) from Deribit data only.

### 3.1 Calibration Methodology
**§3.1.1 Implied Volatility Surface Extraction** — COMPLETE
- ~57.000 parquet files → cross-sectional snapshot at reference time t*
- ±4 hour window around t*
- Lightweight cached index: instrument → file, timestamp range, volume
- Filters: volume > 0,05 BTC, T ∈ [7, 90] days, moneyness m=K/S₀ ∈ [0,8; 1,2]
- VWAP aggregation: σ_IV^VWAP = Σ(q_j·σ_{IV,j})/Σq_j [eq:vwap_iv]
- OTM only: puts for K<S₀, calls for K>S₀
- Validity thresholds: ≥15 points, ≥30 trades, ≥3 maturities

**§3.1.2 Forward Variance Curve Estimation** — COMPLETE
- ATM via delta-neutral: d₁(K,T_k) with r=0, select min|d₁| < 0,5 [eq:d1_atm]
- ξ_0(T_k) = σ̂_ATM(T_k)² [eq:xi0_curve]
- Piecewise-constant interpolation

**§3.1.3 Loss Function** — COMPLETE
- Procedure per θ=(H,η,ρ): (1) simulate variance paths via Hybrid Scheme, (2) Mixed Estimator pricing per strike, (3) Brent IV inversion [0,01; 5,0], tol 10⁻⁸
- WRMSE: L(θ) = √(Σ w_i(σ_i^mod − σ̂_i)²) × 100 [eq:loss_function]
- Maturity weights: w_i ∝ 1/√T_i [eq:maturity_weights] — emphasises short-T where H most identifiable
- Mixed Estimator footnote: conditions on variance path, BS per path, F_m and σ_c formulas

**§3.1.4 Optimisation Algorithm** — COMPLETE
Phase 1 — Differential Evolution: bounds H∈[0,01;0,49], η∈[0,5;2,0], ρ∈[−0,99;−0,10] [eq:param_bounds]. Pop 15×3=45, max 25 gen, tol 1,0pp. N_MC=2.000 paths.
Phase 2 — Nelder-Mead: N_MC=10.000, simplex diameter tol 10⁻⁴, loss tol 10⁻⁶.
Setup: n_steps=50, κ=6, Numba JIT parallel, ~3-8 min/snapshot, seed=42. Warm starting for time-series.

### 3.2 Calibration Results
**§3.2.1 Snapshot Selection** — COMPLETE
26 snapshots, Jan 2022 – Mar 2025. Three categories: calm, high-vol, steep-skew.
Table 3.1 [tab:snapshot_summary]: 26 rows with date, spot, ATM IV, N points, category. Range: $16.500–$104.200; ATM IV 34,7%–96,1%; N=33–62.

**§3.2.2 Point-in-Time Smile Fits** — COMPLETE
Table 3.2 [tab:calibration_results]: 14 representative rows with H, η, ρ, ξ̄₀, RMSE, MAE. H ∈ [0,04; 0,15]; η ∈ [0,8; 1,9]; ρ ∈ [−0,85; −0,35]. Median RMSE = 1,31pp.
- Calm fits: RMSE < 1pp, H≈0,11-0,12. [Fig: skew_calm — 2 panels Aug 2023, Jul 2023]
- Bull/institutional fits: RMSE 1,1-1,4pp. [Fig: skew_bull — Feb 2024, Nov 2024]
- Crisis fits: RMSE 1,7-2,5pp. [Fig: skew_crisis — Jun 2022, Nov 2022]

**§3.2.3 Three-Dimensional Surface Fits** — COMPLETE
[Fig: surface_3d_calm — Jul 2023, Sep 2023]
[Fig: surface_3d_crisis — Jun 2022, Nov 2022]

**§3.2.4 Residual Analysis** — COMPLETE
ε_i = (σ_i^mod − σ̂_i) × 100 in pp.
Heatmaps: calm → small/structureless; crisis → mild bias OTM put wing.
[Fig: residuals_heatmap — 4 panels: Aug 2023, Feb 2024, May 2022, Nov 2022]
Histograms: centred at zero, σ_resid 0,6pp (calm) to 2,1pp (crisis).
[Fig: residuals_hist — 3 panels: calm, bull, high vol]

### 3.3 Parameter Dynamics and Market Regimes
**§3.3.1 Longitudinal Parameter Evolution** — COMPLETE
[Fig: skew_timeseries — 4 panels: May 2022, Aug 2023, Jan 2024, Jan 2025]
- H: range [0,04; 0,15], mean≈0,08, σ≈0,02. Lower in crisis (~0,05-0,07), higher calm (~0,11-0,12).
- η: range [1,1; 1,9]. Co-moves with ATM IV (vol-of-vol clustering).
- ρ: range [−0,85; −0,35], median≈−0,58. More negative in stress, less in calm. Weaker than equity (−0,9 to −0,7).

**§3.3.2 Regime Dependence** — COMPLETE
[Fig: residuals_regimes — 6 panels across crisis/calm/bull]
- Crisis (ATM>75%): H≈0,05-0,07, η≈1,7-1,9, ρ≈−0,7 to −0,85. RMSE avg 2,1pp.
- Calm (ATM<45%): H≈0,10-0,12, η≈1,1-1,3, ρ≈−0,35 to −0,55. RMSE avg 0,9pp.
- Transition (45-75%): interpolates. RMSE 1,0-1,5pp.

**§3.3.3 Skew Reproduction Across Term Structure** — COMPLETE
Skew decays as T^{H−1/2} ≈ T^{−0,42} with H≈0,08. Matches Deribit data.
[Fig: skew_additional — 6 panels Dec 2022 through Feb 2025]
[Fig: iv_smile_maturity — 2 panels Sep 2023, decomposed by maturity bucket]

**§3.3.4 Cholesky vs Hybrid Scheme Comparison** — COMPLETE
Differences typically <0,5pp IV. Hybrid marginally better at short T.
[Fig: scheme_comparison — 4 panels: Cholesky vs Hybrid for Jun 2022, Jul 2023]

**§3.3.5 Additional 3D Surface Visualisations** — COMPLETE
[Fig: surface_3d_additional — 6 panels May 2022 through Sep 2023]

### 3.4 Performance Evaluation and Conclusions
**§3.4.1 Comparison with Classical Models** — COMPLETE
BS: median RMSE 8-12pp. Heston: 3-5pp (Cao et al. 2023, Hoang & Baur 2022). rBergomi: 1,31pp. Improvement most at T<30d.

**§3.4.2 Strengths and Limitations** — COMPLETE
Strengths: parsimony (3 params + ξ_0), stable H across regimes, Mixed Estimator 10-50× variance reduction, 3-8 min/calibration.
Limitations: (1) Crisis stress >2pp RMSE, OTM put wing bias — inherent to 1-factor models. (2) ξ_0 from data quality depends on ATM liquidity.

**§3.4.3 Summary of Findings** — COMPLETE
H ∈ [0,04; 0,15], mean 0,08. Median RMSE 1,31pp. Regime-dependent parameters are economically interpretable. Hybrid Scheme + Mixed Estimator makes calibration feasible. rBergomi established as viable tool for inverse crypto-derivatives.

## Key Equations
- VWAP IV: σ_IV^VWAP = Σ(q_j σ_{IV,j})/Σq_j [eq:vwap_iv]
- d₁ for ATM: d₁(K,T) = [ln(S₀/K) + ½σ̂²T]/(σ̂√T) [eq:d1_atm]
- Forward variance: ξ_0(T_k) = σ̂_ATM(T_k)² [eq:xi0_curve]
- Loss: L(θ) = √(Σ w_i(σ_i^mod − σ̂_i)²) × 100 [eq:loss_function]
- Weights: w_i ∝ 1/√T_i [eq:maturity_weights]
- Param bounds: H∈[0,01;0,49], η∈[0,5;2,0], ρ∈[−0,99;−0,10] [eq:param_bounds]

## Figures & Tables
- Table 3.1: Snapshot summary (26 dates) [tab:snapshot_summary]
- Table 3.2: Calibration results (14 representative) [tab:calibration_results]
- Fig 3.1: Calm smile fits [fig:skew_calm]
- Fig 3.2: Bull smile fits [fig:skew_bull]
- Fig 3.3: Crisis smile fits [fig:skew_crisis]
- Fig 3.4: 3D surface calm [fig:surface_3d_calm]
- Fig 3.5: 3D surface crisis [fig:surface_3d_crisis]
- Fig 3.6: Residual heatmaps [fig:residuals_heatmap]
- Fig 3.7: Residual histograms [fig:residuals_hist]
- Fig 3.8: Skew timeseries evolution [fig:skew_timeseries]
- Fig 3.9: Residuals across regimes [fig:residuals_regimes]
- Fig 3.10: Additional smile fits [fig:skew_additional]
- Fig 3.11: Smile by maturity [fig:iv_smile_maturity]
- Fig 3.12: Cholesky vs Hybrid [fig:scheme_comparison]
- Fig 3.13: Additional 3D surfaces [fig:surface_3d_additional]

## Cross-References
- Labels: ch:empirical_analysis, sec:calibration_methodology, sec:calibration_results, sec:parameter_dynamics, sec:performance_evaluation, subsec:iv_extraction, subsec:fwd_var_estimation, subsec:forward_variance_estimation, subsec:loss_function, subsec:optimisation, subsec:snapshot_selection, subsec:smile_fits, subsec:3d_surface, subsec:residual_analysis, subsec:parameter_evolution, subsec:regime_dependence, subsec:skew_term_structure, subsec:scheme_comparison, subsec:additional_3d, subsec:model_comparison, subsec:strengths_limitations, subsec:summary
- Back-refs: Chapter 2 (ch:theoretical_foundations, eq:rbergomi_S, eq:rbergomi_v, alg:hybrid, prop:atm_skew), Section 1.2 (sec:inverse_mechanics, subsec:convexity, subsec:payoff_structure), Section 2.1 (sec:classical_limitations, subsec:limitations_heston)

## Citations Used
[gatheral2018, bennedsen2017, mccrickerd2018, bayer2016, takaishi2020, bennedsen2022, li2025]
