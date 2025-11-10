# Analysis Execution Guide

## Overview
This guide documents the analyses created in `Min_Max_strikeprices.ipynb` that are ready to be executed in Jupyter.

## Analyses Created

### 1. Sampled Risk Aversion Analysis
**Location**: Cell `y24zurnc1b`

**Purpose**: Analyze the impact of load risk aversion (A_L) on contract terms while keeping generator risk aversion fixed at A_G = 0.5.

**Parameters**:
- A_G: 0.5 (fixed)
- A_L: Sampled uniformly from [0, 1] with 100 samples
- Strike price and contract amount: Fixed (not scenario-indexed)
- Scenarios_S = False
- Scenarios_M = False

**Expected Output**:
- DataFrame `df_risk_aversion` with columns: A_L, S_optimal, M_optimal, U_G, U_L, CVaR_G, CVaR_L, Zeta_G, Zeta_L, Earnings_G, Earnings_L, Nash_product
- CSV file: `Results/sampled_risk_aversion_AG=0.5_samples=100_{time_horizon}y_{num_scenarios}s.csv`

### 2. Distribution Plots with 85% Confidence Intervals
**Location**: Cell `g9masmex9nv`

**Purpose**: Visualize distributions of optimal strike prices and contract amounts with 85% confidence intervals.

**Visualizations**:
- Strike price (S_optimal) distribution with histogram and KDE
- Contract amount (M_optimal) distribution with histogram and KDE
- 85% CI shaded regions (7.5th to 92.5th percentiles)
- Mean values indicated with vertical lines

**Dependencies**: Requires `df_risk_aversion` from Cell y24zurnc1b

### 3. Revenue Composition Analysis
**Location**: Cell `dssea53ydf`

**Purpose**: Analyze how contract terms and outcomes vary with load risk aversion, inspired by Gousis et al. (2025).

**Visualizations** (2×2 subplot grid):
1. Strike price vs A_L with mean reference
2. Contract amount vs A_L with mean reference
3. Nash product vs A_L (negotiation efficiency)
4. Generator and Load utilities vs A_L

**Key Metrics**:
- Correlation between A_L and strike price
- Correlation between A_L and contract amount
- Mean strike price and contract amount

**Dependencies**: Requires `df_risk_aversion` from Cell y24zurnc1b

### 4. CVaR-Based Risk Analysis
**Location**: Cell `hr4okoot9wl`

**Purpose**: Quantify risk-return trade-offs using CVaR (Conditional Value at Risk) at 95% confidence level.

**Visualizations** (3-panel layout):
1. CVaR_G and CVaR_L vs A_L (risk exposure)
2. Expected earnings (Earnings_G and Earnings_L) vs A_L
3. Risk-return scatter plot with A_L color mapping

**Key Metrics**:
- CVaR/Earnings ratio for both parties
- Risk-return trade-off visualization
- Impact of risk aversion on downside risk

**Dependencies**: Requires `df_risk_aversion` from Cell y24zurnc1b

### 5. Bankability Metrics Analysis
**Location**: Cell `0832qkma1kd`

**Purpose**: Assess contract viability and stability, focusing on bankability aspects from Gousis et al. (2025).

**Visualizations** (2×2 subplot grid):
1. Threat points (Zeta_G, Zeta_L) vs A_L - disagreement scenario values
2. Utility gains from contract (U_G - Zeta_G, U_L - Zeta_L)
3. Relative utility gains as % of threat point
4. Contract terms stability using rolling standard deviation

**Key Metrics**:
- Pareto improvement verification (U_i > Zeta_i for all samples)
- Coefficient of variation for strike price and contract amount
- Rolling standard deviation for stability assessment
- Mutual benefit quantification

**Dependencies**: Requires `df_risk_aversion` from Cell y24zurnc1b

## Execution Order

To run these analyses in Jupyter:

1. **First**: Execute Cell `y24zurnc1b` (Sampled Risk Aversion Analysis)
   - This will take time as it runs 100 Nash bargaining optimizations
   - Expected runtime: ~5-15 minutes depending on solver performance
   - Watch for INFEASIBLE warnings

2. **Second**: Execute Cell `t2oxu2zy41f` (Save results to CSV)
   - Saves `df_risk_aversion` to Results folder

3. **Third**: Execute Cell `g9masmex9nv` (Distribution plots)
   - Creates 2-panel distribution visualization

4. **Fourth**: Execute Cell `dssea53ydf` (Revenue Composition Analysis)
   - Creates 4-panel sensitivity visualization

5. **Fifth**: Execute Cell `hr4okoot9wl` (CVaR-Based Risk Analysis)
   - Creates 3-panel risk analysis visualization

6. **Sixth**: Execute Cell `0832qkma1kd` (Bankability Metrics)
   - Creates 4-panel bankability visualization

## Expected Results

### CSV Output
- File: `Results/sampled_risk_aversion_AG=0.5_samples=100_{T}y_{S}s.csv`
- Rows: 100 (one per A_L sample)
- Columns: 12 (A_L, S_optimal, M_optimal, U_G, U_L, CVaR_G, CVaR_L, Zeta_G, Zeta_L, Earnings_G, Earnings_L, Nash_product)

### Plots Generated
1. Distribution plot with 85% CI (2 subplots)
2. Revenue composition plot (4 subplots)
3. CVaR-based risk analysis (3 subplots)
4. Bankability metrics (4 subplots)

**Total**: 4 figures with 13 subplots

## Troubleshooting

### Common Issues

1. **INFEASIBLE solutions**
   - Some A_L values may produce infeasible Nash bargaining problems
   - This is expected behavior - the code handles it gracefully
   - Check that threat points (Zeta_G, Zeta_L) are being calculated correctly

2. **Long runtime**
   - Each optimization can take 5-10 seconds
   - 100 samples × ~7 seconds = ~12 minutes total
   - Consider reducing `num_samples` for testing

3. **Missing scenario files**
   - Ensure 20-year scenario files exist in the `scenarios` folder
   - Required files:
     - `price_scenarios_reduced_20y_{S}s.csv`
     - `production_scenarios_reduced_20y_{S}s.csv`
     - `capture_rate_scenarios_reduced_20y_{S}s.csv`
     - `load_scenarios_reduced_20y_{S}s.csv`
     - `load_capture_rate_scenarios_reduced_20y_{S}s.csv`

4. **DataFrame not found**
   - If investigatory analysis cells (3-5) fail, ensure Cell 1 completed successfully
   - Check that `df_risk_aversion` exists in the notebook namespace
   - Re-run Cell `y24zurnc1b` if needed

## Paper Reference

These analyses are inspired by:

**Gousis, T., Kabouris, J., & Kanellos, F. D. (2025).** "Enhancing the viability and bankability of hybrid RES-BESS systems with corporate power purchase agreements and electricity market participation." *Energy Economics*, 141, 108024.

Key concepts applied:
- Semi-contracted vs semi-merchant revenue structures
- CVaR-based risk quantification (α = 0.95)
- Nash bargaining for fair PPA pricing
- Bankability through stable revenue streams
- Trade-offs between contract security and market exposure

## Notes

- All analyses use fixed strike price and contract amount (Scenarios_S=False, Scenarios_M=False)
- This differs from some other notebook analyses that use scenario-indexed contracts
- Risk aversion parameters range from 0 (risk-neutral) to 1 (highly risk-averse)
- Nash product measures negotiation efficiency and Pareto optimality
- Threat points represent no-contract (disagreement) scenarios
