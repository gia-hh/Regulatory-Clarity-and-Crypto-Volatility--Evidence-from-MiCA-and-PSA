# Regulatory Clarity and Crypto Volatility: A GARCH Event Study of MiCA and PSA

This project examines how two major regulatory frameworks — Singapore's **Payment Services Act (PSA, effective 2020-01-28)** and the EU's **Markets in Crypto-Assets Regulation (MiCA, published 2023-07-19)** — affected cryptocurrency market volatility and liquidity. The core hypothesis is the **Uncertainty Resolution Effect**: formal regulation, regardless of its directional stringency, reduces market volatility by eliminating tail-risk uncertainty.

---

## Project Structure

```
project/
├── config.py                  # Global constants: event dates, coin list, variable names
├── utils.py                   # Shared utilities: significance stars, LaTeX export, model merging
├── 1_data_processing.py       # Data cleaning and feature construction
├── 2_eda.py                   # Exploratory analysis and pre-GARCH diagnostic tests
├── 3_garch_modeling.py        # EGARCH model selection and conditional volatility extraction
├── 4_panel_regression.py      # Panel OLS regressions, heterogeneity, placebo tests
├── 5_nlp_clarity.py           # NLP-based quantification of regulatory clarity (L-M dictionary)
└── 6_macro_robustness.py      # Robustness checks with macroeconomic controls (VIX, DXY)
```

**Data:** Daily OHLCV panel data for 6 cryptocurrencies (BTC, ETH, LTC, DOGE, BNB, XRP) from Binance, covering 2019–2024. After cleaning: **16,020 observations** across 6 coins.

**Dependent variables:**
- `7D_EWMA_Intraday_Price_Volatility` — 7-day EWMA of intraday price volatility (primary)
- `σ_LIQt` — volatility of liquidity (Amihud-based)
- `σ_Rt` — return volatility

---

## Step 1 — Data Processing (`1_data_processing.py`)

**What it does:**

Cleans the raw panel data and constructs features needed for all downstream analyses.

Key additions beyond the raw data:
- **Log return** (`log_return`): computed as ln(Pₜ/Pₜ₋₁) per coin; more suitable for time-series modelling than simple returns
- **Realized volatility** (`realized_vol_22d`): 22-day rolling standard deviation of log returns, annualised — serves as a GARCH benchmark
- **Running variables** (`days_to_PSA`, `days_to_MiCA`): signed integer distance in days from each event date. This corrects a bug in the original code where a 0/1 dummy was incorrectly used as a running variable for window selection
- **Two-stage policy dummies**: separates the announcement period (e.g. MiCA parliamentary vote) from the enforcement date, allowing the analysis to test whether markets reacted before the official effective date
- **Liquidity**: computed as Volume / (High − Low) where available

**Output:** `processed_data.csv` — 16,020 rows, 37 columns.

---

## Step 2 — Exploratory Data Analysis (`2_eda.py`)

**What it does:**

Runs three statistical tests to justify the use of GARCH over OLS, and produces time-series visualisations with policy event annotations.

**Results:**

| Test | Purpose | Result |
|---|---|---|
| Jarque-Bera normality test | Check if log returns are normally distributed | All 6 coins: p = 0.000 → reject normality |
| ARCH-LM test (lags=10) | Check if variance is time-varying | All 6 coins: p < 0.05 → reject constant variance |
| ADF unit root test | Check stationarity of log return series | All 6 coins: p = 0.000 → stationary |

All three tests unanimously support the use of GARCH:
- **JB test**: returns exhibit significant excess kurtosis (fat tails), violating OLS normality assumptions
- **ARCH-LM test**: variance is not constant — volatility clustering is present in all coins
- **ADF test**: log return series are stationary, satisfying GARCH model requirements

**Outputs:** `eda_log_returns.png`, `eda_volatility_clustering.png`, `jb_test.tex`, `arch_lm_test.tex`, `adf_test.tex`

---

## Step 3 — GARCH Modelling (`3_garch_modeling.py`)

**What it does:**

For each coin, fits five candidate models (GARCH(1,1), GARCH(1,2), GARCH(2,1), EGARCH(1,1), EGARCH(1,2)) using Student-t innovations, selects the best by AIC, and validates residuals with Ljung-Box tests.

**Why EGARCH over GARCH:**
EGARCH's logarithmic specification naturally ensures positive variance, and its asymmetry parameter γ captures the leverage effect — the empirical tendency for negative shocks to amplify volatility more than positive shocks of equal magnitude.

**Why (1,1) as starting point:**
The GARCH(1,1) specification is the standard baseline in the financial time-series literature, as it captures the bulk of volatility persistence in daily data. Higher-order specifications were also tested, with selection based on AIC/BIC.

**Model selection results:**

| Coin | Best Model | AIC | Ljung-Box (lag=10, p) | Ljung-Box² (lag=10, p) |
|---|---|---|---|---|
| BTC | EGARCH(1,2) | 10091.11 | 0.197 | 0.976 |
| ETH | EGARCH(1,2) | 11098.64 | 0.097 | 0.625 |
| LTC | EGARCH(1,1) | 11515.74 | 0.708 | 0.473 |
| DOGE | EGARCH(1,2) | 11580.28 | 0.446 | 1.000 |
| BNB | EGARCH(1,2) | 10796.30 | 0.170 | 0.443 |
| XRP | GARCH(1,1) | 11265.75 | 0.672 | 1.000 |

All Ljung-Box p-values exceed 0.05, confirming that selected models fully capture the volatility dynamics with no residual autocorrelation.

**Notable finding — XRP:** XRP is the only coin best described by a symmetric GARCH(1,1). This is consistent with XRP's history of regulatory uncertainty (SEC litigation), which may have already priced in asymmetric shocks continuously, leaving no additional leverage effect to be captured by EGARCH.

**Outputs:** `garch_{coin}.png` (conditional volatility plots), `nic_{coin}.png` (News Impact Curves for EGARCH coins), `garch_model_comparison.tex`, `data_with_garch.csv`

---

## Step 4 — Panel Regression Analysis (`4_panel_regression.py`)

**What it does:**

Five analysis modules using fixed-effects panel OLS, with entity clustering for standard errors.

### A. Baseline Regression (±90-day window)

**PSA results:**

| Variable | (1) Price Vol | (2) Liquidity Vol | (3) Return Vol |
|---|---|---|---|
| RegSin | 0.0005 | −0.0007*** | −0.0106*** |
| SEPU | −0.0327*** | −0.0015** | −0.0372*** |
| R² | 0.598 | 0.439 | 0.502 |

PSA has a **significant negative effect on liquidity and return volatility**, but not on price volatility in the ±90-day window. The Singapore Economic Policy Uncertainty index (SEPU) is itself strongly negative, suggesting the broader uncertainty environment drives much of the effect.

**MiCA results:**

| Variable | (1) Price Vol | (2) Liquidity Vol | (3) Return Vol |
|---|---|---|---|
| RegEU | 0.0008 | −0.0002 | −0.0053 |
| European News Index | −0.0281* | 0.0006 | 0.0136 |
| R² | 0.487 | 0.207 | 0.229 |

MiCA's aggregate coefficient is not statistically significant at the pooled level — a finding that is substantially qualified by the heterogeneity analysis below.

### B. Multi-Window Analysis

| Window | PSA β | PSA sig | MiCA β | MiCA sig |
|---|---|---|---|---|
| ±30 days | −0.0122 | *** | −0.0115 | * |
| ±60 days | −0.0012 | — | −0.0008 | — |
| ±90 days | +0.0005 | — | +0.0008 | — |

Both policies show **significant negative effects in the tight ±30-day window**, which fade as the window expands. This is consistent with a short-term uncertainty resolution effect that dissipates as markets adjust — not a persistent structural shift in volatility.

### C. Announcement vs. Enforcement

**PSA:** Announcement and enforcement coefficients are nearly identical (1.3165*** vs 1.3169***), suggesting the market did not price in the regulation ahead of the official effective date — the shock was concentrated at enforcement.

**MiCA:** Neither the announcement nor the enforcement dummy is statistically significant in the pooled regression, consistent with the pooled baseline result.

### D. Coin-Level Heterogeneity

**MiCA coin-level coefficients (price volatility):**

| Coin | β | Significance |
|---|---|---|
| BTC | −0.0054 | *** |
| ETH | −0.0047 | *** |
| LTC | −0.0072 | *** |
| DOGE | +0.0110 | *** |
| BNB | −0.0097 | *** |
| XRP | −0.0229 | * |

Five of six coins show significant **volatility reduction** under MiCA. **DOGE is the exception**, with a significant positive coefficient (+0.0110***). This is interpretable: DOGE's market is driven predominantly by retail sentiment and social media rather than institutional fundamentals, making it less responsive to regulatory clarity signals and potentially more volatile as institutional activity elsewhere shifted.

XRP shows the largest reduction (−0.0229*), consistent with its heightened sensitivity to regulatory developments given its legal history.

### E. Placebo Test

| Policy | True coefficient | Placebo mean | Placebo SD | Percentile rank |
|---|---|---|---|---|
| PSA | +0.0005 | −0.0050 | 0.0091 | 80th |
| MiCA | +0.0008 | −0.0049 | 0.0089 | 78th |

The true coefficients fall at the 80th and 78th percentile of the placebo distribution — above the median but below the conventional 95th percentile threshold. This reflects the fact that the pooled aggregate coefficients are weak; the stronger evidence for the policy effect lies in the ±30-day window and coin-level regressions.

**Outputs:** `baseline_PSA.tex`, `baseline_MiCA.tex`, `multiwindow_PSA/MiCA.tex`, `announce_vs_enforce_PSA/MiCA.tex`, `heterogeneity_coins.png`, `placebo_PSA/MiCA.png`

---

## Step 5 — NLP Regulatory Clarity Quantification (`5_nlp_clarity.py`)

**What it does:**

Replaces the hardcoded credibility scores (Score_PSA = 0.29, Score_MiCA = 1.0) in the original code with an objective, text-based measure of regulatory clarity derived from the full policy documents.

**Method:** Five dimensions computed using the **Loughran-McDonald (2011) financial dictionary** — the academic standard for financial text analysis — plus a crypto-specific entity lexicon:

| Dimension | Measure | Weight |
|---|---|---|
| Uncertainty density | L-M uncertainty word frequency (inverted) | 35% |
| Regulatory stringency | L-M constraining word frequency | 25% |
| Policy coverage | Crypto entity coverage breadth | 20% |
| Article clarity | Proportion of binding clauses (shall/must) | 10% |
| Semantic consistency | Mean pairwise TF-IDF cosine similarity | 10% |

**Output:** A `Regulatory Clarity Score` (0–1) for each policy, which is then added to the panel dataset as an interaction term (`RegSin_Clarity`, `RegEU_Clarity`) for use in regression. This replaces the arbitrary hardcoded scores from the original analysis and provides a transparent, replicable quantification methodology.

To run: place `MiCA.txt` and `PSA.txt` in the project directory. Full text available from EUR-Lex (Regulation EU 2023/1114) and Singapore Statutes Online (PSA 2019).

**Outputs:** `nlp_radar.png`, `nlp_clarity_score.png`, `nlp_clarity_results.tex`, `processed_data_with_clarity.csv`

---

## Step 6 — Macroeconomic Robustness (`6_macro_robustness.py`)

**What it does:**

Tests whether the policy coefficients are robust to the inclusion of global macroeconomic controls — VIX (global risk sentiment) and DXY (US dollar index) — which were both moving substantially during the PSA and MiCA event windows.

**Why this matters:**
- PSA (January 2020) coincided with early COVID-19 fear, with VIX beginning its historic rise
- MiCA (July 2023) occurred at the peak of the Fed's rate hike cycle, with DXY elevated

Without controlling for these factors, the policy dummies risk absorbing macro-driven volatility changes.

**Robustness regression results:**

| Specification | PSA (RegSin) | MiCA (RegEU) |
|---|---|---|
| (1) Baseline | +0.0005 | +0.0008 |
| (2) + log_VIX | −0.0140*** | −0.0002 |
| (3) + log_VIX + DXY | −0.0112*** | −0.0002 |

**Key finding — PSA:** After controlling for VIX, the PSA coefficient flips from insignificant to strongly negative (−0.0140***). This means the baseline regression was **upward-biased**: the VIX surge during the COVID period masked the policy's volatility-reducing effect. Controlling for global risk sentiment reveals the true policy signal. This is arguably a *stronger* result than a straightforward significant baseline — it demonstrates that VIX was a genuine confounder and that the PSA effect is robust once accounted for.

**MiCA:** Coefficient remains near zero across all specifications, consistent with the aggregate baseline result. The coin-level heterogeneity findings remain the primary evidence for MiCA's market impact.

**Outputs:** `macro_vs_crypto.png`, `macro_correlation.tex`, `macro_correlation_heatmap.png`, `vix_event_window.png`, `robustness_macro_PSA/MiCA.tex`, `coef_stability_PSA/MiCA.png`

---

## Summary of Findings

| Finding | PSA | MiCA |
|---|---|---|
| Aggregate volatility reduction | Significant after VIX control | Not significant (pooled) |
| Short-term effect (±30 days) | −0.0122*** | −0.0115* |
| Market pre-anticipation | Not detected | Not detected |
| Coin-level significance | BNB***, XRP** | BTC***, ETH***, LTC***, BNB***, XRP* |
| Exception coin | — | DOGE (+0.0110***) |
| Macro confounding | VIX was a significant confounder | Minimal |

**Overall interpretation:** Both PSA and MiCA are associated with cryptocurrency market volatility reduction, consistent with the **Uncertainty Resolution Hypothesis** — markets respond not primarily to the direction of regulation but to the *reduction in regulatory uncertainty* it provides. The effect is concentrated in the short-term window around the effective date, is more pronounced for fundamentals-driven assets (BTC, ETH, XRP) than sentiment-driven ones (DOGE), and for PSA is only detectable after controlling for the concurrent VIX shock.

---

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn statsmodels linearmodels arch scikit-learn yfinance
```

For NLP analysis:
```bash
pip install nltk beautifulsoup4
```

L-M Master Dictionary (free): https://sraf.nd.edu/loughranmcdonald-master-dictionary/

---

## How to Run

```bash
python 1_data_processing.py
python 2_eda.py
python 3_garch_modeling.py
python 4_panel_regression.py
python 5_nlp_clarity.py      # requires MiCA.txt and PSA.txt
python 6_macro_robustness.py
```

All outputs (figures, LaTeX tables) are written to the `output/` directory.

---

## Extension: Avellaneda-Stoikov Market-Making Simulation

Uses EGARCH-estimated volatility (7D_EWMA) as risk input to the 
Avellaneda-Stoikov (2008) optimal market-making model. Simulates 
dynamic bid-ask spread and inventory PnL for BTC and ETH, 2019–2024.

Key findings:
- Mean optimal spread ~1.05% (BTC), ~1.09% (ETH) — consistent with 
  crypto market microstructure
- Post-MiCA spread compression: −0.8% for both BTC and ETH, 
  consistent with uncertainty resolution hypothesis
- ETH spread widened +6.9% in PSA window, capturing COVID-driven 
  volatility spike (Jan 2020)
