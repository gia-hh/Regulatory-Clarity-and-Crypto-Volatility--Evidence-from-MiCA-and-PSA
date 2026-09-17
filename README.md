# Regulatory Clarity and Crypto Volatility

### Undergraduate Thesis Extension — NLP, Volatility Modeling, and Policy Event Analysis

This project extends undergraduate thesis research on cryptocurrency regulation by asking a narrower empirical question:

> **Can regulatory clarity be quantified from policy text, and is greater regulatory clarity associated with changes in cryptocurrency-market volatility around major regulatory events?**

The analysis combines:

* daily cryptocurrency market data;
* GARCH / EGARCH volatility modeling;
* fixed-effects panel regressions;
* multi-window event analysis;
* placebo and heterogeneity checks;
* macroeconomic robustness controls;
* an NLP-based measure of regulatory clarity.

The study focuses on Singapore's **Payment Services Act (PSA)** and the EU's **Markets in Crypto-Assets Regulation (MiCA)**.

> **Interpretation boundary:** this is an observational event-study extension, not a randomized or quasi-experimental design that proves a causal policy effect. Results are interpreted as event-window associations and evidence consistent with an uncertainty-resolution hypothesis.

---

## Project Summary

The motivating hypothesis is that regulation may influence markets not only through its substantive direction, but also through the amount of uncertainty it resolves.

The project therefore examines three related questions:

1. Do cryptocurrency volatility measures change around major regulatory events?
2. Are those patterns concentrated in short event windows or persistent over longer horizons?
3. Can the textual clarity of regulation itself be measured systematically rather than assigned an arbitrary score?

The empirical pipeline uses **16,020 daily observations across six cryptocurrencies from 2019–2024**:

```text
BTC
ETH
LTC
DOGE
BNB
XRP
```

---

## Key Findings

### Short-window volatility association

In the configured ±30-day event windows, both PSA and MiCA are associated with lower price-volatility measures:

| Event | Coefficient | Significance |
| ----- | ----------: | -----------: |
| PSA   |     −0.0122 |          *** |
| MiCA  |     −0.0115 |            * |

The coefficients attenuate in wider ±60-day and ±90-day windows.

This pattern is **consistent with a short-horizon uncertainty-resolution interpretation**, but should not be interpreted as proof that regulation caused the volatility decline.

---

### Pooled results are weaker

The broader ±90-day pooled regressions do not show a statistically significant negative price-volatility coefficient for either policy in the baseline specification.

This matters because it prevents the short-window finding from being interpreted as a persistent structural change.

---

### Cross-asset heterogeneity

The MiCA event-window regressions show substantial heterogeneity across cryptocurrencies.

Five of the six assets have negative estimated coefficients in the coin-level specification, while DOGE has a positive coefficient.

These differences indicate that the association is not uniform across assets.

Possible explanations include differences in:

* investor composition;
* market structure;
* sensitivity to regulatory news;
* liquidity;
* contemporaneous market conditions.

These explanations are hypotheses rather than directly identified causal mechanisms.

---

### Macroeconomic controls materially affect PSA estimates

Adding VIX and DXY controls changes the PSA coefficient substantially:

| Specification    | PSA coefficient | MiCA coefficient |
| ---------------- | --------------: | ---------------: |
| Baseline         |         +0.0005 |          +0.0008 |
| + log(VIX)       |      −0.0140*** |          −0.0002 |
| + log(VIX) + DXY |      −0.0112*** |          −0.0002 |

The PSA estimate becomes more negative after controlling for global risk sentiment.

This demonstrates that inference around the January 2020 PSA event is sensitive to contemporaneous macroeconomic conditions, particularly the rapidly changing global risk environment surrounding early COVID-19.

It should **not** be interpreted as proving that VIX is the unique causal confounder or that the adjusted coefficient is the “true” policy effect.

---

## NLP Regulatory-Clarity Extension

A central extension of the original thesis was replacing manually assigned policy-credibility scores with a reproducible text-based measure.

The pipeline constructs a **Regulatory Clarity Score** from the full PSA and MiCA texts using the Loughran–McDonald financial dictionary, a crypto-specific entity lexicon, and document-structure measures.

### Components

| Dimension             | Measure                                           | Weight |
| --------------------- | ------------------------------------------------- | -----: |
| Uncertainty density   | Inverted L-M uncertainty-word frequency           |    35% |
| Regulatory stringency | L-M constraining-word frequency                   |    25% |
| Policy coverage       | Crypto-entity coverage breadth                    |    20% |
| Article clarity       | Share of binding clauses such as `shall` / `must` |    10% |
| Semantic consistency  | Mean pairwise TF-IDF cosine similarity            |    10% |

The resulting score is normalized to a 0–1 scale and incorporated into the panel dataset through policy × clarity interaction terms.

This is intended as a **transparent research proxy for regulatory clarity**, not as a validated universal measure of legal quality.

---

## Research Pipeline

```text
Market + policy data
        │
        ▼
Data Cleaning & Feature Construction
        │
        ▼
Volatility Diagnostics
        │
        ▼
GARCH / EGARCH Model Selection
        │
        ▼
Panel Event-Window Analysis
        │
        ├── Multi-window tests
        ├── Announcement vs. event timing
        ├── Coin-level heterogeneity
        └── Placebo tests
        │
        ▼
NLP Regulatory-Clarity Scoring
        │
        ▼
Macroeconomic Robustness
        │
        ├── VIX
        └── DXY
```

---

## Repository Structure

```text
.
├── config.py
├── utils.py
├── 1_data_processing.py
├── 2_eda.py
├── 3_garch_modeling.py
├── 4_panel_regression.py
├── 5_nlp_clarity.py
├── 6_macro_robustness.py
├── LM_dictionary.csv
├── MiCA.txt
├── PSA.txt
├── merged_data.csv
├── processed_data.csv
├── processed_data_with_clarity.csv
└── output/
```

### Modules

**`1_data_processing.py`**

* cleans the cryptocurrency panel;
* computes log returns;
* constructs volatility and liquidity features;
* creates event-distance variables;
* prepares policy indicators.

**`2_eda.py`**

* Jarque–Bera diagnostics;
* ARCH-LM tests;
* ADF stationarity tests;
* volatility and return visualizations.

**`3_garch_modeling.py`**

Fits candidate specifications including:

* GARCH(1,1);
* GARCH(1,2);
* GARCH(2,1);
* EGARCH(1,1);
* EGARCH(1,2);

with Student-t innovations.

Models are compared using information criteria and residual diagnostics.

**`4_panel_regression.py`**

Implements:

* fixed-effects panel regressions;
* clustered standard errors;
* ±30 / ±60 / ±90-day specifications;
* announcement-vs-event comparisons;
* coin-level heterogeneity;
* placebo tests.

**`5_nlp_clarity.py`**

Constructs the NLP-based Regulatory Clarity Score from policy documents.

**`6_macro_robustness.py`**

Evaluates coefficient sensitivity to:

* VIX;
* DXY.

---

## Volatility Modeling

For each cryptocurrency, the pipeline compares multiple GARCH-family specifications.

Current model-selection output:

| Coin | Selected Model |
| ---- | -------------- |
| BTC  | EGARCH(1,2)    |
| ETH  | EGARCH(1,2)    |
| LTC  | EGARCH(1,1)    |
| DOGE | EGARCH(1,2)    |
| BNB  | EGARCH(1,2)    |
| XRP  | GARCH(1,1)     |

Residual Ljung–Box diagnostics are used to evaluate remaining serial dependence.

The model-selection results are descriptive modeling choices rather than evidence that one market has a particular behavioral mechanism.

For example, XRP selecting a symmetric GARCH model does **not** by itself establish that prior regulatory uncertainty explains its volatility structure.

---

## Multi-Window Analysis

Price-volatility coefficients:

| Window   |        PSA |     MiCA |
| -------- | ---------: | -------: |
| ±30 days | −0.0122*** | −0.0115* |
| ±60 days |    −0.0012 |  −0.0008 |
| ±90 days |    +0.0005 |  +0.0008 |

The concentration of statistical significance in the shortest window motivates the project's short-horizon interpretation.

At the same time, attenuation across wider windows is an important limitation:

> the evidence does not support describing regulation as producing a persistent reduction in cryptocurrency volatility.

---

## Placebo Analysis

The pooled ±90-day coefficients lie around the:

```text
PSA  → 80th percentile
MiCA → 78th percentile
```

of their respective placebo distributions.

Neither reaches a conventional 95th-percentile benchmark.

This weakens any claim that the pooled policy coefficients are unusually large relative to arbitrary event timing and is one reason the project emphasizes the short-window evidence cautiously.

---

## Event-Date Definition

The analysis relies on explicitly configured policy-event dates.

For PSA, the primary event is:

```text
2020-01-28
```

corresponding to the Act's commencement.

The current repository configuration uses:

```text
2023-07-19
```

as the primary MiCA event date.

Because regulatory processes contain multiple economically meaningful dates — proposal, parliamentary approval, legal publication, entry into force, and staged application — conclusions can depend on the event definition.

**The MiCA date should therefore be treated as a research-design choice and subjected to event-date sensitivity analysis rather than assumed to be uniquely correct.**

A future revision should explicitly compare alternative legally relevant MiCA dates and rerun the event-window specifications.

---

## Interpretation Framework

The results are broadly consistent with the following hypothesis:

> **Greater regulatory resolution may coincide with lower short-horizon market uncertainty.**

But the analysis does not separately identify regulation from every contemporaneous influence.

Accordingly, preferred language is:

* **associated with**;
* **consistent with**;
* **coincides with**;
* **the coefficient changes after controlling for...**

rather than:

* caused;
* true policy effect;
* proved;
* regulation reduced volatility by X.

---

## Research Limitations

The main limitations are:

* observational rather than randomized identification;
* sensitivity to event-date choice;
* major macroeconomic events overlap some regulatory windows;
* only six cryptocurrencies are included;
* crypto markets evolved materially between 2019 and 2024;
* coin-level regressions involve relatively small cross-sectional scope;
* placebo evidence for the pooled coefficients is not unusually extreme;
* NLP clarity weights are researcher-defined;
* the Regulatory Clarity Score is a research proxy rather than an externally validated legal-quality measure;
* statistical significance in a narrow event window does not establish a persistent causal effect.

These limitations are part of the interpretation rather than hidden behind the model outputs.

---

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn statsmodels linearmodels arch scikit-learn yfinance
```

For the NLP module:

```bash
pip install nltk beautifulsoup4
```

---

## Running the Analysis

Run the modules sequentially:

```bash
python 1_data_processing.py
python 2_eda.py
python 3_garch_modeling.py
python 4_panel_regression.py
python 5_nlp_clarity.py
python 6_macro_robustness.py
```

Generated figures and LaTeX tables are written to:

```text
output/
```

---

## Optional Market-Making Extension

A separate exploratory extension used estimated volatility as an input to an Avellaneda–Stoikov-style market-making simulation for BTC and ETH.

The simulation produced model-implied spread changes including approximately:

```text
~0.8% post-MiCA spread compression
~6.9% ETH widening around the PSA window
```

These are **simulation outputs**, not observed market spreads or realized trading results.

They are secondary to the main regulatory-volatility analysis and should not be interpreted as evidence of a profitable market-making strategy.

---

## What This Project Demonstrates

The project combines:

* financial time-series modeling;
* event-study reasoning;
* panel econometrics;
* NLP-based policy measurement;
* robustness analysis;
* explicit distinction between statistical association and causal interpretation.

The main methodological lesson is:

> **A statistically significant event-window coefficient is only the beginning of the analysis; event definition, macroeconomic overlap, heterogeneity, placebo evidence, and measurement choices determine how strongly the result can be interpreted.**
