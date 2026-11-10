# eval-inflation

Study notes for **"Metric Scope Violations in Tabular Classification: An Eight-Stage Correction Protocol"** (submitted, IEEE BigData 2026).

On the synthetic IBM HR benchmark, five-fold AUC was 0.801 while the Brier score was 0.1594, compared with 0.1599 for the reference baseline. AUC measures how well the model ranks cases. The Brier score measures error in the predicted probabilities.

## Repository contents

This checkout contains the study summary, dependencies, and [dataset instructions](data/README.md). Reproducing the reported values requires `hardened_pipeline.py`, `generate_figures.py`, and `hardened_results.json`, which are absent from this checkout.

## Key results

| Metric | Value |
|---|---|
| AUC (5-fold CV) | 0.801 (95% CI: 0.720–0.882) |
| ECE | 0.053 |
| Brier score | 0.1594 (baseline: 0.1599) |
| DPD (Gender) | 0.015 |
| EOD (Age) | 0.266 |
| Proxy MI (StockOptionLevel) | 0.426 |
| Clustering silhouette | 0.236 (below the 0.30 threshold) |

## Pipeline stages

1. Stratified k-fold cross-validation (k=5)
2. Bootstrap confidence intervals on AUC
3. Calibration audit with ECE, reliability diagrams
4. Brier score with no-skill baseline comparison
5. SHAP attribution with provenance documentation
6. Subgroup fairness with Demographic Parity Difference, Equalized Odds Difference
7. Proxy variable mutual information audit
8. Clustering validity with algorithm-metric matching

## Dataset

The [dataset notes](data/README.md) give the download link and file path for the synthetic IBM HR Analytics Employee Attrition & Performance dataset. It has 1,470 rows and 35 columns.
