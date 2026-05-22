# eval-inflation

Code for **"Metric Scope Violations in Tabular Classification: An Eight-Stage Correction Protocol"** (submitted, IEEE BigData 2026).

The paper documents how standard ML evaluation pipelines on tabular data produce inflated or misleading results when calibration, uncertainty quantification, subgroup fairness, and clustering validity are omitted. It proposes a correction protocol and applies it to the IBM HR attrition benchmark.

## The problem

Most attrition classification papers report AUC, maybe F1, and one model comparison. The metrics aren't wrong — they're just incomplete in ways that matter when the output is used for real decisions. An AUC without a confidence interval is a point estimate with unquantified variance. Brier = 0.1594 against a baseline of 0.1599 is not model confirmation — it's a finding that F1 reporting concealed.

## Key results

| Metric | Value |
|---|---|
| AUC (5-fold CV) | 0.801 (95% CI: 0.720–0.882) |
| ECE | 0.053 |
| Brier score | 0.1594 (baseline: 0.1599) |
| DPD (Gender) | 0.015 |
| EOD (Age) | 0.266 |
| Proxy MI (StockOptionLevel) | 0.426 |
| Clustering silhouette | 0.236 — below 0.30 threshold |

## Pipeline stages

1. Stratified k-fold cross-validation (k=5)
2. Bootstrap confidence intervals on AUC
3. Calibration audit — ECE, reliability diagrams
4. Brier score with no-skill baseline comparison
5. SHAP attribution with provenance documentation
6. Subgroup fairness — Demographic Parity Difference, Equalized Odds Difference
7. Proxy variable mutual information audit
8. Clustering validity with algorithm-metric matching

## Structure

```
hardened_pipeline.py   # full eight-stage evaluation
generate_figures.py    # all paper figures
hardened_results.json  # verified output snapshot
data/README.md         # dataset instructions
figures/               # generated plots
requirements.txt
```

## Reproduce

```bash
pip install -r requirements.txt
# place ibm_hr_attrition.csv in data/ — see data/README.md
python hardened_pipeline.py
python generate_figures.py
```

All numbers in the paper trace to `hardened_results.json`.

## Dataset

IBM HR Analytics Employee Attrition dataset — 1,470 observations, 35 features, synthetic, no real employee records.
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
