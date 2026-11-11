# eval-inflation

Reproduction code for the IEEE BigData 2026 evaluation-inflation study on synthetic IBM HR data. On the test set, logistic regression had a Brier score of 0.1594, compared with 0.1599 for the baseline. The Brier score measures error in the predicted probabilities.

## Reproduce the results

Download the dataset using the [data instructions](data/README.md). From the repository root, run the following commands. The restored runner was checked with Python 3.14.

```bash
python -m pip install -r requirements.txt
python run_real_empirical_pipeline.py
```

The runner recreates the results tables and writes figures to `paper/figures/`. The three tables below matched runs using both the original semicolon-separated data file and the comma-separated download.

## Results

| Measurement | Value |
|---|---|
| CatBoost test AUC | 0.8181 (95% bootstrap CI 0.7394–0.9012) |
| CatBoost test Brier score | 0.0996 |
| Logistic regression test Brier score | 0.1594 |
| Baseline test Brier score | 0.1599 |
| Logistic regression AUC across seven split seeds | 0.7597–0.8443 |
| Highest clustering silhouette score | 0.273 (DBSCAN, with 17.3% of records marked as noise) |

The output tables record the [model comparison](results/model_outputs/model_benchmark_metrics.csv), [split-seed results](results/model_outputs/seed_robustness.csv), and [clustering checks](results/clustering_outputs/clustering_metrics.csv).
