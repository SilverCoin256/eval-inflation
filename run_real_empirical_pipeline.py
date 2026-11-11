from pathlib import Path
import json
import hashlib
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "data/ibm_hr_attrition.csv"
PROCESSED = ROOT / "data/processed"
CLUSTER_OUT = ROOT / "results/clustering_outputs"
MODEL_OUT = ROOT / "results/model_outputs"
FIG_DIR = ROOT / "paper/figures"
TABLE_DIR = ROOT / "paper/tables"
for p in [PROCESSED, CLUSTER_OUT, MODEL_OUT, FIG_DIR, TABLE_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_md(path, text):
    path.write_text(text, encoding="utf-8")


def markdown_table(df):
    df = df.copy()
    if df.empty:
        return "_No rows._"
    df = df.fillna("")
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |"]
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def safe_auc(y, score):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score)) if len(set(y)) > 1 else float("nan")


def bootstrap_ci(y, score, metric_fn, n=300, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    y = np.asarray(y)
    score = np.asarray(score)
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        vals.append(metric_fn(y[idx], score[idx]))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def cramers_v(table):
    from scipy.stats import chi2_contingency
    chi2, _, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    if n == 0:
        return float("nan")
    r, k = table.shape
    return float(np.sqrt((chi2 / n) / max(1, min(k - 1, r - 1))))


def disparate_impact(predicted_positive, group):
    df = pd.DataFrame({"pred": predicted_positive, "group": group}).dropna()
    rates = df.groupby("group")["pred"].mean()
    if len(rates) < 2 or rates.max() == 0:
        return float("nan")
    return float(rates.min() / rates.max())


def main():
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.impute import SimpleImputer
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score, accuracy_score
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.dummy import DummyClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.calibration import calibration_curve

    df = pd.read_csv(RAW_PATH)
    if len(df.columns) == 1:
        df = pd.read_csv(RAW_PATH, sep=";")
    checksum = md5(RAW_PATH)
    schema = {c: str(df[c].dtype) for c in df.columns}

    registry = {
        "dataset": "IBM HR Analytics Employee Attrition and Performance",
        "source_url": "https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset",
        "license_note": "Public GitHub mirror of Kaggle IBM HR Analytics dataset; original Kaggle page should be checked before external redistribution.",
        "download_timestamp": datetime.utcnow().isoformat() + "Z",
        "path": str(RAW_PATH.relative_to(ROOT)),
        "md5": checksum,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "schema": schema,
    }
    (ROOT / "real_dataset_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    save_md(ROOT / "real_dataset_registry.md", "# Real Dataset Registry\n\n" + "\n".join([f"- **{k}:** `{v}`" for k, v in registry.items() if k not in ["columns", "schema"]]) + "\n\n## Columns\n" + "\n".join([f"- `{c}`: `{schema[c]}`" for c in df.columns]))

    y = df["Attrition"].map({"Yes": 1, "No": 0}).astype(int)
    protected_cols = ["Age", "Gender", "MaritalStatus", "Department", "MonthlyIncome"]
    protected = df[protected_cols].copy()
    protected["age_band"] = pd.cut(protected["Age"], bins=[17, 30, 40, 50, 65], labels=["18-30", "31-40", "41-50", "51-65"])
    protected["compensation_band"] = pd.qcut(protected["MonthlyIncome"], q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])

    dropped = ["Attrition", "EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    direct_protected = ["Age", "Gender", "MaritalStatus"]
    X_base = df.drop(columns=[c for c in dropped + direct_protected if c in df.columns]).copy()
    X_base["OverTime"] = X_base["OverTime"].map({"Yes": 1, "No": 0}).astype(int)

    engineered = X_base.copy()
    engineered["overtime_intensity"] = engineered["OverTime"]
    engineered["burnout_proxy"] = (
        engineered["OverTime"] * 0.45
        + (engineered["WorkLifeBalance"].max() - engineered["WorkLifeBalance"]) / engineered["WorkLifeBalance"].max() * 0.25
        + (engineered["JobSatisfaction"].max() - engineered["JobSatisfaction"]) / engineered["JobSatisfaction"].max() * 0.15
        + engineered["DistanceFromHome"] / engineered["DistanceFromHome"].max() * 0.15
    )
    engineered["engagement_indicator"] = engineered[["JobInvolvement", "JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction"]].mean(axis=1)
    engineered["promotion_stagnation"] = engineered["YearsSinceLastPromotion"] / (engineered["YearsAtCompany"] + 1)
    engineered["workload_imbalance"] = engineered["OverTime"] * (5 - engineered["WorkLifeBalance"])
    engineered["salary_progression_proxy"] = engineered["MonthlyIncome"] / (engineered["TotalWorkingYears"] + 1)
    engineered["recognition_indicator"] = engineered[["JobInvolvement", "PerformanceRating", "PercentSalaryHike"]].rank(pct=True).mean(axis=1)
    engineered["promotion_interval_proxy"] = engineered["YearsSinceLastPromotion"]
    engineered["flexibility_proxy"] = (5 - engineered["WorkLifeBalance"]) + engineered["DistanceFromHome"] / engineered["DistanceFromHome"].std()
    engineered["satisfaction_composite"] = engineered[["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance"]].mean(axis=1)

    feature_df = engineered.copy()
    PROCESSED.joinpath("features_v3.0.csv").write_text(feature_df.to_csv(index=False), encoding="utf-8")
    protected.to_csv(PROCESSED / "protected_attributes.csv", index=False)
    y.to_csv(PROCESSED / "target_attrition.csv", index=False)

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))]), categorical_cols),
    ])
    X_all = preprocessor.fit_transform(feature_df)
    feature_names = list(preprocessor.get_feature_names_out())
    X_all_df = pd.DataFrame(X_all, columns=feature_names)
    X_all_df.to_csv(PROCESSED / "features_ml_encoded_v3.0.csv", index=False)

    X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
        X_all_df, y, protected, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train.to_csv(PROCESSED / "X_train_v3.0.csv", index=False)
    X_test.to_csv(PROCESSED / "X_test_v3.0.csv", index=False)
    y_train.to_csv(PROCESSED / "y_train_v3.0.csv", index=False)
    y_test.to_csv(PROCESSED / "y_test_v3.0.csv", index=False)

    preprocessing_report = f"""# Preprocessing Execution Report

## Dataset
- Rows: {df.shape[0]}
- Columns: {df.shape[1]}
- MD5: `{checksum}`

## Target
- Attrition positive rate: {y.mean():.4f}
- Positive cases: {int(y.sum())}
- Negative cases: {int((1-y).sum())}

## Missing values
- Total missing cells: {int(df.isna().sum().sum())}

## Dropped leakage/uninformative features
{chr(10).join('- `' + c + '`' for c in dropped if c in df.columns)}

## Direct protected attributes excluded from model inputs
{chr(10).join('- `' + c + '`' for c in direct_protected)}

## Encoded feature matrix
- Encoded columns: {X_all_df.shape[1]}
- Train rows: {len(X_train)}
- Test rows: {len(X_test)}

## Limitations
The IBM HR dataset is fictional/simulated and observational; results cannot establish real-world causal incentive effects.
"""
    save_md(ROOT / "preprocessing_execution_report.md", preprocessing_report)

    feature_report = "# Feature Engineering Results\n\n## Engineered features created\n" + "\n".join([
        "- `overtime_intensity`",
        "- `burnout_proxy`",
        "- `engagement_indicator`",
        "- `promotion_stagnation`",
        "- `workload_imbalance`",
        "- `salary_progression_proxy`",
        "- `recognition_indicator`",
        "- `promotion_interval_proxy`",
        "- `flexibility_proxy`",
        "- `satisfaction_composite`",
    ]) + f"\n\n## Output\n- Feature rows: {feature_df.shape[0]}\n- Raw + engineered columns before encoding: {feature_df.shape[1]}\n- Encoded ML columns: {X_all_df.shape[1]}\n\n## Caution\nThese are behavioral proxies, not validated psychometric constructs.\n"
    save_md(ROOT / "feature_engineering_results.md", feature_report)

    cluster_feature_cols = [c for c in X_all_df.columns if any(s in c for s in ["burnout_proxy", "engagement_indicator", "promotion_stagnation", "workload_imbalance", "salary_progression_proxy", "recognition_indicator", "flexibility_proxy", "satisfaction_composite", "OverTime", "JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction"])]
    X_cluster = X_all_df[cluster_feature_cols].copy()
    cluster_rows = []
    labels_store = {}
    for algo in ["kmeans", "gmm", "agglomerative"]:
        for k in [2, 3, 4, 5, 6]:
            if algo == "kmeans":
                model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
                labels = model.fit_predict(X_cluster)
            elif algo == "gmm":
                model = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=5)
                labels = model.fit_predict(X_cluster)
            else:
                model = AgglomerativeClustering(n_clusters=k)
                labels = model.fit_predict(X_cluster)
            cluster_rows.append({
                "algorithm": algo, "k": k,
                "silhouette": float(silhouette_score(X_cluster, labels)),
                "davies_bouldin": float(davies_bouldin_score(X_cluster, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(X_cluster, labels)),
                "n_clusters_found": int(len(set(labels))),
                "noise_fraction": 0.0,
            })
            labels_store[(algo, k)] = labels
    for eps in [0.5, 1.0, 1.5, 2.0, 3.0]:
        model = DBSCAN(eps=eps, min_samples=10)
        labels = model.fit_predict(X_cluster)
        non_noise = labels != -1
        n_clusters = len(set(labels) - {-1})
        row = {"algorithm": "dbscan", "k": None, "eps": eps, "n_clusters_found": int(n_clusters), "noise_fraction": float((labels == -1).mean())}
        if n_clusters >= 2 and non_noise.sum() > n_clusters:
            row.update({"silhouette": float(silhouette_score(X_cluster[non_noise], labels[non_noise])), "davies_bouldin": float(davies_bouldin_score(X_cluster[non_noise], labels[non_noise])), "calinski_harabasz": float(calinski_harabasz_score(X_cluster[non_noise], labels[non_noise]))})
        else:
            row.update({"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan})
        cluster_rows.append(row)
        labels_store[("dbscan", eps)] = labels
    cluster_df = pd.DataFrame(cluster_rows).sort_values("silhouette", ascending=False, na_position="last")
    cluster_df.to_csv(CLUSTER_OUT / "clustering_metrics.csv", index=False)
    best = cluster_df.dropna(subset=["silhouette"]).iloc[0]
    if best["algorithm"] == "dbscan":
        best_labels = labels_store.get(("dbscan", float(best["eps"])))
    else:
        best_labels = labels_store.get((best["algorithm"], int(best["k"])))
    if best_labels is None or len(set(best_labels) - {-1}) < 2:
        best_labels = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=20).fit_predict(X_cluster)
    pd.DataFrame({"cluster": best_labels}).to_csv(PROCESSED / "cluster_assignments_v1.csv", index=False)

    stability = []
    for seed in [0, 7, 21, 42, 99, 123, 2024]:
        labels = KMeans(n_clusters=int(best["k"]) if pd.notna(best["k"]) else 3, random_state=seed, n_init=20).fit_predict(X_cluster)
        stability.append({"seed": seed, "silhouette": float(silhouette_score(X_cluster, labels))})
    stability_df = pd.DataFrame(stability)
    stability_df.to_csv(CLUSTER_OUT / "seed_stability.csv", index=False)

    profiles = feature_df.assign(cluster=best_labels).groupby("cluster")[["burnout_proxy", "engagement_indicator", "promotion_stagnation", "workload_imbalance", "salary_progression_proxy", "recognition_indicator", "flexibility_proxy", "satisfaction_composite"]].mean()
    profiles.to_csv(CLUSTER_OUT / "cluster_profiles.csv")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=cluster_df.dropna(subset=["silhouette"]).head(12), x="silhouette", y="algorithm", hue="k")
    plt.title("Top Clustering Configurations by Silhouette")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "clustering_silhouette_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.heatmap(profiles, cmap="vlag", center=profiles.to_numpy().mean(), annot=False)
    plt.title("Cluster Profile Heatmap")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cluster_profile_heatmap.png", dpi=200)
    plt.close()

    models = {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, min_samples_leaf=3),
    }
    optional_failures = []
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_estimators=200, max_depth=3, learning_rate=0.05)
    except Exception as e:
        optional_failures.append(f"xgboost unavailable: {e}")
    try:
        from catboost import CatBoostClassifier
        models["catboost"] = CatBoostClassifier(random_seed=RANDOM_STATE, verbose=False, iterations=200, depth=4, learning_rate=0.05)
    except Exception as e:
        optional_failures.append(f"catboost unavailable: {e}")

    salary_cols = [c for c in X_all_df.columns if any(s in c for s in ["MonthlyIncome", "salary_progression_proxy", "PercentSalaryHike", "StockOptionLevel"])]
    benchmark_sets = {"all_features": list(X_all_df.columns), "salary_only": salary_cols}
    model_rows = []
    preds = {}
    for set_name, cols in benchmark_sets.items():
        Xt, Xs = X_train[cols], X_test[cols]
        for name, model in models.items():
            try:
                model.fit(Xt, y_train)
                proba = model.predict_proba(Xs)[:, 1]
                pred = (proba >= 0.5).astype(int)
                model_rows.append({
                    "feature_set": set_name, "model": name,
                    "roc_auc": safe_auc(y_test, proba),
                    "roc_auc_ci_low": bootstrap_ci(y_test, proba, roc_auc_score)[0],
                    "roc_auc_ci_high": bootstrap_ci(y_test, proba, roc_auc_score)[1],
                    "average_precision": float(average_precision_score(y_test, proba)),
                    "brier_score": float(brier_score_loss(y_test, proba)),
                    "accuracy": float(accuracy_score(y_test, pred)),
                    "f1": float(f1_score(y_test, pred, zero_division=0)),
                })
                if set_name == "all_features":
                    preds[name] = (model, proba, pred)
            except Exception as e:
                optional_failures.append(f"{name}/{set_name} failed: {e}")
    model_df = pd.DataFrame(model_rows).sort_values(["feature_set", "roc_auc"], ascending=[True, False])
    model_df.to_csv(MODEL_OUT / "model_benchmark_metrics.csv", index=False)
    model_df.to_csv(TABLE_DIR / "model_benchmark_metrics.csv", index=False)

    best_model_row = model_df[model_df["feature_set"] == "all_features"].sort_values("roc_auc", ascending=False).iloc[0]
    best_model_name = best_model_row["model"]
    best_model, best_proba, best_pred = preds[best_model_name]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(models["logistic_regression"], X_train, y_train, cv=cv, scoring="roc_auc")
    pd.DataFrame({"fold": range(1, 6), "logistic_cv_roc_auc": cv_scores}).to_csv(MODEL_OUT / "cross_validation_scores.csv", index=False)

    frac_pos, mean_pred = calibration_curve(y_test, best_proba, n_bins=8, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, marker="o", label=best_model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed attrition rate")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "calibration_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, best_pred)
    plt.title(f"Confusion Matrix: {best_model_name}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix_best_model.png", dpi=200)
    plt.close()

    perm = permutation_importance(best_model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE)
    importance = pd.DataFrame({"feature": X_test.columns, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
    importance.to_csv(MODEL_OUT / "permutation_importance.csv", index=False)
    importance.head(20).to_csv(TABLE_DIR / "top_permutation_importance.csv", index=False)
    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance.head(15), x="importance_mean", y="feature")
    plt.title("Permutation Importance for Best Model")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "permutation_importance_best_model.png", dpi=200)
    plt.close()

    shap_status = "not_run"
    shap_message = "SHAP package unavailable or model unsupported; permutation importance used as fallback."
    try:
        import shap
        if best_model_name in ["xgboost", "catboost", "random_forest"]:
            sample = X_test.sample(min(150, len(X_test)), random_state=RANDOM_STATE)
            explainer = shap.TreeExplainer(best_model)
            vals = explainer.shap_values(sample)
            if isinstance(vals, list):
                vals = vals[-1]
            shap_imp = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": np.abs(vals).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
            shap_imp.to_csv(MODEL_OUT / "shap_importance.csv", index=False)
            plt.figure()
            shap.summary_plot(vals, sample, show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(FIG_DIR / "shap_summary_best_model.png", dpi=200, bbox_inches="tight")
            plt.close()
            shap_status = "completed"
            shap_message = f"SHAP completed for {best_model_name} on {len(sample)} test samples."
    except Exception as e:
        shap_message = f"SHAP failed: {e}; permutation importance used as fallback."

    fairness_rows = []
    for attr in ["Gender", "age_band", "Department", "compensation_band", "MaritalStatus"]:
        fairness_rows.append({"audit_type": "prediction_positive_rate", "attribute": attr, "disparate_impact_ratio": disparate_impact(best_pred, prot_test[attr])})
        tab = pd.crosstab(prot_test[attr], best_pred)
        fairness_rows.append({"audit_type": "prediction_association", "attribute": attr, "cramers_v": cramers_v(tab)})
    fairness_df = pd.DataFrame(fairness_rows)
    fairness_df.to_csv(MODEL_OUT / "fairness_audit.csv", index=False)
    fairness_df.to_csv(TABLE_DIR / "fairness_audit.csv", index=False)

    cluster_fair_rows = []
    cluster_assign_test = pd.Series(best_labels, index=X_all_df.index).loc[X_test.index]
    for attr in ["Gender", "age_band", "Department", "compensation_band", "MaritalStatus"]:
        cluster_fair_rows.append({"attribute": attr, "cluster_cramers_v": cramers_v(pd.crosstab(prot_test[attr], cluster_assign_test))})
    pd.DataFrame(cluster_fair_rows).to_csv(CLUSTER_OUT / "cluster_demographic_association.csv", index=False)

    plt.figure(figsize=(8, 5))
    plot_df = prot_test.copy()
    plot_df["predicted_high_risk"] = best_pred
    sns.barplot(data=plot_df, x="Gender", y="predicted_high_risk")
    plt.title("Predicted High-Risk Rate by Gender")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fairness_gender_predicted_positive_rate.png", dpi=200)
    plt.close()

    robustness = []
    for seed in [0, 7, 21, 42, 99, 123, 2024]:
        Xt, Xs, yt, ys = train_test_split(X_all_df, y, test_size=0.2, stratify=y, random_state=seed)
        m = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        m.fit(Xt, yt)
        pr = m.predict_proba(Xs)[:, 1]
        robustness.append({"seed": seed, "logistic_roc_auc": safe_auc(ys, pr), "logistic_average_precision": float(average_precision_score(ys, pr))})
    robustness_df = pd.DataFrame(robustness)
    robustness_df.to_csv(MODEL_OUT / "seed_robustness.csv", index=False)
    robustness_df.to_csv(TABLE_DIR / "seed_robustness.csv", index=False)

    plt.figure(figsize=(7, 4))
    sns.lineplot(data=robustness_df, x="seed", y="logistic_roc_auc", marker="o")
    plt.title("Seed Sensitivity: Logistic ROC-AUC")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "seed_sensitivity_logistic_auc.png", dpi=200)
    plt.close()

    cluster_df.to_csv(TABLE_DIR / "clustering_metrics.csv", index=False)
    profiles.to_csv(TABLE_DIR / "cluster_profiles.csv")


if __name__ == "__main__":
    main()
