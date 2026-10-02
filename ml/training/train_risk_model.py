"""Main Machine Learning Training and Selection Pipeline for Kidney Stone Risk prediction.

Loads processed train/validation/test datasets, applies pre-fitted Phase 3
ColumnTransformers, performs cross-validation & hyperparameter search, selects
the best model (ROC-AUC/F1/MCC based), and generates charts and reports.
"""

import sys
from pathlib import Path
import json
import logging
import time
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, precision_score, recall_score

sys.path.append(str(Path(__file__).resolve().parents[1] / "preprocessing"))

from model import MODELS, PARAM_GRIDS
from evaluate import evaluate_ml_model, evaluate_repeated_cv, evaluate_models_paired_cv, save_ml_charts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("MLRiskTrainer")


class MLRiskTrainer:
    """Trainer class for optimizing, comparing, and selecting tabular risk prediction models."""

    def __init__(
        self,
        processed_dir: Path,
        artifacts_dir: Path,
        models_dir: Path,
        charts_dir: Path,
        reports_dir: Path
    ):
        self.processed_dir = Path(processed_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.models_dir = Path(models_dir)
        self.charts_dir = Path(charts_dir)
        self.reports_dir = Path(reports_dir)

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_splits(self) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, List[str], str]:
        """Loads split datasets without transforming features to prevent leakage."""
        logger.info("Loading raw split datasets...")
        train_df = pd.read_csv(self.processed_dir / "train.csv")
        val_df = pd.read_csv(self.processed_dir / "validation.csv")
        test_df = pd.read_csv(self.processed_dir / "test.csv")

        # Determine target column
        target_col = "target"
        for col in ["Class", "target", "label"]:
            if col in train_df.columns:
                target_col = col
                break

        # Separate features and labels
        X_train_raw = train_df.drop(columns=[target_col])
        y_train = train_df[target_col].values

        X_val_raw = val_df.drop(columns=[target_col])
        y_val = val_df[target_col].values

        X_test_raw = test_df.drop(columns=[target_col])
        y_test = test_df[target_col].values

        # Retrieve feature names
        num_cols = list(X_train_raw.select_dtypes(include=[np.number]).columns)
        cat_cols = list(X_train_raw.select_dtypes(include=['object', 'category']).columns)
        feature_names = num_cols + cat_cols

        logger.info(f"Loaded raw splits - Train: {X_train_raw.shape}, Val: {X_val_raw.shape}, Test: {X_test_raw.shape}, features: {feature_names}")
        return X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test, feature_names, target_col

    def train_and_compare(self) -> Dict[str, Any]:
        """Runs hyperparameter search on all classifiers inside sklearn Pipelines and selects optimal model."""
        from sklearn.pipeline import Pipeline
        from tabular_preprocessor import TabularPreprocessor

        X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test, feature_names, target_col = self.load_raw_splits()

        logger.info("Starting Cross-Validated Hyperparameter Search using sklearn Pipelines...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        comparison_records: List[Dict[str, Any]] = []
        trained_pipelines: Dict[str, Any] = {}

        for mname, clf in MODELS.items():
            if mname == "XGBoost":
                class_counts = np.bincount(y_train.astype(int))
                if len(class_counts) == 2 and class_counts[1] > 0:
                    clf.set_params(scale_pos_weight=class_counts[0] / class_counts[1])

            # Instantiate a fresh preprocessor ColumnTransformer for each model pipeline
            preprocessor = TabularPreprocessor(target_column=target_col)
            preprocessor.numerical_cols = list(X_train_raw.select_dtypes(include=[np.number]).columns)
            preprocessor.categorical_cols = list(X_train_raw.select_dtypes(include=['object', 'category']).columns)
            preprocessor_transformer = preprocessor.build_pipeline()

            # Construct full pipeline: Preprocessing + Classifier
            model_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor_transformer),
                ('classifier', clf)
            ])

            # Prefix parameter grid keys for the pipeline's classifier step
            grid = {f"classifier__{k}": v for k, v in PARAM_GRIDS[mname].items()}
            logger.info(f"Tuning hyper-parameters for {mname} inside Pipeline...")

            start_time = time.time()
            grid_search = GridSearchCV(
                estimator=model_pipeline,
                param_grid=grid,
                cv=cv,
                scoring="roc_auc",
                n_jobs=1
            )
            # Fits preprocessor strictly on each CV training fold
            grid_search.fit(X_train_raw, y_train)
            elapsed_time = time.time() - start_time

            best_pipeline = grid_search.best_estimator_
            trained_pipelines[mname] = best_pipeline

            # Predict on validation split using the trained pipeline
            val_preds = best_pipeline.predict(X_val_raw)
            val_probs = best_pipeline.predict_proba(X_val_raw)[:, 1] if hasattr(best_pipeline, "predict_proba") else val_preds

            # Metrics
            roc_auc = float(roc_auc_score(y_val, val_probs))
            f1 = float(f1_score(y_val, val_preds, zero_division=0))
            mcc = float(matthews_corrcoef(y_val, val_preds))
            acc = float(best_pipeline.score(X_val_raw, y_val))

            record = {
                "Model": mname,
                "Accuracy": round(acc, 4),
                "Precision": round(float(precision_score(y_val, val_preds, zero_division=0)), 4),
                "Recall": round(float(recall_score(y_val, val_preds, zero_division=0)), 4),
                "F1": round(f1, 4),
                "ROC-AUC": round(roc_auc, 4),
                "MCC": round(mcc, 4),
                "Training Time": round(elapsed_time, 4)
            }
            comparison_records.append(record)
            logger.info(f"{mname} best validation performance: ROC-AUC={roc_auc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")

        # Save comparison results CSV
        comparison_df = pd.DataFrame(comparison_records)
        comparison_csv_path = self.models_dir / "comparison_results.csv"
        comparison_df.to_csv(comparison_csv_path, index=False)
        logger.info(f"Saved model comparison table to {comparison_csv_path}")

        # Plot Model Comparison Bar Chart
        self.plot_comparison_chart(comparison_df)

        # Select Best Model based on ROC-AUC, F1, and MCC
        comparison_df["rank_score"] = comparison_df["ROC-AUC"] + comparison_df["F1"] + comparison_df["MCC"]
        best_row = comparison_df.sort_values(by="rank_score", ascending=False).iloc[0]
        best_mname = best_row["Model"]
        best_pipeline = trained_pipelines[best_mname]
        best_classifier = best_pipeline.named_steps['classifier']
        fitted_preprocessor = best_pipeline.named_steps['preprocessor']

        logger.info(f"--> Selected BEST model: {best_mname} with rank score sum: {best_row['rank_score']:.4f}")

        # Save optimal preprocessing pipeline and model
        best_model_path = self.models_dir / "kidney_risk_model.pkl"
        joblib.dump(best_classifier, best_model_path)
        logger.info(f"Saved optimal ML classifier pkl to {best_model_path}")

        pipe_path = self.artifacts_dir / "preprocessing_pipeline.pkl"
        joblib.dump(fitted_preprocessor, pipe_path)
        logger.info(f"Saved fitted preprocessing ColumnTransformer to {pipe_path}")

        if 'num' in fitted_preprocessor.named_transformers_:
            scaler = fitted_preprocessor.named_transformers_['num'].named_steps.get('scaler')
            if scaler is not None:
                joblib.dump(scaler, self.artifacts_dir / "scaler.pkl")

        # Primary Statistical Evaluation: RepeatedStratifiedKFold (5 splits, 10 repeats = 50 folds)
        # Evaluating both XGBoost and plain Logistic Regression baseline on identical folds
        logger.info("Executing 50-fold RepeatedStratifiedKFold Cross-Validation (5 splits, 10 repeats, seed=42) on identical folds...")
        dataset_path = Path(__file__).resolve().parents[1] / "datasets" / "kidneyData.csv"
        raw_full_df = pd.read_csv(dataset_path)
        cleaner = TabularPreprocessor(target_column=target_col)
        cleaned_full_df = cleaner.clean_data(raw_full_df)
        X_full = cleaned_full_df.drop(columns=[target_col])
        y_full = cleaned_full_df[target_col].values

        def xgb_pipeline_factory():
            p = TabularPreprocessor(target_column=target_col)
            p.numerical_cols = list(X_full.select_dtypes(include=[np.number]).columns)
            p.categorical_cols = list(X_full.select_dtypes(include=['object', 'category']).columns)
            from sklearn.base import clone
            return Pipeline(steps=[
                ('preprocessor', p.build_pipeline()),
                ('classifier', clone(best_classifier))
            ])

        def logreg_pipeline_factory():
            p = TabularPreprocessor(target_column=target_col)
            p.numerical_cols = list(X_full.select_dtypes(include=[np.number]).columns)
            p.categorical_cols = list(X_full.select_dtypes(include=['object', 'category']).columns)
            from sklearn.linear_model import LogisticRegression
            return Pipeline(steps=[
                ('preprocessor', p.build_pipeline()),  # StandardScaler included inside ColumnTransformer
                ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
            ])

        cv_models = {
            "XGBoost": xgb_pipeline_factory,
            "LogisticRegression": logreg_pipeline_factory
        }

        cv_summaries, cv_comparison_df, fold_results_df = evaluate_models_paired_cv(
            models_dict=cv_models,
            X=X_full,
            y=y_full,
            n_splits=5,
            n_repeats=10,
            random_state=42
        )

        # Save fold-level results CSV (100 rows total: 50 folds x 2 models)
        fold_csv_path = self.models_dir / "cv_fold_metrics.csv"
        fold_results_df.to_csv(fold_csv_path, index=False)
        logger.info(f"Saved fold-level metrics (50 folds x 2 models) to {fold_csv_path}")

        # Save comparison results CSV
        comparison_csv_path = self.models_dir / "comparison_results.csv"
        cv_comparison_df.to_csv(comparison_csv_path, index=False)
        baseline_comp_path = self.models_dir / "baseline_model_comparison.csv"
        cv_comparison_df.to_csv(baseline_comp_path, index=False)
        logger.info(f"Saved comparison results CSV to {comparison_csv_path} and {baseline_comp_path}")

        # Save primary metrics JSON
        metrics_payload = {
            "evaluation_strategy": "RepeatedStratifiedKFold (5 splits, 10 repeats = 50 folds, seed=42)",
            "total_folds": 50,
            "models": cv_summaries,
            # Top-level aliases pointing to primary model (XGBoost) for backward compatibility
            **cv_summaries["XGBoost"]
        }
        metrics_path = self.models_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=4)
        logger.info(f"Saved primary 50-fold CV metrics to {metrics_path}")

        for mname, msummary in cv_summaries.items():
            logger.info(
                f"Repeated CV (50 folds) Results for {mname}:\n"
                f"  AUC:       {msummary['auc_mean']:.4f} ± {msummary['auc_std']:.4f}\n"
                f"  Recall:    {msummary['recall_mean']:.4f} ± {msummary['recall_std']:.4f}\n"
                f"  Precision: {msummary['precision_mean']:.4f} ± {msummary['precision_std']:.4f}\n"
                f"  F1-Score:  {msummary['f1_mean']:.4f} ± {msummary['f1_std']:.4f}\n"
                f"  Accuracy:  {msummary['accuracy_mean']:.4f} ± {msummary['accuracy_std']:.4f}"
            )

        # Save class mapping
        mapping_path = self.models_dir / "class_mapping.json"
        class_mapping = {0: "Low Risk", 1: "High Risk"}
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(class_mapping, f, indent=4)

        # Generate and save diagnostic charts
        X_test_trans = fitted_preprocessor.transform(X_test_raw)
        _, test_preds, test_probs = evaluate_ml_model(best_classifier, X_test_trans, y_test, ["Low Risk", "High Risk"])
        save_ml_charts(y_test, test_preds, test_probs, ["Low Risk", "High Risk"], self.charts_dir)

        # Feature Importance for best model
        self.save_feature_importance(best_classifier, feature_names)

        # Write comprehensive model report with paired comparison without automatically declaring a winner
        self.write_model_report(cv_summaries, cv_comparison_df, feature_names)

        logger.info("ML Risk Model Pipeline completed successfully.")
        return metrics_payload

    def plot_comparison_chart(self, df: pd.DataFrame) -> None:
        """Plots and saves model comparison bar chart."""
        sns.set_theme(style="whitegrid")
        df_melt = pd.melt(df, id_vars=["Model"], value_vars=["Accuracy", "F1", "ROC-AUC"], var_name="Metric", value_name="Score")

        plt.figure(figsize=(8, 5))
        sns.barplot(data=df_melt, x="Model", y="Score", hue="Metric", palette="muted")
        plt.title("Kidney Stone Risk Model Validation Comparison", fontsize=13, fontweight="bold", pad=15)
        plt.ylim(0, 1.1)
        plt.tight_layout()
        chart_path = self.charts_dir / "model_comparison.png"
        plt.savefig(chart_path, dpi=300)
        plt.close()

    def save_feature_importance(self, model: Any, feature_names: List[str]) -> None:
        """Computes and plots feature importances for the best model."""
        importances = []
        if hasattr(model, "feature_importances_"):
            importances = list(model.feature_importances_)
        elif hasattr(model, "coef_"):
            importances = list(np.abs(model.coef_[0]))
        else:
            logger.warning("Selected model does not expose feature_importances_ or coef_")
            return

        feat_imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        # Save feature_importance.json
        feat_path = self.models_dir / "feature_importance.json"
        feat_imp_df.to_json(feat_path, orient="records", indent=4)
        logger.info(f"Saved feature importance JSON to {feat_path}")

        # Plot feature_importance.png
        plt.figure(figsize=(8, 5))
        sns.barplot(data=feat_imp_df.head(10), x="importance", y="feature", palette="viridis", hue="feature", legend=False)
        plt.title("Urine Chemistry Feature Importance Ranking", fontweight="bold", fontsize=13, pad=15)
        plt.xlabel("Importance Score", fontweight="bold")
        plt.ylabel("Feature", fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.charts_dir / "feature_importance.png", dpi=300)
        plt.close()

    def write_model_report(
        self,
        cv_summaries: Dict[str, Any],
        comparison_df: pd.DataFrame,
        features: List[str]
    ) -> None:
        """Generates model_b_report.md comparing XGBoost and LogisticRegression baseline without automatically declaring a winner."""
        report_path = self.reports_dir / "model_b_report.md"

        feature_summary = ", ".join(f"`{f}`" for f in features)
        
        xgb_summary = cv_summaries.get("XGBoost", {})
        logreg_summary = cv_summaries.get("LogisticRegression", {})
        
        xgb_cm = xgb_summary.get("aggregated_confusion_matrix", [[0, 0], [0, 0]])
        logreg_cm = logreg_summary.get("aggregated_confusion_matrix", [[0, 0], [0, 0]])

        report_md = f"""# Phase 5 Report — Model B: Kidney Stone Risk Prediction (ML)

**Models Evaluated:** XGBoost vs. Logistic Regression (Baseline with StandardScaler Pipeline)
**Dataset:** Clinical Urine Analysis Dataset (79 observations)
**Evaluation Protocol:** 50-Fold Repeated Stratified Cross-Validation (`RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)`) on Identical Folds
**Statistical Uncertainty:** Non-Parametric Patient-Level (Cluster) Bootstrap (2,000 iterations, 95% CI)
**Target Variable:** `target` (0: Low Risk, 1: High Risk)

---

## 📊 Cross-Validation Performance Comparison (50 Identical Folds with 95% Bootstrap CIs)

Both models were evaluated on the **exact same 50 cross-validation folds** with fold-level preprocessing (imputation + scaling fit strictly on each fold's training split):

| Model | ROC-AUC (Mean ± SD [95% CI]) | Recall (Mean ± SD [95% CI]) | Precision (Mean ± SD) | F1-Score (Mean ± SD) | Accuracy (Mean ± SD) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **{xgb_summary.get('auc_mean', 0):.4f} ± {xgb_summary.get('auc_std', 0):.4f}** (`{xgb_summary.get('auc_ci_95', 'N/A')}`) | **{xgb_summary.get('recall_mean', 0):.4f} ± {xgb_summary.get('recall_std', 0):.4f}** (`{xgb_summary.get('recall_ci_95', 'N/A')}`) | **{xgb_summary.get('precision_mean', 0):.4f} ± {xgb_summary.get('precision_std', 0):.4f}** | **{xgb_summary.get('f1_mean', 0):.4f} ± {xgb_summary.get('f1_std', 0):.4f}** | **{xgb_summary.get('accuracy_mean', 0) * 100:.2f}% ± {xgb_summary.get('accuracy_std', 0) * 100:.2f}%** |
| **Logistic Regression** | **{logreg_summary.get('auc_mean', 0):.4f} ± {logreg_summary.get('auc_std', 0):.4f}** (`{logreg_summary.get('auc_ci_95', 'N/A')}`) | **{logreg_summary.get('recall_mean', 0):.4f} ± {logreg_summary.get('recall_std', 0):.4f}** (`{logreg_summary.get('recall_ci_95', 'N/A')}`) | **{logreg_summary.get('precision_mean', 0):.4f} ± {logreg_summary.get('precision_std', 0):.4f}** | **{logreg_summary.get('f1_mean', 0):.4f} ± {logreg_summary.get('f1_std', 0):.4f}** | **{logreg_summary.get('accuracy_mean', 0) * 100:.2f}% ± {logreg_summary.get('accuracy_std', 0) * 100:.2f}%** |

- **Total Folds:** 50 folds per model (100 fold evaluations total).
- **Bootstrap Sampling Unit:** Individual patient/row ($N=79$). Repeated evaluations on the same patient across the 10 CV repeats are clustered and evaluated together per repeat rather than treated as independent observations.
- **Fold-by-Fold Results:** Stored in `ml/models/cv_fold_metrics.csv`.
- **Model Comparison Table:** Stored in `ml/models/comparison_results.csv` and `ml/models/baseline_model_comparison.csv`.

---

## 🔍 Aggregated Confusion Matrices (50 Folds, Total Out-of-Fold Predictions = 790)

*Note: The 790 predictions represent 10 repeated out-of-fold evaluations of the 79 clinical observations across 10 repeats.*

### XGBoost Aggregated Confusion Matrix (Rows: True, Columns: Predicted)
```
{xgb_cm}
```

### Logistic Regression Aggregated Confusion Matrix (Rows: True, Columns: Predicted)
```
{logreg_cm}
```

---

## 🧬 Feature Summary & Clinical Predictors

The clinical urine chemistry features utilized for risk scoring:
{feature_summary}

- Key indicators such as calcium (`calc`), specific gravity (`gravity`), and pH play primary roles in scoring patient stone forming risk.
- Feature importance visualization and ranking list are saved to `ml/outputs/charts/feature_importance.png` and `ml/models/feature_importance.json`.
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        logger.info(f"Generated Phase 5 report at {report_path}")


def main():
    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = base_dir / "processed"
    artifacts_dir = base_dir / "artifacts"
    models_dir = base_dir / "models"
    charts_dir = base_dir / "outputs" / "charts"
    reports_dir = base_dir / "outputs" / "reports"

    trainer = MLRiskTrainer(
        processed_dir=processed_dir,
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
        charts_dir=charts_dir,
        reports_dir=reports_dir
    )
    trainer.train_and_compare()


if __name__ == "__main__":
    main()
