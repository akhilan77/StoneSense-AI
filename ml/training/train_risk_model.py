"""Main Machine Learning Training, Nested CV Optimization, and Model Selection Pipeline.

Implements:
1. Nested Repeated Stratified K-Fold CV (5 folds x 10 repeats = 50 folds) with sklearn Pipeline.
2. Row-level Bootstrap 95% Confidence Intervals for ROC-AUC, F1, MCC, Sensitivity, and Specificity.
3. Non-parametric Permutation Significance Testing (1000 permutations, n_jobs=-1).
4. 1-Standard-Error Model Selection Rule (with Nadeau-Bengio corrected SE, baseline: Logistic Regression).
5. Data integrity checks (duplicate checking failing loudly, class balance verification).
6. Preserves separate artifacts (candidate_risk_model.pkl, preprocessing_pipeline.pkl, scaler.pkl) without overwriting legacy kidney_risk_model.pkl.
"""

import sys
from pathlib import Path
import json
import logging
import time
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.base import clone

# Add paths for preprocessing and training modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "ml" / "preprocessing"))
sys.path.append(str(PROJECT_ROOT / "ml" / "training"))

from tabular_preprocessor import TabularPreprocessor, check_duplicate_rows, check_class_balance
from model import MODELS, PARAM_GRIDS, MODEL_COMPLEXITY_ORDER
from evaluate import (
    evaluate_nested_cv,
    compute_row_level_bootstrap_cis,
    run_permutation_test,
    select_model_1se_rule,
    save_ml_charts
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("MLRiskTrainer")


class MLRiskTrainer:
    """Trainer class for nested CV optimization, statistical comparison, and selection of risk models."""

    def __init__(
        self,
        dataset_path: Path,
        artifacts_dir: Path,
        models_dir: Path,
        charts_dir: Path,
        reports_dir: Path
    ):
        self.dataset_path = Path(dataset_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.models_dir = Path(models_dir)
        self.charts_dir = Path(charts_dir)
        self.reports_dir = Path(reports_dir)

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def load_and_validate_dataset(self, drop_duplicates: bool = True) -> Tuple[pd.DataFrame, np.ndarray, List[str], str]:
        """Loads dataset, executes strict data integrity checks, and prepares feature matrix."""
        logger.info(f"Loading raw clinical dataset from {self.dataset_path}...")
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        raw_df = pd.read_csv(self.dataset_path)
        if 'Unnamed: 0' in raw_df.columns:
            raw_df = raw_df.drop(columns=['Unnamed: 0'])

        target_col = "target"
        for candidate in ["Class", "target", "label"]:
            if candidate in raw_df.columns:
                target_col = candidate
                break

        # 1. Strict Duplicate Check (fails loudly if drop_duplicates=False)
        cleaned_df = check_duplicate_rows(raw_df, drop_duplicates=drop_duplicates)

        # 2. Strict Class Balance Check (fails loudly if invalid)
        check_class_balance(cleaned_df, target_column=target_col)

        X = cleaned_df.drop(columns=[target_col])
        y = cleaned_df[target_col].values.astype(int)

        feature_names = list(X.columns)
        logger.info(f"Dataset validated successfully: {len(cleaned_df)} observations, features: {feature_names}")
        return X, y, feature_names, target_col

    def build_pipeline_factory(self, model_name: str, target_col: str, feature_names: List[str]):
        """Returns a factory function creating an isolated sklearn Pipeline."""
        def pipeline_factory():
            preprocessor = TabularPreprocessor(target_column=target_col)
            preprocessor.numerical_cols = feature_names
            preprocessor.categorical_cols = []
            transformer = preprocessor.build_pipeline()

            clf = clone(MODELS[model_name])
            return Pipeline(steps=[
                ('preprocessor', transformer),
                ('classifier', clf)
            ])
        return pipeline_factory

    def run_nested_evaluation_and_comparison(self) -> Dict[str, Any]:
        """Runs Nested Repeated Stratified CV, Bootstrap CIs, Permutation Tests, and 1-SE selection."""
        X, y, feature_names, target_col = self.load_and_validate_dataset(drop_duplicates=True)
        N = len(y)

        logger.info("=================================================================")
        logger.info("Starting Phase 1 Tabular ML Rigorous Evaluation Protocol")
        logger.info("Protocol: Nested RepeatedStratifiedKFold (5 splits x 10 repeats = 50 outer folds)")
        logger.info("=================================================================")

        nested_cv_results: Dict[str, Any] = {}
        comparison_records: List[Dict[str, Any]] = []
        all_fold_records: List[Dict[str, Any]] = []
        best_hyperparams_per_model: Dict[str, Dict[str, Any]] = {}

        for mname in MODELS.keys():
            logger.info(f"--> Running Nested CV for {mname}...")
            pipe_factory = self.build_pipeline_factory(mname, target_col, feature_names)
            grid = PARAM_GRIDS[mname]

            cv_summary, fold_df, oof_probs, oof_preds = evaluate_nested_cv(
                model_factory_or_pipeline=pipe_factory,
                param_grid=grid,
                X=X,
                y=y,
                n_splits=5,
                n_repeats=10,
                random_state=42,
                scoring="roc_auc"
            )

            for rec in cv_summary["fold_results"]:
                rec_copy = dict(rec)
                rec_copy["model"] = mname
                all_fold_records.append(rec_copy)

            # Compute row-level bootstrap 95% CIs (averaging across repeats per row)
            logger.info(f"Computing row-level Bootstrap 95% CIs (2,000 resamples) for {mname}...")
            bootstrap_cis = compute_row_level_bootstrap_cis(
                y_true=y,
                oof_probs_matrix=oof_probs,
                n_bootstraps=2000,
                confidence_level=0.95,
                random_state=42
            )

            # Find overall best hyperparameters on the full dataset for permutation testing and candidate fit
            full_search = GridSearchCV(
                estimator=pipe_factory(),
                param_grid=grid,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring="roc_auc",
                n_jobs=1
            )
            full_search.fit(X, y)
            best_params = full_search.best_params_
            best_hyperparams_per_model[mname] = best_params

            # Permutation significance test (1000 permutations, fixed best params, n_jobs=-1)
            best_fixed_pipe = full_search.best_estimator_
            perm_result = run_permutation_test(
                model_pipeline=best_fixed_pipe,
                X=X,
                y=y,
                n_permutations=1000,
                n_splits=5,
                random_state=42,
                n_jobs=-1,
                scoring="roc_auc"
            )

            # Record metrics
            comp_record = {
                "Model": mname,
                "Complexity Rank": MODEL_COMPLEXITY_ORDER[mname],
                "ROC-AUC Mean": cv_summary["auc_mean"],
                "ROC-AUC SD": cv_summary["auc_std"],
                "Nadeau-Bengio SE": cv_summary["auc_se_nadeau_bengio"],
                "Repeat Means SE": cv_summary["auc_se_repeats"],
                "ROC-AUC (Mean ± SD)": f"{cv_summary['auc_mean']:.4f} ± {cv_summary['auc_std']:.4f}",
                "ROC-AUC 95% CI": bootstrap_cis["roc_auc"]["ci_str"],
                "F1 (Mean ± SD)": f"{cv_summary['f1_mean']:.4f} ± {cv_summary['f1_std']:.4f}",
                "F1 95% CI": bootstrap_cis["f1"]["ci_str"],
                "MCC (Mean ± SD)": f"{cv_summary['mcc_mean']:.4f} ± {cv_summary['mcc_std']:.4f}",
                "MCC 95% CI": bootstrap_cis["mcc"]["ci_str"],
                "Sensitivity 95% CI": bootstrap_cis["sensitivity"]["ci_str"],
                "Specificity 95% CI": bootstrap_cis["specificity"]["ci_str"],
                "Accuracy (Mean ± SD)": f"{cv_summary['accuracy_mean'] * 100:.2f}% ± {cv_summary['accuracy_std'] * 100:.2f}%",
                "Accuracy 95% CI": bootstrap_cis["accuracy"]["ci_str"],
                "Permutation p-value": perm_result["p_value"],
                "Permutation Runtime (s)": perm_result["runtime_seconds"],
                "Best Hyperparameters": str(best_params)
            }
            comparison_records.append(comp_record)

            nested_cv_results[mname] = {
                "nested_cv_summary": cv_summary,
                "bootstrap_95_ci": bootstrap_cis,
                "permutation_test": perm_result,
                "best_hyperparameters": best_params
            }

        # Apply 1-Standard-Error Selection Rule
        selected_model_name, selection_diagnostics = select_model_1se_rule(
            comparison_records=comparison_records,
            primary_metric="ROC-AUC Mean",
            se_metric="Nadeau-Bengio SE",
            baseline_model="LogisticRegression"
        )

        for rec in comparison_records:
            rec["1-SE Selected"] = (rec["Model"] == selected_model_name)

        # Save comparison_results.csv
        comparison_df = pd.DataFrame(comparison_records)
        comp_csv_path = self.models_dir / "comparison_results.csv"
        comparison_df.to_csv(comp_csv_path, index=False)
        logger.info(f"Saved comparison results CSV to {comp_csv_path}")

        # Save fold metrics CSV
        fold_df = pd.DataFrame(all_fold_records)
        fold_csv_path = self.models_dir / "cv_fold_metrics.csv"
        fold_df.to_csv(fold_csv_path, index=False)
        logger.info(f"Saved 50-fold metrics CSV to {fold_csv_path}")

        # Save cv_results.json
        cv_payload = {
            "evaluation_protocol": "Nested RepeatedStratifiedKFold (5 splits x 10 repeats = 50 outer folds)",
            "selection_rule": "1-Standard-Error Rule on ROC-AUC with Nadeau-Bengio corrected SE (Baseline: LogisticRegression)",
            "selected_model": selected_model_name,
            "selection_diagnostics": selection_diagnostics,
            "models": nested_cv_results
        }
        cv_json_path = self.models_dir / "cv_results.json"
        with open(cv_json_path, "w", encoding="utf-8") as f:
            json.dump(cv_payload, f, indent=4)
        logger.info(f"Saved full nested CV results to {cv_json_path}")

        # Train final candidate model on ALL observations (N=79)
        logger.info(f"Fitting selected model '{selected_model_name}' on all {N} observations...")
        selected_pipe_factory = self.build_pipeline_factory(selected_model_name, target_col, feature_names)
        final_pipeline = selected_pipe_factory()
        final_pipeline.set_params(**best_hyperparams_per_model[selected_model_name])
        final_pipeline.fit(X, y)

        # Separate fitted preprocessor and classifier
        fitted_preprocessor = final_pipeline.named_steps['preprocessor']
        fitted_classifier = final_pipeline.named_steps['classifier']

        # Save candidate model as candidate_risk_model.pkl (do NOT overwrite legacy kidney_risk_model.pkl)
        candidate_model_path = self.models_dir / "candidate_risk_model.pkl"
        joblib.dump(fitted_classifier, candidate_model_path)
        logger.info(f"Saved selected candidate classifier to {candidate_model_path}")

        # Save preprocessor artifacts
        pipe_path = self.artifacts_dir / "preprocessing_pipeline.pkl"
        joblib.dump(fitted_preprocessor, pipe_path)
        logger.info(f"Saved fitted preprocessing ColumnTransformer to {pipe_path}")

        if 'num' in fitted_preprocessor.named_transformers_:
            scaler = fitted_preprocessor.named_transformers_['num'].named_steps.get('scaler')
            if scaler is not None:
                joblib.dump(scaler, self.artifacts_dir / "scaler.pkl")
                logger.info("Saved fitted StandardScaler to scaler.pkl")

        # Save combined pipeline as extra file
        combined_pipe_path = self.models_dir / "candidate_risk_pipeline.pkl"
        joblib.dump(final_pipeline, combined_pipe_path)
        logger.info(f"Saved combined candidate pipeline to {combined_pipe_path}")

        # Generate diagnostic comparison charts
        self.plot_comparison_chart(comparison_df)
        self.save_feature_importance(fitted_classifier, feature_names)

        # Write comprehensive model report
        self.write_model_report(nested_cv_results, comparison_df, selection_diagnostics, feature_names)

        logger.info("Phase 1 Tabular ML Evaluation & Model Selection completed successfully.")
        return cv_payload

    def plot_comparison_chart(self, df: pd.DataFrame) -> None:
        """Plots and saves model comparison bar chart."""
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(9, 5))
        sns.barplot(data=df, x="Model", y="ROC-AUC Mean", palette="Blues_d", hue="Model", legend=False)
        plt.title("Kidney Stone Risk Model Nested CV Comparison (50 Folds ROC-AUC)", fontsize=13, fontweight="bold", pad=15)
        plt.ylim(0, 1.05)
        plt.ylabel("ROC-AUC (Nested CV Mean)")
        plt.tight_layout()
        chart_path = self.charts_dir / "model_comparison.png"
        plt.savefig(chart_path, dpi=300)
        plt.close()

    def save_feature_importance(self, model: Any, feature_names: List[str]) -> None:
        """Computes and saves feature importances for the selected candidate model."""
        importances = []
        if hasattr(model, "feature_importances_"):
            importances = list(model.feature_importances_)
        elif hasattr(model, "coef_"):
            importances = list(np.abs(model.coef_[0]))
        else:
            logger.warning("Selected candidate model does not expose feature_importances_ or coef_")
            return

        feat_imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        feat_path = self.models_dir / "feature_importance.json"
        feat_imp_df.to_json(feat_path, orient="records", indent=4)
        logger.info(f"Saved feature importance JSON to {feat_path}")

        plt.figure(figsize=(8, 5))
        sns.barplot(data=feat_imp_df, x="importance", y="feature", palette="viridis", hue="feature", legend=False)
        plt.title("Candidate Model Feature Importance Ranking", fontweight="bold", fontsize=13, pad=15)
        plt.xlabel("Importance / Absolute Coefficient", fontweight="bold")
        plt.ylabel("Feature", fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.charts_dir / "feature_importance.png", dpi=300)
        plt.close()

    def write_model_report(
        self,
        nested_results: Dict[str, Any],
        comparison_df: pd.DataFrame,
        diagnostics: Dict[str, Any],
        features: List[str]
    ) -> None:
        """Generates comprehensive model_b_report.md with transparent statistical comparison."""
        report_path = self.reports_dir / "model_b_report.md"
        feature_summary = ", ".join(f"`{f}`" for f in features)

        table_rows = []
        for _, row in comparison_df.iterrows():
            table_rows.append(
                f"| **{row['Model']}** | {row['ROC-AUC (Mean ± SD)']} (`{row['ROC-AUC 95% CI']}`) | "
                f"{row['F1 (Mean ± SD)']} (`{row['F1 95% CI']}`) | `{row['MCC 95% CI']}` | "
                f"`{row['Sensitivity 95% CI']}` | `{row['Specificity 95% CI']}` | "
                f"{row['Permutation p-value']:.4f} ({row['Permutation Runtime (s)']}s) | {'✅ Yes' if row['1-SE Selected'] else 'No'} |"
            )
        table_md = "\n".join(table_rows)

        selected_model = diagnostics["selected_model"]
        is_complex_better = diagnostics["is_any_model_significantly_better_than_baseline"]

        significance_summary = (
            "No complex model (RandomForest or XGBoost) demonstrated statistically significant superiority over the "
            "simpler Logistic Regression baseline beyond 1 Standard Error. Therefore, adhering to Occam's razor and the 1-SE "
            "selection rule, **LogisticRegression** is selected as the robust, parsimonious model."
            if not is_complex_better else
            f"**{selected_model}** demonstrated superior performance exceeding the 1-Standard-Error threshold and was selected."
        )

        report_md = f"""# Phase 1 Evaluation Report — Model B: Kidney Stone Risk Prediction

**Dataset:** Clinical Urine Analysis Dataset (79 observations, 6 physiological features)  
**Evaluation Protocol:** Nested Repeated Stratified Cross-Validation (`5 splits x 10 repeats = 50 outer folds`)  
**Hyperparameter Search:** Inner 5-fold Stratified GridSearchCV fit strictly on outer training partitions  
**Statistical Uncertainty:** Row-Level Non-Parametric Bootstrap (2,000 resamples, 95% CI)  
**Significance Testing:** Non-Parametric Permutation Tests (1,000 permutations per model, `n_jobs=-1`)  
**Selection Rule:** 1-Standard-Error Rule with Nadeau-Bengio Corrected Standard Error (Baseline: Logistic Regression)  
**Selected Candidate Model:** `{selected_model}` (saved to `ml/models/candidate_risk_model.pkl`)  

---

## 📌 Critical Protocol & Dataset Assumptions

1. **Independent Patient Assumption**:
   - Each row is treated as an independent patient because no patient identifier exists in the clinical dataset ($N=79$).
   - Bootstrap confidence intervals are computed by first averaging out-of-fold predictions per row across the 10 CV repeats, followed by 2,000 row-level bootstrap resamples.

2. **Final Model Training**:
   - **The final candidate model is trained on all 79 rows and has no held-out test set.**
   - Generalization performance is strictly estimated via the 50 outer folds of the nested cross-validation protocol.

3. **Leakage & Synthetic Data Guardrails**:
   - Feature engineering derives purely elementwise physiological ratios (`cond/osmo`, `urea/calc`) and computes no aggregate dataset statistics prior to splitting.
   - Zero SMOTE or synthetic oversampling is applied; class balance is managed strictly via loss/sample weighting.

---

## 📊 Nested Cross-Validation Performance Comparison (50 Outer Folds)

| Model | ROC-AUC (Mean ± SD [95% CI]) | F1-Score (Mean ± SD [95% CI]) | MCC 95% CI | Sensitivity 95% CI | Specificity 95% CI | Permutation $p$-value (Runtime) | 1-SE Selected |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{table_md}

---

## 🎯 Model Selection & Statistical Significance

- **Selection Metric:** ROC-AUC
- **Best-Performing Model:** `{diagnostics['best_performing_model']}` (Mean ROC-AUC: `{diagnostics['best_model_mean_score']:.4f}`, Nadeau-Bengio SE: `{diagnostics['best_model_se']:.4f}`)
- **1-SE Selection Threshold:** `{diagnostics['selection_threshold_1se']:.4f}`
- **Qualifying Models within 1-SE:** {', '.join(f'`{m}`' for m in diagnostics['qualifying_models_within_1se'])}
- **Selected Candidate:** **`{selected_model}`**

### Honest Findings & Significance Summary:
{significance_summary}

---

## 🧬 Physiological Features Utilized
{feature_summary}
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        logger.info(f"Generated comprehensive Phase 1 evaluation report at {report_path}")


def main():
    base_dir = Path(__file__).resolve().parents[1]
    dataset_path = base_dir / "datasets" / "kidneyData.csv"
    artifacts_dir = base_dir / "artifacts"
    models_dir = base_dir / "models"
    charts_dir = base_dir / "outputs" / "charts"
    reports_dir = base_dir / "outputs" / "reports"

    trainer = MLRiskTrainer(
        dataset_path=dataset_path,
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
        charts_dir=charts_dir,
        reports_dir=reports_dir
    )
    trainer.run_nested_evaluation_and_comparison()


if __name__ == "__main__":
    main()
