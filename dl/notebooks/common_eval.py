"""StoneSense-AI Common Evaluation & Artifacts Contract Module.

Provides standard evaluation metrics (slice & cluster-level, per-class recall/precision/F1,
confusion matrix, 1000-iteration cluster bootstrap 95% CIs, pooled val+test) and manages
the exact output artifacts contract for all CT model families (ResNet18, DINOv3, YOLO26, QKNN).
"""

from pathlib import Path
import json
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

CLASS_NAMES: List[str] = ["Cyst", "Normal", "Stone", "Tumor"]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {i: c for i, c in enumerate(CLASS_NAMES)}


def compute_sha256(filepath: Union[str, Path]) -> str:
    """Computes SHA-256 hash of a file for integrity tracking."""
    filepath = Path(filepath)
    if not filepath.exists() or not filepath.is_file():
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_slice_and_cluster_metrics(
    df_preds: pd.DataFrame,
    class_names: List[str] = CLASS_NAMES,
) -> Dict[str, Any]:
    """Computes comprehensive slice and cluster-level performance metrics.

    Args:
        df_preds: DataFrame containing ['filename', 'cluster_id', 'y_true', 'y_pred',
                  'p_Cyst', 'p_Normal', 'p_Stone', 'p_Tumor']
        class_names: Ordered list of class labels.

    Returns:
        Dictionary of slice metrics, cluster metrics, per-class scores, and confusion matrix.
    """
    y_true = df_preds["y_true"].values
    y_pred = df_preds["y_pred"].values

    # Slice-level metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=class_names, average="weighted", zero_division=0))

    # Per-class metrics
    per_class_prec = precision_score(y_true, y_pred, labels=class_names, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, labels=class_names, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=class_names, average=None, zero_division=0)

    per_class_metrics = {}
    for i, cname in enumerate(class_names):
        per_class_metrics[cname] = {
            "precision": float(per_class_prec[i]),
            "recall": float(per_class_rec[i]),
            "f1_score": float(per_class_f1[i]),
            "support": int(np.sum(y_true == cname))
        }

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_norm = confusion_matrix(y_true, y_pred, labels=class_names, normalize="true")

    # Cluster-level aggregation (average predicted probabilities per cluster)
    prob_cols = [f"p_{c}" for c in class_names]
    cluster_agg = df_preds.groupby("cluster_id").agg({
        "y_true": "first",
        **{col: "mean" for col in prob_cols}
    }).reset_index()

    cluster_prob_matrix = cluster_agg[prob_cols].values
    cluster_pred_indices = np.argmax(cluster_prob_matrix, axis=1)
    cluster_y_pred = [class_names[idx] for idx in cluster_pred_indices]
    cluster_y_true = cluster_agg["y_true"].values

    cluster_acc = float(accuracy_score(cluster_y_true, cluster_y_pred))
    cluster_macro_f1 = float(f1_score(cluster_y_true, cluster_y_pred, labels=class_names, average="macro", zero_division=0))
    cluster_macro_rec = float(recall_score(cluster_y_true, cluster_y_pred, labels=class_names, average="macro", zero_division=0))

    return {
        "n_samples": int(len(df_preds)),
        "n_clusters": int(df_preds["cluster_id"].nunique()),
        "slice_accuracy": acc,
        "slice_macro_precision": macro_prec,
        "slice_macro_recall": macro_rec,
        "slice_macro_f1": macro_f1,
        "slice_weighted_f1": weighted_f1,
        "cluster_accuracy": cluster_acc,
        "cluster_macro_recall": cluster_macro_rec,
        "cluster_macro_f1": cluster_macro_f1,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "class_names": class_names
    }


def compute_cluster_bootstrap_ci(
    df_preds: pd.DataFrame,
    class_names: List[str] = CLASS_NAMES,
    n_iterations: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Calculates non-parametric 95% confidence intervals by resampling whole clusters.

    Cluster-bootstrap prevents optimistic variance estimation on grouped/near-duplicate data.
    """
    rng = np.random.RandomState(seed)
    unique_clusters = df_preds["cluster_id"].unique()
    n_clusters = len(unique_clusters)

    # Pre-group index arrays for fast resampling
    cluster_to_indices = {
        cid: group.index.to_numpy() for cid, group in df_preds.groupby("cluster_id")
    }

    metrics_list = {
        "slice_accuracy": [],
        "slice_macro_f1": [],
        "slice_macro_recall": [],
        "cluster_accuracy": [],
        "cluster_macro_f1": [],
    }
    for cname in class_names:
        metrics_list[f"{cname}_recall"] = []
        metrics_list[f"{cname}_f1"] = []

    prob_cols = [f"p_{c}" for c in class_names]

    for _ in range(n_iterations):
        resampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sample_indices = np.concatenate([cluster_to_indices[cid] for cid in resampled_clusters])
        sample_df = df_preds.iloc[sample_indices]

        y_true = sample_df["y_true"].values
        y_pred = sample_df["y_pred"].values

        metrics_list["slice_accuracy"].append(accuracy_score(y_true, y_pred))
        metrics_list["slice_macro_f1"].append(f1_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0))
        metrics_list["slice_macro_recall"].append(recall_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0))

        # Per class
        recs = recall_score(y_true, y_pred, labels=class_names, average=None, zero_division=0)
        f1s = f1_score(y_true, y_pred, labels=class_names, average=None, zero_division=0)
        for i, cname in enumerate(class_names):
            metrics_list[f"{cname}_recall"].append(recs[i])
            metrics_list[f"{cname}_f1"].append(f1s[i])

        # Cluster level
        c_agg = sample_df.groupby("cluster_id").agg({
            "y_true": "first",
            **{col: "mean" for col in prob_cols}
        })
        c_prob = c_agg[prob_cols].values
        c_pred = [class_names[idx] for idx in np.argmax(c_prob, axis=1)]
        c_true = c_agg["y_true"].values

        metrics_list["cluster_accuracy"].append(accuracy_score(c_true, c_pred))
        metrics_list["cluster_macro_f1"].append(f1_score(c_true, c_pred, labels=class_names, average="macro", zero_division=0))

    ci_results = {}
    for metric_name, values in metrics_list.items():
        ci_results[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "ci_lower_95": float(np.percentile(values, 2.5)),
            "ci_upper_95": float(np.percentile(values, 97.5)),
        }

    return ci_results


def generate_full_evaluation(
    df_val_preds: pd.DataFrame,
    df_test_preds: pd.DataFrame,
    class_names: List[str] = CLASS_NAMES,
    bootstrap_iterations: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generates complete validation, test, and pooled evaluation metrics with CIs.

    Note: When pooled (val + test) metrics are computed, the test set is no longer untouched.
    """
    df_val_preds = df_val_preds.copy()
    df_test_preds = df_test_preds.copy()
    df_val_preds["split"] = "validation"
    df_test_preds["split"] = "test"

    df_pooled = pd.concat([df_val_preds, df_test_preds], ignore_index=True)

    val_metrics = compute_slice_and_cluster_metrics(df_val_preds, class_names)
    val_ci = compute_cluster_bootstrap_ci(df_val_preds, class_names, n_iterations=bootstrap_iterations, seed=seed)

    test_metrics = compute_slice_and_cluster_metrics(df_test_preds, class_names)
    test_ci = compute_cluster_bootstrap_ci(df_test_preds, class_names, n_iterations=bootstrap_iterations, seed=seed)

    pooled_metrics = compute_slice_and_cluster_metrics(df_pooled, class_names)
    pooled_ci = compute_cluster_bootstrap_ci(df_pooled, class_names, n_iterations=bootstrap_iterations, seed=seed)

    return {
        "validation": {**val_metrics, "confidence_intervals_95": val_ci},
        "test": {**test_metrics, "confidence_intervals_95": test_ci},
        "pooled_val_test": {
            **pooled_metrics,
            "confidence_intervals_95": pooled_ci,
            "evaluation_note": "Pooled across validation and test sets; test set is no longer untouched in pooled score."
        },
        "all_predictions": df_pooled
    }


def save_contract_artifacts(
    out_dir: Union[str, Path],
    family: str,
    version_tag: str,
    display_name: str,
    base_model: str,
    license_name: str,
    hyperparameters: Dict[str, Any],
    preprocessing_spec: Dict[str, Any],
    library_versions: Dict[str, str],
    training_time_sec: float,
    evaluation_results: Dict[str, Any],
    additional_artifact_paths: Optional[List[Union[str, Path]]] = None,
    predict_example_code: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Path]:
    """Serializes all artifacts to adhere strictly to the Phase 8 output contract."""
    family_dir = Path(out_dir) / family
    family_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save predictions.csv
    pred_path = family_dir / "predictions.csv"
    cols = ["filename", "cluster_id", "split", "y_true", "y_pred", "p_Cyst", "p_Normal", "p_Stone", "p_Tumor"]
    df_all: pd.DataFrame = evaluation_results["all_predictions"]
    df_all[cols].to_csv(pred_path, index=False)

    # 2. Save metrics.json
    metrics_path = family_dir / "metrics.json"
    metrics_data = {
        "family": family,
        "version_tag": version_tag,
        "validation": evaluation_results["validation"],
        "test": evaluation_results["test"],
        "pooled_val_test": evaluation_results["pooled_val_test"]
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # 3. Save predict_example.py
    example_path = family_dir / "predict_example.py"
    if predict_example_code:
        with open(example_path, "w", encoding="utf-8") as f:
            f.write(predict_example_code)

    # 4. Compute SHA-256 hashes of all artifacts in family folder
    all_files_to_hash = [pred_path, metrics_path]
    if example_path.exists():
        all_files_to_hash.append(example_path)
    if additional_artifact_paths:
        for p in additional_artifact_paths:
            all_files_to_hash.append(Path(p))

    artifact_hashes = {}
    for p in all_files_to_hash:
        if p.exists() and p.is_file():
            artifact_hashes[p.name] = compute_sha256(p)

    # 5. Save model_card.json
    model_card = {
        "family": family,
        "version_tag": version_tag,
        "display_name": display_name,
        "base_model": base_model,
        "license": license_name,
        "trained_on": "dl/processed_grouped",
        "seed": seed,
        "class_names": CLASS_NAMES,
        "input_size": preprocessing_spec.get("input_size", [224, 224]),
        "preprocessing": preprocessing_spec,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "library_versions": library_versions,
        "training_time_sec": round(training_time_sec, 2),
        "hyperparameters": hyperparameters,
        "is_approved": False,
        "artifact_hashes": artifact_hashes,
    }
    card_path = family_dir / "model_card.json"
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    return {
        "model_card": card_path,
        "metrics": metrics_path,
        "predictions": pred_path,
        "predict_example": example_path,
        "family_dir": family_dir
    }
