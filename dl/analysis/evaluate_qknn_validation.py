"""Reproducible validation-only QKNN neighbor/weighting sweep.

The AngleEmbedding(Y) fidelity kernel in the deployed circuit has no entangling
gates, so its |0...0> probability is exactly prod(cos((x - p) / 2) ** 2).
This evaluates that same quantum kernel analytically, without requiring PennyLane.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "dl" / "models" / "ct" / "qknn"
EMBEDDINGS_DIR = PROJECT_ROOT / "dl" / "models" / "ct" / "dinov3" / "embeddings"
CLASS_NAMES = ("Cyst", "Normal", "Stone", "Tumor")
K_CANDIDATES = (3, 5, 7, 9, 11, 15, 20, 30, 40)


def _load_artifacts() -> Tuple[Any, Any, np.ndarray, np.ndarray, int]:
    bundle_path = MODEL_DIR / "qknn_bundle.joblib"
    if bundle_path.exists():
        bundle = joblib.load(bundle_path)
        pca, scaler = bundle["pca"], bundle["scaler"]
        prototypes_x, prototypes_y = bundle["proto_X"], bundle["proto_y"]
        configured_k = int(bundle.get("k", len(prototypes_y)))
    else:
        bundle = joblib.load(MODEL_DIR / "pca_pipeline.pkl")
        pca, scaler = bundle["pca"], bundle["scaler"]
        prototypes = np.load(MODEL_DIR / "prototypes.npz")
        prototypes_x, prototypes_y = prototypes["X"], prototypes["y"]
        model = joblib.load(MODEL_DIR / "qknn_model.pkl")
        configured_k = int(model.get("k", len(prototypes_y))) if isinstance(model, dict) else len(prototypes_y)

    return pca, scaler, np.asarray(prototypes_x), np.asarray(prototypes_y), configured_k


def _transform_embeddings(values: np.ndarray, pca: Any, scaler: Any) -> np.ndarray:
    """Uses the transform order encoded by the fitted artifacts' feature dimensions."""
    values = np.asarray(values, dtype=np.float32)
    if pca is not None and scaler is not None:
        if getattr(scaler, "n_features_in_", None) == values.shape[1]:
            transformed = pca.transform(scaler.transform(values))
        elif getattr(pca, "n_features_in_", None) == values.shape[1]:
            transformed = scaler.transform(pca.transform(values))
        else:
            raise ValueError("Saved QKNN scaler/PCA dimensions do not match the embedding data.")
    elif pca is not None:
        transformed = pca.transform(values)
    elif scaler is not None:
        transformed = scaler.transform(values)
    else:
        transformed = values
    return np.asarray(transformed, dtype=np.float32)


def _predict(similarities: np.ndarray, labels: np.ndarray, k: int, weighted: bool) -> np.ndarray:
    neighbor_indices = np.argpartition(similarities, -k, axis=1)[:, -k:]
    neighbor_weights = (
        np.take_along_axis(similarities, neighbor_indices, axis=1)
        if weighted
        else np.ones(neighbor_indices.shape, dtype=np.float32)
    )
    class_scores = np.zeros((len(similarities), len(CLASS_NAMES)), dtype=np.float64)
    np.add.at(
        class_scores,
        (np.arange(len(similarities))[:, None], labels[neighbor_indices]),
        neighbor_weights,
    )
    return np.argmax(class_scores, axis=1)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(CLASS_NAMES)), average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def main() -> None:
    train = np.load(EMBEDDINGS_DIR / "dinov3_train.npz")
    validation = np.load(EMBEDDINGS_DIR / "dinov3_validation.npz")
    pca, scaler, prototypes_x, prototypes_y, configured_k = _load_artifacts()

    x_validation = _transform_embeddings(validation["X"], pca, scaler)
    similarities = np.prod(
        np.cos((x_validation[:, None, :] - prototypes_x[None, :, :]) / 2) ** 2,
        axis=2,
    )
    y_validation = validation["y"]
    candidates = sorted(
        {min(k, len(prototypes_y)) for k in K_CANDIDATES}
        | {len(prototypes_y), min(configured_k, len(prototypes_y))}
    )
    results = []

    print("QKNN validation-only hyperparameter sweep")
    print(
        f"Train/validation samples: {len(train['y'])}/{len(y_validation)}; "
        f"features/qubits: {x_validation.shape[1]}; prototypes: {len(prototypes_y)}"
    )
    print("Selection uses validation only; test data is loaded after selection for final reporting.")
    for weighted in (True, False):
        weighting = "similarity" if weighted else "uniform"
        for k in candidates:
            prediction = _predict(similarities, prototypes_y, k, weighted)
            metrics = _metrics(y_validation, prediction)
            result = {"k": k, "weighting": weighting, **metrics}
            results.append(result)
            print(
                f"k={k:2d} weighting={weighting:10s} "
                f"accuracy={metrics['accuracy']:.4f} "
                f"balanced_accuracy={metrics['balanced_accuracy']:.4f} "
                f"macro_precision={metrics['precision_macro']:.4f} "
                f"macro_recall={metrics['recall_macro']:.4f} "
                f"macro_f1={metrics['f1_macro']:.4f}"
            )

    baseline_k = min(configured_k, len(prototypes_y))
    baseline = next(
        result for result in results
        if result["k"] == baseline_k and result["weighting"] == "similarity"
    )
    selected = max(results, key=lambda result: (result["f1_macro"], result["accuracy"]))
    print(
        f"Baseline (k={baseline_k}, similarity): "
        f"accuracy={baseline['accuracy']:.4f}, macro_f1={baseline['f1_macro']:.4f}"
    )
    print(
        f"Validation-selected: k={selected['k']}, weighting={selected['weighting']}, "
        f"accuracy={selected['accuracy']:.4f}, macro_f1={selected['f1_macro']:.4f}"
    )

    # Keep test data out of model selection; load it only after choosing the validation setting.
    test = np.load(EMBEDDINGS_DIR / "dinov3_test.npz")
    x_test = _transform_embeddings(test["X"], pca, scaler)
    test_similarities = np.prod(
        np.cos((x_test[:, None, :] - prototypes_x[None, :, :]) / 2) ** 2,
        axis=2,
    )
    test_candidates = (
        ("baseline", baseline_k, True),
        ("validation-selected", selected["k"], selected["weighting"] == "similarity"),
    )
    for name, k, weighted in test_candidates:
        test_prediction = _predict(test_similarities, prototypes_y, k, weighted)
        metrics = _metrics(test["y"], test_prediction)
        print(
            f"Test {name} (k={k}, weighting={'similarity' if weighted else 'uniform'}): "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"macro_precision={metrics['precision_macro']:.4f}, "
            f"macro_recall={metrics['recall_macro']:.4f}, "
            f"macro_f1={metrics['f1_macro']:.4f}"
        )
        print(
            classification_report(
                test["y"],
                test_prediction,
                labels=np.arange(len(CLASS_NAMES)),
                target_names=CLASS_NAMES,
                zero_division=0,
            )
        )
        print(f"Confusion matrix (true rows, predicted columns):\n{confusion_matrix(test['y'], test_prediction)}")


if __name__ == "__main__":
    main()
