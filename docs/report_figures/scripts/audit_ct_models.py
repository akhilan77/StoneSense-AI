import json
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

models = ["resnet18", "yolo26", "dinov3", "qknn"]

for m in models:
    print(f"=== Model: {m} ===")
    metrics_path = f"dl/models/ct/{m}/metrics.json"
    pred_path = f"dl/models/ct/{m}/predictions.csv"
    
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            if m == "resnet18":
                held_out = meta.get("held_out_test_metrics", {})
                acc = held_out.get("slice_accuracy")
                f1 = held_out.get("slice_f1_macro")
                st_rec = held_out.get("slice_per_class", {}).get("Stone", {}).get("recall")
                st_prec = held_out.get("slice_per_class", {}).get("Stone", {}).get("precision")
                print(f"  metrics.json: acc={acc:.4f}, f1={f1:.4f}, stone_recall={st_rec:.4f}, stone_prec={st_prec:.4f}")
            else:
                acc = meta.get("test_accuracy")
                f1 = meta.get("test_f1_macro")
                st_rec = meta.get("stone_recall")
                print(f"  metrics.json: acc={acc}, f1={f1}, stone_recall={st_rec}")
                
    if os.path.exists(pred_path):
        df = pd.read_csv(pred_path)
        print(f"  predictions.csv shape: {df.shape}, cols: {df.columns.tolist()[:8]}")
        # Identify true and pred columns
        y_true_col = "true_label" if "true_label" in df.columns else "true_class" if "true_class" in df.columns else "ground_truth" if "ground_truth" in df.columns else "label"
        y_pred_col = "predicted_label" if "predicted_label" in df.columns else "pred_class" if "pred_class" in df.columns else "prediction" if "prediction" in df.columns else "pred"
        print(f"  Using cols: true={y_true_col}, pred={y_pred_col}")
        y_true = df[y_true_col]
        y_pred = df[y_pred_col]
        
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        classes = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        print(f"  classes: {classes}")
        print(f"  confusion matrix:\n{cm}")
        
        # Stone metrics
        stone_label = "Stone" if "Stone" in classes else 2 if 2 in classes else None
        if stone_label is not None:
            idx = classes.index(stone_label)
            tp = cm[idx, idx]
            fn = cm[idx, :].sum() - tp
            fp = cm[:, idx].sum() - tp
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            print(f"  Computed: acc={acc:.4f}, f1_macro={f1_macro:.4f}, stone_recall={rec:.4f}, stone_prec={prec:.4f}")
