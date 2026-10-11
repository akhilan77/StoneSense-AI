import os
import json
import sqlite3
import pandas as pd

def audit_summary():
    print("=================== PHASE 1 AUDIT DETAILED ===================")
    
    # 1. Check tabular ML files
    print("\n--- 1. TABULAR ML FILES ---")
    kidney_data_path = "ml/datasets/kidneyData.csv"
    if os.path.exists(kidney_data_path):
        df = pd.read_csv(kidney_data_path)
        print(f"kidneyData shape: {df.shape}, columns: {list(df.columns)}")
        print("First row values:\n", df.iloc[0].to_dict())
        
    comp_results_path = "ml/models/comparison_results.csv"
    if os.path.exists(comp_results_path):
        df_comp = pd.read_csv(comp_results_path)
        print("\ncomparison_results.csv:\n", df_comp.to_string())

    cv_results_path = "ml/models/cv_results.json"
    if os.path.exists(cv_results_path):
        with open(cv_results_path) as f:
            cv_res = json.load(f)
            print("\ncv_results.json:\n", json.dumps(cv_res, indent=2))

    cv_folds_path = "ml/models/cv_fold_metrics.csv"
    if os.path.exists(cv_folds_path):
        df_folds = pd.read_csv(cv_folds_path)
        print(f"\ncv_fold_metrics shape: {df_folds.shape}, columns: {list(df_folds.columns)}")
        print(df_folds.head(3))

    # 2. Check DL CT files
    print("\n--- 2. DL CT FILES ---")
    split_manifest = "dl/outputs/reports/grouped_split_manifest.csv"
    if os.path.exists(split_manifest):
        df_split = pd.read_csv(split_manifest)
        print(f"grouped_split_manifest shape: {df_split.shape}")
        print("Split distribution:")
        print(df_split['split'].value_counts())
        print("Class by split:")
        print(pd.crosstab(df_split['class'], df_split['split']))

    for m in ['resnet18', 'yolo26', 'dinov3', 'qknn']:
        p_csv = f"dl/models/ct/{m}/predictions.csv"
        p_json = f"dl/models/ct/{m}/metrics.json"
        print(f"\nModel {m}:")
        if os.path.exists(p_json):
            with open(p_json) as f:
                d = json.load(f)
                if m == 'resnet18':
                    print("  ResNet18 held_out_test_metrics:", d.get('held_out_test_metrics', {}).get('slice_accuracy'), d.get('held_out_test_metrics', {}).get('slice_f1_macro'))
                else:
                    print("  metrics:", d)
        if os.path.exists(p_csv):
            df_p = pd.read_csv(p_csv)
            print(f"  predictions shape: {df_p.shape}, columns: {list(df_p.columns)}")

if __name__ == "__main__":
    audit_summary()
