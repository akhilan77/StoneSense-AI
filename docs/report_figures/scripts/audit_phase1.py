import os
import json
import sqlite3
import pandas as pd

def audit():
    print("=================== PHASE 1 AUDIT ===================")
    
    # 1. Check tabular ML files
    print("\n--- 1. TABULAR ML FILES ---")
    kidney_data_path = "ml/datasets/kidneyData.csv"
    print(f"kidneyData.csv exists: {os.path.exists(kidney_data_path)}")
    if os.path.exists(kidney_data_path):
        df = pd.read_csv(kidney_data_path)
        print(f"kidneyData shape: {df.shape}, columns: {list(df.columns)}")
        
    comp_results_path = "ml/models/comparison_results.csv"
    print(f"comparison_results.csv exists: {os.path.exists(comp_results_path)}")
    if os.path.exists(comp_results_path):
        df_comp = pd.read_csv(comp_results_path)
        print("comparison_results.csv content:")
        print(df_comp.to_string())

    cv_results_path = "ml/models/cv_results.json"
    print(f"cv_results.json exists: {os.path.exists(cv_results_path)}")
    if os.path.exists(cv_results_path):
        with open(cv_results_path) as f:
            cv_res = json.load(f)
            print("cv_results.json keys:", list(cv_res.keys()))

    cv_folds_path = "ml/models/cv_fold_metrics.csv"
    print(f"cv_fold_metrics.csv exists: {os.path.exists(cv_folds_path)}")
    if os.path.exists(cv_folds_path):
        df_folds = pd.read_csv(cv_folds_path)
        print(f"cv_fold_metrics shape: {df_folds.shape}, columns: {list(df_folds.columns)}")

    ml_roc_path = "ml/outputs/charts/roc_curve.png"
    print(f"ml roc_curve.png exists: {os.path.exists(ml_roc_path)}")
    
    # 2. Check DL CT files
    print("\n--- 2. DL CT FILES ---")
    split_manifest = "dl/outputs/reports/grouped_split_manifest.csv"
    print(f"grouped_split_manifest.csv exists: {os.path.exists(split_manifest)}")
    if os.path.exists(split_manifest):
        df_split = pd.read_csv(split_manifest)
        print(f"grouped_split_manifest shape: {df_split.shape}, columns: {list(df_split.columns)}")
        print("split counts:\n", df_split['split'].value_counts() if 'split' in df_split.columns else "no split col")
        if 'cluster_id' in df_split.columns:
            print("distinct clusters:", df_split['cluster_id'].nunique())

    ct_train_dir = "dl/processed_grouped/train"
    print(f"dl/processed_grouped/train exists: {os.path.exists(ct_train_dir)}")
    if os.path.exists(ct_train_dir):
        for cls in os.listdir(ct_train_dir):
            p = os.path.join(ct_train_dir, cls)
            if os.path.isdir(p):
                print(f"  {cls}: {len(os.listdir(p))} images")

    ct_test_dir = "dl/processed_grouped/test"
    print(f"dl/processed_grouped/test exists: {os.path.exists(ct_test_dir)}")
    if os.path.exists(ct_test_dir):
        for cls in os.listdir(ct_test_dir):
            p = os.path.join(ct_test_dir, cls)
            if os.path.isdir(p):
                print(f"  {cls}: {len(os.listdir(p))} images")

    # Audit report
    audit_rep = "dl/outputs/reports/resnet18_audit_report.md"
    print(f"resnet18_audit_report.md exists: {os.path.exists(audit_rep)}")

    # Check dl roc curve
    dl_roc_path = "dl/outputs/charts/roc_curve.png"
    print(f"dl roc_curve.png exists: {os.path.exists(dl_roc_path)}")

    # Check CT model predictions
    for m in ['resnet18', 'yolo26', 'dinov3', 'qknn']:
        p_csv = f"dl/models/ct/{m}/predictions.csv"
        p_json = f"dl/models/ct/{m}/metrics.json"
        print(f"Model {m}: predictions.csv exists={os.path.exists(p_csv)}, metrics.json exists={os.path.exists(p_json)}")
        if os.path.exists(p_csv):
            df_p = pd.read_csv(p_csv)
            print(f"  {m} predictions shape: {df_p.shape}, columns: {list(df_p.columns)}")

    # Check Grad-CAM
    gradcam_dir = "dl/outputs/gradcam"
    print(f"Grad-CAM dir exists: {os.path.exists(gradcam_dir)}")
    if os.path.exists(gradcam_dir):
        for cls in os.listdir(gradcam_dir):
            p = os.path.join(gradcam_dir, cls)
            if os.path.isdir(p):
                print(f"  GradCAM {cls}: {len(os.listdir(p))} images")

    # Check SHAP
    shap_summary = "ml/outputs/shap/summary_plot.png"
    print(f"SHAP summary plot exists: {os.path.exists(shap_summary)}")
    shap_dir = "ml/outputs/shap"
    if os.path.exists(shap_dir):
        print("SHAP outputs:", os.listdir(shap_dir))

    # 3. Check Federated DB
    print("\n--- 3. FEDERATED DB & COORDINATOR ---")
    for db_path in ["stonesense.db", "backend/stonesense.db"]:
        if os.path.exists(db_path):
            print(f"Inspecting {db_path}...")
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [r[0] for r in cur.fetchall()]
            print("  Tables:", tables)
            if 'federated_rounds' in tables:
                cur.execute("SELECT * FROM federated_rounds;")
                rows = cur.fetchall()
                col_names = [d[0] for d in cur.description]
                print(f"  Columns: {col_names}")
                for row in rows:
                    print(f"  Round: {dict(zip(col_names, row))}")
            conn.close()

if __name__ == "__main__":
    audit()
