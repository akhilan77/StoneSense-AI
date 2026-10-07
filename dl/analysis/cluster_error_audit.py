"""StoneSense-AI Cluster Distribution & Validation Error Concentration Audit.

Audits cluster size distributions across splits and classes, inspects mixed-label
clusters, and reports error concentrations across patient/near-duplicate clusters.
"""

import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "dl" / "outputs" / "reports" / "grouped_split_manifest.csv"

import sys
sys.path.append(str(PROJECT_ROOT / "dl" / "training"))
sys.path.append(str(PROJECT_ROOT / "dl" / "preprocessing"))
from model import build_resnet18_classifier, CLASS_MAPPING
from transforms import get_val_test_transforms


def run_audit():
    manifest_df = pd.read_csv(MANIFEST_PATH)

    print("=" * 80)
    print("1. CLUSTER-SIZE DISTRIBUTION PER SPLIT AND CLASS")
    print("=" * 80)
    for split in ['train', 'validation', 'test']:
        print(f"\n--- SPLIT: {split.upper()} ---")
        sdf = manifest_df[manifest_df['destination_split'] == split]
        for cls in ['Cyst', 'Normal', 'Stone', 'Tumor']:
            c_df = sdf[sdf['class'] == cls]
            grp_sizes = c_df.groupby('near_duplicate_group_id').size()
            desc = grp_sizes.describe(percentiles=[0.25, 0.5, 0.75])
            print(f"Class: {cls:<7} | Clusters: {len(grp_sizes):<3} | Total Slices: {grp_sizes.sum():<4} | "
                  f"Min: {desc['min']:<3.0f} | 25%: {desc['25%']:<4.1f} | Median (50%): {desc['50%']:<4.1f} | "
                  f"75%: {desc['75%']:<4.1f} | Max: {desc['max']:<4.0f} | Mean: {desc['mean']:<5.2f} | Std: {desc['std']:<5.2f}")

    print("\n" + "=" * 80)
    print("2. VALIDATION ERROR CONCENTRATION ANALYSIS")
    print("=" * 80)
    val_manifest = manifest_df[manifest_df['destination_split'] == 'validation'].set_index('filename')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18_classifier(num_classes=4, freeze_backbone=False)
    ckpt_path = PROJECT_ROOT / "dl" / "models" / "kidney_resnet18.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    val_tf = get_val_test_transforms(image_size=(224, 224))
    hospitals = ["hospital_1", "hospital_2", "hospital_3"]

    class CustomImageDataset(Dataset):
        def __init__(self, file_list, transform):
            self.file_list = file_list
            self.transform = transform
            self.class_to_idx = {'Cyst': 0, 'Normal': 1, 'Stone': 2, 'Tumor': 3}

        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, idx):
            path = self.file_list[idx]
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            cls_name = Path(path).parent.name
            label = self.class_to_idx[cls_name]
            filename = Path(path).name
            return img, label, filename

    all_val_files = []
    for h in hospitals:
        for cls in ['Cyst', 'Normal', 'Stone', 'Tumor']:
            all_val_files.extend(glob.glob(str(PROJECT_ROOT / "dl" / "datasets" / "partitions" / h / "validation" / cls / "*.*")))

    ds = CustomImageDataset(all_val_files, val_tf)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    records = []
    with torch.no_grad():
        for imgs, labels, filenames in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            labels = labels.numpy()
            for fn, y_t, y_p in zip(filenames, labels, preds):
                cid = val_manifest.loc[fn, 'near_duplicate_group_id']
                records.append({
                    'filename': fn,
                    'cluster_id': cid,
                    'y_true': y_t,
                    'y_pred': y_p,
                    'true_class': CLASS_MAPPING[y_t],
                    'pred_class': CLASS_MAPPING[y_p],
                    'is_error': int(y_t != y_p),
                    'is_tumor_to_normal': int(y_t == 3 and y_p == 1),
                    'is_stone_error': int(y_t == 2 and y_p != 2)
                })

    val_eval_df = pd.DataFrame(records)
    total_val_errors = val_eval_df['is_error'].sum()
    total_tumor_to_normal = val_eval_df['is_tumor_to_normal'].sum()
    total_stone_errors = val_eval_df['is_stone_error'].sum()

    print(f"Total Validation Samples: {len(val_eval_df)}")
    print(f"Total Errors: {total_val_errors} / {len(val_eval_df)} ({total_val_errors/len(val_eval_df)*100:.2f}%)")
    print(f"Total Tumor -> Normal Misses: {total_tumor_to_normal}")
    print(f"Total Stone Errors: {total_stone_errors} (out of 207 Stone slices)")

    cluster_errs = val_eval_df.groupby('cluster_id').agg(
        total_slices=('filename', 'count'),
        true_class=('true_class', lambda s: s.iloc[0] if s.nunique() == 1 else s.value_counts().to_dict()),
        total_errors=('is_error', 'sum'),
        tumor_to_normal_errors=('is_tumor_to_normal', 'sum'),
        stone_errors=('is_stone_error', 'sum'),
        error_types=('pred_class', lambda s: dict(s[val_eval_df.loc[s.index, 'is_error'] == 1].value_counts()))
    ).sort_values(by='total_errors', ascending=False)

    top10 = cluster_errs.head(10)
    print("\nTOP 10 CLUSTERS BY ERROR COUNT:")
    print(f"{'Rank':<4} {'Cluster ID':<16} {'True Class':<10} {'Slices':<8} {'Errors':<8} {'Err Rate':<10} {'Tumor->Normal':<15} {'Stone FN':<10} {'Predicted As'}")
    print("-" * 105)
    for rank, (cid, row) in enumerate(top10.iterrows(), 1):
        err_rate = row['total_errors'] / row['total_slices'] * 100
        print(f"{rank:<4} {cid:<16} {str(row['true_class']):<10} {row['total_slices']:<8} {row['total_errors']:<8} {err_rate:<9.1f}% {row['tumor_to_normal_errors']:<15} {row['stone_errors']:<10} {row['error_types']}")


if __name__ == "__main__":
    run_audit()
