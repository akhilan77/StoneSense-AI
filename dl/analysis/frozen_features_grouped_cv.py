"""Nested Stratified Grouped Cross-Validation on Frozen ResNet18 Features.

Extracts 512-dim features from pretrained ResNet18 and performs nested StratifiedGroupKFold
CV with StandardScaler, class_weight='balanced', and inner grid search for regularization C.
"""

import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import recall_score, accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("always", category=ConvergenceWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "dl" / "outputs" / "reports" / "grouped_split_manifest.csv"


def run_frozen_cv():
    manifest_df = pd.read_csv(MANIFEST_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading pretrained ResNet18 backbone on {device}...")

    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone.to(device)
    backbone.eval()

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class ManifestDataset(Dataset):
        def __init__(self, df, transform):
            self.df = df.reset_index(drop=True)
            self.transform = transform
            self.class_to_idx = {'Cyst': 0, 'Normal': 1, 'Stone': 2, 'Tumor': 3}

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            path = row['source_path']
            if not os.path.isabs(path):
                path = str(PROJECT_ROOT / path)
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            label = self.class_to_idx[row['class']]
            group = row['near_duplicate_group_id']
            return img, label, group

    dataset = ManifestDataset(manifest_df, val_tf)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

    features_list, labels_list, groups_list = [], [], []
    print("Extracting 512-dim frozen features for all 12,446 images...")
    with torch.no_grad():
        for imgs, lbls, grps in loader:
            imgs = imgs.to(device)
            feats = backbone(imgs)
            features_list.append(feats.cpu().numpy())
            labels_list.append(lbls.numpy())
            groups_list.extend(grps)

    X = np.vstack(features_list)
    y = np.concatenate(labels_list)
    groups = np.array(groups_list)

    outer_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    c_grid = [0.001, 0.01, 0.1, 1.0, 10.0]

    outer_accs, outer_f1s = [], []
    outer_recalls = {'Cyst': [], 'Normal': [], 'Stone': [], 'Tumor': []}
    best_c_per_fold = []
    convergence_warnings = 0
    class_names = ['Cyst', 'Normal', 'Stone', 'Tumor']

    print("\nRunning Nested StratifiedGroupKFold CV...")
    for fold, (train_idx, val_idx) in enumerate(outer_sgkf.split(X, y, groups=groups)):
        X_train_raw, y_train = X[train_idx], y[train_idx]
        groups_train = groups[train_idx]
        X_val_raw, y_val = X[val_idx], y[val_idx]

        inner_sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        best_c, best_inner_f1 = 1.0, -1.0

        for c_cand in c_grid:
            inner_f1_scores = []
            for inner_train_idx, inner_val_idx in inner_sgkf.split(X_train_raw, y_train, groups=groups_train):
                X_in_tr_raw, y_in_tr = X_train_raw[inner_train_idx], y_train[inner_train_idx]
                X_in_va_raw, y_in_va = X_train_raw[inner_val_idx], y_train[inner_val_idx]

                in_scaler = StandardScaler()
                X_in_tr = in_scaler.fit_transform(X_in_tr_raw)
                X_in_va = in_scaler.transform(X_in_va_raw)

                with warnings.catch_warnings(record=True) as w:
                    clf_in = LogisticRegression(
                        C=c_cand, class_weight='balanced', max_iter=2000, solver='lbfgs', random_state=42
                    )
                    clf_in.fit(X_in_tr, y_in_tr)
                    if len(w) > 0 and any(issubclass(item.category, ConvergenceWarning) for item in w):
                        convergence_warnings += 1

                y_in_pred = clf_in.predict(X_in_va)
                inner_f1_scores.append(f1_score(y_in_va, y_in_pred, average='macro'))

            mean_f1 = np.mean(inner_f1_scores)
            if mean_f1 > best_inner_f1:
                best_inner_f1 = mean_f1
                best_c = c_cand

        best_c_per_fold.append(best_c)

        outer_scaler = StandardScaler()
        X_train = outer_scaler.fit_transform(X_train_raw)
        X_val = outer_scaler.transform(X_val_raw)

        with warnings.catch_warnings(record=True) as w:
            clf_out = LogisticRegression(
                C=best_c, class_weight='balanced', max_iter=2000, solver='lbfgs', random_state=42
            )
            clf_out.fit(X_train, y_train)
            if len(w) > 0 and any(issubclass(item.category, ConvergenceWarning) for item in w):
                convergence_warnings += 1

        y_pred = clf_out.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro')
        recs = recall_score(y_val, y_pred, average=None, labels=[0, 1, 2, 3])

        outer_accs.append(acc)
        outer_f1s.append(f1)
        for c_idx, c_name in enumerate(class_names):
            outer_recalls[c_name].append(recs[c_idx])

        print(f"Outer Fold {fold+1}: Best C={best_c} | Acc={acc:.4f}, Macro F1={f1:.4f} | Recalls: Cyst={recs[0]:.4f}, Normal={recs[1]:.4f}, Stone={recs[2]:.4f}, Tumor={recs[3]:.4f}")

    print("\n" + "=" * 80)
    print("NESTED STRATIFIED GROUPED CV SUMMARY")
    print("=" * 80)
    print("Scaling: StandardScaler (train-fold fit only)")
    print("Class Weighting: class_weight='balanced'")
    print(f"Convergence Warnings: {convergence_warnings}")
    print(f"Selected C Grid: {best_c_per_fold}")
    print(f"Accuracy: {np.mean(outer_accs):.4f} +/- {np.std(outer_accs):.4f}")
    print(f"Macro F1: {np.mean(outer_f1s):.4f} +/- {np.std(outer_f1s):.4f}")
    for c_name in class_names:
        r_list = outer_recalls[c_name]
        print(f"Recall {c_name:<7}: {np.mean(r_list):.4f} +/- {np.std(r_list):.4f}")


if __name__ == "__main__":
    run_frozen_cv()
