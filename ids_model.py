#!/usr/bin/env python3
"""
AI-Based Intrusion Detection System
Author: Your Name
Description: Trains Random Forest and Decision Tree classifiers on the
             NSL-KDD dataset to classify network traffic as normal or malicious.
"""

import argparse
import pickle
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

from feature_engineering import FeatureEngineer


# ─────────────────────────────────────────────
#  NSL-KDD Column Names
# ─────────────────────────────────────────────

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

# Map multi-class labels to binary: normal vs attack
BINARY_MAP = {
    "normal": 0,
    # All attack types → 1
}


# ─────────────────────────────────────────────
#  Data Loading
# ─────────────────────────────────────────────

def load_nsl_kdd(filepath: str) -> tuple:
    """
    Load and preprocess the NSL-KDD dataset.

    Returns:
        X (features), y (labels), feature_names
    """
    print(f"[*] Loading dataset: {filepath}")

    df = pd.read_csv(filepath, header=None, names=NSL_KDD_COLUMNS)
    print(f"[*] Loaded {len(df)} records")

    # Drop difficulty column (not a feature)
    if "difficulty" in df.columns:
        df = df.drop("difficulty", axis=1)

    # Binary classification: normal=0, attack=1
    df["label_binary"] = df["label"].apply(
        lambda x: 0 if x.strip() == "normal" else 1
    )

    print(f"[*] Class distribution:")
    print(f"    Normal: {(df['label_binary'] == 0).sum()}")
    print(f"    Attack: {(df['label_binary'] == 1).sum()}")

    # Encode categorical features
    engineer = FeatureEngineer()
    X = engineer.fit_transform(df.drop(["label", "label_binary"], axis=1))
    y = df["label_binary"].values

    return X, y, engineer


def generate_demo_data(n_samples: int = 1000) -> tuple:
    """
    Generate synthetic demo data when NSL-KDD is not available.
    Useful for testing the pipeline without downloading the dataset.
    """
    print("[*] Generating synthetic demo data (NSL-KDD not found)")
    np.random.seed(42)

    n_features = 38
    X_normal = np.random.randn(n_samples // 2, n_features) * 0.5
    X_attack = np.random.randn(n_samples // 2, n_features) * 2.0 + 3.0

    X = np.vstack([X_normal, X_attack])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Shuffle
    idx = np.random.permutation(len(y))
    return X[idx], y[idx], None


# ─────────────────────────────────────────────
#  Model Training
# ─────────────────────────────────────────────

def train_model(X_train, y_train, model_type: str = "rf") -> object:
    """
    Train a classification model.

    Args:
        X_train: Training features
        y_train: Training labels
        model_type: 'rf' (Random Forest) or 'dt' (Decision Tree)

    Returns:
        Trained model
    """
    print(f"\n[*] Training {'Random Forest' if model_type == 'rf' else 'Decision Tree'}...")

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=100,       # 100 decision trees in the forest
            max_depth=20,           # Max depth per tree
            min_samples_split=5,    # Min samples to split a node
            random_state=42,
            n_jobs=-1,              # Use all CPU cores
            verbose=0
        )
    else:
        model = DecisionTreeClassifier(
            max_depth=20,
            min_samples_split=5,
            random_state=42
        )

    start = datetime.now()
    model.fit(X_train, y_train)
    duration = (datetime.now() - start).total_seconds()

    print(f"[*] Training complete in {duration:.2f}s")
    return model


# ─────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """Print full evaluation metrics."""
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average="weighted")

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1 Score  : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Normal", "Attack"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"             Predicted Normal  Predicted Attack")
    print(f"  Actual Normal    {cm[0][0]:8d}        {cm[0][1]:8d}")
    print(f"  Actual Attack    {cm[1][0]:8d}        {cm[1][1]:8d}")

    # False negative rate (missed attacks) — critical for IDS
    if cm[1][0] + cm[1][1] > 0:
        fnr = cm[1][0] / (cm[1][0] + cm[1][1])
        print(f"\n  False Negative Rate (missed attacks): {fnr:.4f} ({fnr*100:.2f}%)")
        print(f"  [!] In IDS context, a lower FNR is critical")

    print("=" * 60)
    return y_pred


# ─────────────────────────────────────────────
#  Feature Importance
# ─────────────────────────────────────────────

def print_feature_importance(model, feature_names: list, top_n: int = 10):
    """Print top N most important features."""
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print(f"\n  Top {top_n} Most Important Features:")
    print("  " + "-" * 40)
    for i in range(min(top_n, len(feature_names))):
        idx = sorted_idx[i]
        print(f"  {i+1:2}. {feature_names[idx]:<30} {importances[idx]:.4f}")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Based Intrusion Detection System (NSL-KDD)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--train", action="store_true",
                        help="Train the model")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate model on test set")
    parser.add_argument("--demo", action="store_true",
                        help="Run with synthetic demo data (no dataset needed)")
    parser.add_argument("--model-type", choices=["rf", "dt"], default="rf",
                        help="Model: rf=Random Forest, dt=Decision Tree (default: rf)")
    parser.add_argument("--train-file", default="data/raw/KDDTrain+.txt",
                        help="Path to NSL-KDD training file")
    parser.add_argument("--test-file", default="data/raw/KDDTest+.txt",
                        help="Path to NSL-KDD test file")
    parser.add_argument("--save", default="models/ids_model.pkl",
                        help="Save trained model to file")
    parser.add_argument("--predict",
                        help="Path to file to classify")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  AI-Based Intrusion Detection System")
    print("  Model: NSL-KDD | sklearn | Random Forest / Decision Tree")
    print("=" * 60)

    # ── DEMO MODE ──────────────────────────────
    if args.demo:
        X, y, _ = generate_demo_data(2000)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = train_model(X_train, y_train, args.model_type)
        evaluate_model(model, X_test, y_test)
        return

    # ── TRAIN ──────────────────────────────────
    if args.train:
        if not os.path.exists(args.train_file):
            print(f"[!] Training file not found: {args.train_file}")
            print("[!] Run with --demo for synthetic data, or download NSL-KDD:")
            print("    wget https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt -P data/raw/")
            sys.exit(1)

        X_train, y_train, engineer = load_nsl_kdd(args.train_file)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        model = train_model(X_tr, y_tr, args.model_type)
        print("\n[*] Validation set evaluation:")
        evaluate_model(model, X_val, y_val)

        if engineer:
            feature_names = engineer.get_feature_names()
            print_feature_importance(model, feature_names)

        # Save model
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        with open(args.save, "wb") as f:
            pickle.dump({"model": model, "engineer": engineer}, f)
        print(f"\n[*] Model saved to: {args.save}")

    # ── EVALUATE ───────────────────────────────
    if args.evaluate:
        if not os.path.exists(args.save):
            print(f"[!] No saved model found at {args.save}. Run --train first.")
            sys.exit(1)

        with open(args.save, "rb") as f:
            saved = pickle.load(f)

        model    = saved["model"]
        engineer = saved["engineer"]

        X_test, y_test, _ = load_nsl_kdd(args.test_file)
        print("\n[*] Test set evaluation:")
        evaluate_model(model, X_test, y_test)

        if engineer:
            print_feature_importance(model, engineer.get_feature_names())


if __name__ == "__main__":
    main()
