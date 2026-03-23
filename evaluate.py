#!/usr/bin/env python3
"""
Model Evaluation & Reporting
Detailed evaluation of the trained IDS model with SOC-relevant metrics.
"""

import pickle
import argparse
import numpy as np
import sys
import os

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve,
    accuracy_score, f1_score
)

sys.path.insert(0, os.path.dirname(__file__))
from ids_model import load_nsl_kdd


def evaluate(model_path: str, test_file: str):
    """Full evaluation report for SOC context."""

    print("=" * 65)
    print("  AI IDS — Model Evaluation Report")
    print("=" * 65)

    # Load model
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    model    = saved["model"]
    engineer = saved["engineer"]

    # Load test data
    X_test, y_test, _ = load_nsl_kdd(test_file)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Core metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred, average="weighted")
    roc_auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else "N/A"

    print(f"\n  Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}" if isinstance(roc_auc, float) else f"  ROC-AUC   : {roc_auc}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Confusion Matrix:")
    print(f"  {'':20} Predicted Normal  Predicted Attack")
    print(f"  Actual Normal    {tn:12d}       {fp:12d}")
    print(f"  Actual Attack    {fn:12d}       {tp:12d}")

    # SOC-relevant metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Detection rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False alarm rate
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0  # Miss rate

    print(f"\n  SOC-Relevant Metrics:")
    print(f"  Detection Rate (TPR)    : {tpr:.4f} ({tpr*100:.2f}%) — attacks correctly flagged")
    print(f"  False Alarm Rate (FPR)  : {fpr:.4f} ({fpr*100:.2f}%) — normal traffic flagged as attack")
    print(f"  Miss Rate (FNR)         : {fnr:.4f} ({fnr*100:.2f}%) — attacks missed (CRITICAL to minimize)")

    print(f"\n  Full Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

    # Feature importance
    if hasattr(model, "feature_importances_") and engineer:
        feature_names = engineer.get_feature_names()
        importances   = model.feature_importances_
        sorted_idx    = np.argsort(importances)[::-1]

        print(f"  Top 10 Features Driving Predictions:")
        print(f"  {'Rank':<5} {'Feature':<30} {'Importance'}")
        print(f"  {'-'*55}")
        for i in range(min(10, len(feature_names))):
            idx = sorted_idx[i]
            bar = "█" * int(importances[idx] * 200)
            print(f"  {i+1:<5} {feature_names[idx]:<30} {importances[idx]:.4f} {bar}")

    print("\n" + "=" * 65)
    print("  SOC INTEGRATION NOTES")
    print("=" * 65)
    print("""
  In a real SOC environment, this model would:
  1. Receive live network flow data (NetFlow, PCAP)
  2. Extract the same 27 features per connection
  3. Output: NORMAL (pass) or ATTACK (create alert)
  4. Tier 1 analyst reviews flagged traffic
  5. Model is retrained periodically on new labeled data

  Key thresholds to tune for your environment:
  - Increase precision → fewer false alarms for analysts
  - Increase recall    → catch more attacks (fewer misses)
  - Use predict_proba threshold tuning instead of default 0.5
    """)


def parse_args():
    parser = argparse.ArgumentParser(description="AI IDS Model Evaluator")
    parser.add_argument("--model", required=True, help="Path to saved model (.pkl)")
    parser.add_argument("--test",  required=True, help="Path to NSL-KDD test file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model, args.test)
