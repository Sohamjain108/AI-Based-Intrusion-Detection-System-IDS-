#!/usr/bin/env python3
"""
Feature Engineering for AI IDS
Handles encoding, scaling, and feature selection for NSL-KDD data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Categorical columns in NSL-KDD requiring encoding
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# Top features selected by domain knowledge + importance analysis
# These are the most relevant network traffic attributes for anomaly detection
IMPORTANT_FEATURES = [
    "src_bytes",                    # Bytes from source — DoS attacks send massive data
    "dst_bytes",                    # Bytes to destination
    "duration",                     # Connection duration — scans are very short
    "count",                        # Connections to same host in last 2 seconds
    "srv_count",                    # Connections to same service in last 2 seconds
    "dst_host_count",               # Connections to same destination host
    "dst_host_srv_count",           # Connections to same dest+service
    "dst_host_same_srv_rate",       # % of same service connections
    "dst_host_diff_srv_rate",       # % of diff service connections — probe indicator
    "dst_host_same_src_port_rate",  # Port reuse rate
    "serror_rate",                  # % of SYN error connections — SYN flood indicator
    "srv_serror_rate",              # SYN errors for same service
    "rerror_rate",                  # REJ error rate
    "same_srv_rate",                # % of connections to same service
    "diff_srv_rate",                # % of connections to different services
    "num_failed_logins",            # Failed login attempts — brute force
    "logged_in",                    # Was login successful?
    "num_compromised",              # Compromised conditions triggered
    "root_shell",                   # Root shell obtained?
    "su_attempted",                 # su root attempted?
    "num_root",                     # Root accesses
    "num_file_creations",           # File creation operations
    "num_shells",                   # Shell prompts
    "land",                         # Same src/dst IP and port (attack signature)
    "wrong_fragment",               # Malformed packets
    "urgent",                       # Urgent packets
    "hot",                          # "Hot" indicators in content
]


class FeatureEngineer:
    """
    Preprocessing pipeline for NSL-KDD network traffic data.

    Steps:
    1. Encode categorical features (protocol_type, service, flag)
    2. Scale numerical features
    3. Select most important features
    """

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit encoders and scaler, then transform data."""
        df = df.copy()

        # Step 1: Encode categorical columns
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # Step 2: Select features that exist in the dataframe
        available = [f for f in IMPORTANT_FEATURES if f in df.columns]
        remaining = [c for c in df.columns if c not in available]
        self.feature_names = available + remaining

        df_selected = df[self.feature_names].fillna(0)

        # Step 3: Scale
        X = self.scaler.fit_transform(df_selected)
        self.fitted = True
        return X

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted encoders."""
        if not self.fitted:
            raise RuntimeError("Call fit_transform first.")

        df = df.copy()

        for col in CATEGORICAL_COLS:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen categories
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                df[col] = le.transform(df[col])

        available_features = [f for f in self.feature_names if f in df.columns]
        df_selected = df[available_features].fillna(0)
        return self.scaler.transform(df_selected)

    def get_feature_names(self) -> list:
        """Return list of feature names in order."""
        return self.feature_names


def explain_features():
    """Print explanation of key network traffic features for learning."""
    print("\n=== Network Traffic Features for IDS ===\n")

    explanations = {
        "duration":          "Connection length in seconds. Very short = port scan. Very long = data exfil.",
        "protocol_type":     "TCP / UDP / ICMP. ICMP flooding = DoS.",
        "service":           "Network service (HTTP, FTP, SSH, etc.)",
        "flag":              "TCP connection state (SF=normal, S0=no reply, REJ=rejected)",
        "src_bytes":         "Bytes sent from source. High = data upload / DoS flood.",
        "dst_bytes":         "Bytes sent from destination. High = data download.",
        "land":              "1 if source IP = dest IP (specific attack signature)",
        "wrong_fragment":    "Malformed packet fragments — evasion technique",
        "num_failed_logins": "Failed login count — brute force indicator",
        "logged_in":         "1 = logged in successfully, 0 = not logged in",
        "root_shell":        "1 = root shell obtained (privilege escalation!)",
        "serror_rate":       "% of SYN error connections — SYN flood (DoS) indicator",
        "rerror_rate":       "% of REJ (rejected) connections — port scan indicator",
        "same_srv_rate":     "% connections to same service — horizontal scan indicator",
        "count":             "Connections to same host in past 2 seconds — scan speed",
    }

    for feature, explanation in explanations.items():
        print(f"  {feature:<28} {explanation}")


if __name__ == "__main__":
    explain_features()
