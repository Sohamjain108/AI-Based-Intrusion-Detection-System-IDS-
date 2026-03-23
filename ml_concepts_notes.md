# Machine Learning for Cybersecurity — Study Notes

## Why ML for IDS?

Traditional IDS (Snort, Suricata) use **signatures** — pre-written rules matching known attack patterns.
- ✅ Fast, accurate for known threats
- ❌ Cannot detect zero-day or novel attacks

ML-based IDS uses **behavioral patterns** — learns what "normal" looks like.
- ✅ Can detect novel and unknown attacks
- ❌ Requires training data; produces false positives

---

## Supervised Learning Concepts

### Classification Task

Given a network connection's features → predict: **Normal (0)** or **Attack (1)**

```
Input:  [duration=0, src_bytes=1032, protocol=tcp, service=http, flag=SF, ...]
Output: 0 (Normal) or 1 (Attack)
```

### Training vs Testing

| Phase | Purpose |
|-------|---------|
| Training | Model learns patterns from labeled data |
| Validation | Tune model hyperparameters |
| Testing | Final honest evaluation on unseen data |

---

## Random Forest — How It Works

```
Training Data
      ↓
Build 100 Decision Trees (each on random subset of data + features)
      ↓
New Data → Each tree votes (Normal or Attack)
      ↓
Majority Vote → Final prediction
```

**Why it works well for IDS:**
- Handles high-dimensional data (41 features)
- Resistant to overfitting
- Provides feature importance scores
- Fast prediction (critical for real-time IDS)

### Decision Tree — Single Tree

```
             src_bytes > 5000?
            /                  \
          Yes                   No
           |                     |
    duration < 1?           logged_in == 1?
    /           \            /          \
 ATTACK        NORMAL    ATTACK        NORMAL
```

---

## Key Evaluation Metrics for IDS

### Why accuracy alone is not enough

If 99% of traffic is normal and you predict "normal" always → 99% accuracy but 0% attack detection!

### Better metrics for IDS:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **True Positive Rate (TPR)** | TP / (TP+FN) | % of attacks correctly detected |
| **False Positive Rate (FPR)** | FP / (FP+TN) | % of normal traffic falsely flagged |
| **False Negative Rate (FNR)** | FN / (TP+FN) | % of attacks MISSED (minimize this!) |
| **Precision** | TP / (TP+FP) | When model says "attack", how often correct? |
| **F1 Score** | 2 × P×R/(P+R) | Harmonic mean of precision & recall |
| **ROC-AUC** | Area under ROC curve | Overall discriminative ability |

### The IDS Trade-off

```
High Precision → Fewer false alarms → Analysts not overwhelmed
High Recall    → Fewer missed attacks → Better security coverage

You cannot maximize both simultaneously (trade-off).
Tune the decision threshold (default 0.5) based on risk tolerance.
```

---

## Feature Engineering Concepts

### Why features matter

Raw network packets → 41 numerical features → Model input

Key feature categories:
1. **Volume features** (src_bytes, dst_bytes) → detect DoS floods
2. **Time-window features** (count, srv_count) → detect rapid scanning
3. **Error rate features** (serror_rate, rerror_rate) → detect probes
4. **Content features** (num_failed_logins, root_shell) → detect R2L/U2R

### Label Encoding

Categorical → Numerical (required for sklearn)
```
protocol_type: tcp=2, udp=1, icmp=0
service:       http=10, ftp=5, ssh=18, ...
flag:          SF=10, S0=5, REJ=7, ...
```

### Standard Scaling

Normalizes features to mean=0, std=1
- Prevents features with large values (src_bytes) from dominating
- Required for distance-based algorithms (not strictly for tree models, but good practice)

---

## Attack Types in NSL-KDD

| Category | What it looks like in traffic |
|----------|------------------------------|
| **DoS** (Denial of Service) | Huge src_bytes, high count, serror_rate spike |
| **Probe** (Port Scan) | Short duration, high diff_srv_rate, low src_bytes |
| **R2L** (Remote to Local) | Failed logins, unusual service, small src_bytes |
| **U2R** (User to Root) | num_root > 0, root_shell = 1, su_attempted = 1 |

---

## SOC Connection

| ML Concept | SOC Application |
|-----------|----------------|
| Model prediction | SIEM alert severity score |
| Feature importance | Analyst investigation focus |
| False positive rate | Alert fatigue metric |
| Model retraining | Threat intelligence updates |
| Anomaly score | Behavioral risk score (UEBA) |

---

## Further Learning

- NSL-KDD paper: Tavallaee et al. (2009) — "A Detailed Analysis of the KDD Cup 99 Data Set"
- CICIDS 2018 dataset (newer alternative to NSL-KDD)
- MITRE ATT&CK + ML: MITRE ATLAS (Adversarial Threat Landscape for AI Systems)
- DeepLog: LSTM-based anomaly detection on log sequences
