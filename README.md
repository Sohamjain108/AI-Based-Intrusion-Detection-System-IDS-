# 🤖 Project 4: AI-Based Intrusion Detection System (IDS)

## Overview

A conceptual and implemented **AI-powered Intrusion Detection System** using Python and scikit-learn to classify network traffic as **normal** or **malicious**. Trained on the **NSL-KDD dataset** — the standard benchmark for network intrusion detection research.

---

## 🎯 Learning Objectives

- Understand supervised machine learning for cybersecurity
- Implement Random Forest and Decision Tree classifiers
- Perform feature engineering on network traffic data
- Evaluate model accuracy and interpret results
- Connect ML output to SOC alert triage workflows

---

## 🧠 How It Works

```
Network Traffic (NSL-KDD Dataset)
          ↓
  Feature Engineering
  (packet size, protocol, duration, flags...)
          ↓
  Train ML Model (Random Forest / Decision Tree)
          ↓
  Classify: NORMAL or ATTACK
          ↓
  Output Alert to SOC Analyst
```

---

## 📊 NSL-KDD Dataset

The **NSL-KDD** dataset is the improved version of the classic KDD Cup 1999 dataset.

### Download Instructions

```bash
# Create data directory
mkdir -p data/raw

# Download NSL-KDD dataset
wget -P data/raw/ https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt
wget -P data/raw/ https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt

# Alternatively, download from:
# https://www.unb.ca/cic/datasets/nsl.html
```

### Dataset Features (41 features)

| Category | Features |
|----------|---------|
| Basic | duration, protocol_type, service, flag, src_bytes, dst_bytes |
| Content | land, wrong_fragment, urgent, hot, num_failed_logins |
| Time-based | count, srv_count, serror_rate, rerror_rate |
| Host-based | dst_host_count, dst_host_srv_count, dst_host_same_src_port_rate |

### Attack Categories

| Category | Examples |
|----------|---------|
| DoS | neptune, smurf, pod, teardrop, land, back |
| Probe | satan, ipsweep, nmap, portsweep |
| R2L | ftp_write, guess_passwd, imap, phf, spy, warezclient |
| U2R | buffer_overflow, loadmodule, perl, rootkit |
| Normal | normal (benign traffic) |

---

## ⚙️ Installation & Setup (Kali Linux)

```bash
cd 4-ai-ids

# Install dependencies
pip3 install -r requirements.txt

# Download dataset (see above)

# Train the model
python3 src/ids_model.py --train

# Evaluate model
python3 src/ids_model.py --evaluate

# Predict on new data
python3 src/ids_model.py --predict data/raw/KDDTest+.txt
```

---

## 🚀 Usage

```bash
# Full pipeline: train + evaluate + save model
python3 src/ids_model.py --train --evaluate --save models/ids_rf_model.pkl

# Quick demo with built-in sample data (no dataset download needed)
python3 src/ids_model.py --demo

# Feature importance analysis
python3 src/feature_engineering.py --importance

# Run model evaluation report
python3 src/evaluate.py --model models/ids_rf_model.pkl --test data/raw/KDDTest+.txt
```

---

## 📈 Expected Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | ~99.2% | ~99.1% | ~99.3% | ~99.2% |
| Decision Tree | ~98.7% | ~98.5% | ~98.9% | ~98.7% |

*Results may vary based on train/test split and preprocessing.*

---

## 📁 Project Files

```
4-ai-ids/
├── src/
│   ├── ids_model.py            # Main model training & prediction
│   ├── feature_engineering.py  # Feature selection & preprocessing
│   └── evaluate.py             # Model evaluation & reporting
├── data/
│   └── README.md               # Dataset download instructions
├── models/
│   └── .gitkeep                # Trained model saved here
├── notebooks/
│   └── ids_exploration.ipynb   # Jupyter notebook walkthrough
├── docs/
│   └── ml_concepts_notes.md    # ML + security concept notes
├── requirements.txt
└── README.md
```

---

## 🔗 SOC Relevance

This project directly maps to real SOC tools:
- **UEBA** (User and Entity Behavior Analytics) uses similar ML classification
- **IDS/IPS** like Snort/Suricata use signature + anomaly detection
- **SOAR** platforms use ML to auto-triage alerts
- Random Forest feature importance → explains WHICH network features triggered an alert

---

## ⚠️ Note

This is a learning/portfolio project using a benchmark dataset.
Real production IDS systems require continuous retraining on live traffic,
feature drift monitoring, and human-in-the-loop validation.
