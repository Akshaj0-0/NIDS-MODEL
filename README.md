# AI-Based Network Intrusion Detection System

## Setup

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Place your dataset files in the same folder as nids_model.py
```
UNSW_NB15_training-set.csv
UNSW_NB15_testing-set.csv
KDDTrain+.csv          ← optional but recommended
KDDTest+.csv           ← optional but recommended
```

### 3. Run
```
python nids_model.py
```

All output files (charts, report, anomaly CSV) are saved in the `outputs/` folder.

---

## Dataset notes

### UNSW-NB15
Download from: https://research.unsw.edu.au/projects/unsw-nb15-dataset
Files needed: UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv

### KDD Cup 99 / NSL-KDD
Download NSL-KDD (improved version) from: https://www.unb.ca/cic/datasets/nsl.html
Files needed: KDDTrain+.txt and KDDTest+.txt
Rename them to KDDTrain+.csv and KDDTest+.csv before running.

### Configuration
At the top of nids_model.py you can set:
  USE_KDD = True   → use both UNSW + KDD (recommended)
  USE_KDD = False  → use only UNSW

---

## What the model does

- Loads and harmonises UNSW-NB15 and KDD datasets
- Fixes class imbalance with SMOTE
- Trains a stacking ensemble (Random Forest + Gradient Boosting + SVM)
- Explains each alert with SHAP values
- Detects unseen anomalies with Isolation Forest
- Finds port scanners with DBSCAN clustering
- Detects model staleness with KS drift test
- Classifies 9 individual attack categories (not just normal/attack)

## Output files

| File | Description |
|------|-------------|
| outputs/report.txt | Full text report with all metrics |
| outputs/confusion_matrix.png | Binary classification heatmap |
| outputs/feature_importance.png | Top 15 features bar chart |
| outputs/shap_summary_bar.png | SHAP global feature importance |
| outputs/shap_beeswarm.png | SHAP direction + magnitude plot |
| outputs/multiclass_confusion.png | Per-attack-type confusion matrix |
| outputs/traffic_distribution.png | Normal vs attack sample counts |
| outputs/top_anomalies.csv | Top anomalies found by Isolation Forest |
| outputs/port_scan_suspects.csv | Hosts flagged by DBSCAN |
