"""
=============================================================
  AI-BASED NETWORK INTRUSION DETECTION SYSTEM (NIDS)
  Upgraded with: SMOTE, Stacking Ensemble, SHAP, DBSCAN
                 Drift Detection, Multi-class Evaluation
                 KDD + UNSW-NB15 dual dataset support

  FAST VERSION — SVM removed, reduced estimators, cv=3
  Expected runtime: 10-15 minutes
=============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from scipy import stats

# --- Preprocessing ---
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# --- Models ---
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    IsolationForest,
)
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN

# --- Metrics ---
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# --- Imbalance fix ---
from imblearn.over_sampling import SMOTE

# --- Explainability ---
import shap

warnings.filterwarnings("ignore")

# =============================================================
# SECTION 1 — CONFIGURATION
# =============================================================

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set to True if you have both datasets; False to use only UNSW
USE_KDD = True

# File paths — update these to match your folder structure
UNSW_TRAIN_PATH = "datasets/UNSW_NB15_training-set.csv"
UNSW_TEST_PATH  = "datasets/UNSW_NB15_testing-set.csv"
KDD_TRAIN_PATH  = "datasets/KDDTrain+.txt"
KDD_TEST_PATH   = "datasets/KDDTest+.txt"


# =============================================================
# SECTION 2 — DATASET LOADING & HARMONISATION
# =============================================================

def load_unsw():
    """Load and combine UNSW-NB15 train/test splits."""
    print("[UNSW] Loading datasets...")
    train = pd.read_csv(UNSW_TRAIN_PATH)
    test  = pd.read_csv(UNSW_TEST_PATH)
    data  = pd.concat([train, test], ignore_index=True)
    print(f"[UNSW] Shape: {data.shape}")

    for col in ["proto", "service", "state"]:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    data["packet_count"] = data["spkts"] + data["dpkts"]
    data["bytes_sent"]   = data["sbytes"]

    data = data.rename(columns={"label": "binary_label"})
    data["dataset_source"] = "UNSW"
    return data


KDD_COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","attack_type","difficulty"
]

KDD_ATTACK_MAP = {
    "normal": 0,
    "back":1,"land":1,"neptune":1,"pod":1,"smurf":1,"teardrop":1,
    "apache2":1,"udpstorm":1,"processtable":1,"worm":1,
    "ipsweep":1,"nmap":1,"portsweep":1,"satan":1,"mscan":1,"saint":1,
    "ftp_write":1,"guess_passwd":1,"imap":1,"multihop":1,"phf":1,
    "spy":1,"warezclient":1,"warezmaster":1,"sendmail":1,"named":1,
    "snmpgetattack":1,"snmpguess":1,"xlock":1,"xsnoop":1,"httptunnel":1,
    "buffer_overflow":1,"loadmodule":1,"perl":1,"rootkit":1,
    "mailbomb":1,"ps":1,"sqlattack":1,"xterm":1,
}

def load_kdd():
    """Load KDDTrain+/KDDTest+ datasets."""
    print("[KDD] Loading datasets...")
    train = pd.read_csv(KDD_TRAIN_PATH, header=None, names=KDD_COLUMNS)
    test  = pd.read_csv(KDD_TEST_PATH,  header=None, names=KDD_COLUMNS)
    data  = pd.concat([train, test], ignore_index=True)
    print(f"[KDD] Shape: {data.shape}")

    data["attack_type"] = data["attack_type"].str.replace(".", "", regex=False).str.strip()
    data["binary_label"] = data["attack_type"].map(KDD_ATTACK_MAP).fillna(1).astype(int)
    data = data.rename(columns={"attack_type": "attack_cat"})

    for col in ["protocol_type", "service", "flag"]:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    data = data.rename(columns={
        "protocol_type": "proto",
        "flag": "state",
        "src_bytes": "sbytes",
        "dst_bytes": "dbytes",
        "duration": "dur",
        "count": "ct_srv_dst",
    })

    data["packet_count"] = data.get("srv_count", 0)
    data["bytes_sent"]   = data["sbytes"]
    data.drop(columns=["difficulty"], inplace=True, errors="ignore")
    data["dataset_source"] = "KDD"
    return data


def harmonise(unsw_df, kdd_df):
    """Keep only columns that exist in both datasets."""
    common = list(set(unsw_df.columns) & set(kdd_df.columns))
    print(f"\n[Harmonise] Common features: {len(common)}")
    return unsw_df[common].copy(), kdd_df[common].copy()


# =============================================================
# SECTION 3 — LOAD & MERGE
# =============================================================

unsw_data = load_unsw()

if USE_KDD and os.path.exists(KDD_TRAIN_PATH) and os.path.exists(KDD_TEST_PATH):
    kdd_data = load_kdd()
    unsw_aligned, kdd_aligned = harmonise(unsw_data, kdd_data)
    data = pd.concat([unsw_aligned, kdd_aligned], ignore_index=True)
    print(f"\n[Combined] Total shape: {data.shape}")
else:
    print("\n[Info] Using UNSW-NB15 only (KDD files not found or USE_KDD=False)")
    data = unsw_data.copy()

print("\n[Data] Column list:")
print(list(data.columns))
print("\n[Data] Binary label distribution:")
print(data["binary_label"].value_counts())


# =============================================================
# SECTION 4 — FEATURE / LABEL SETUP
# =============================================================

DROP_COLS = ["binary_label", "attack_cat", "dataset_source"]
feature_cols = [c for c in data.columns if c not in DROP_COLS]

data[feature_cols] = data[feature_cols].apply(
    lambda col: col.fillna(col.median()) if col.dtype in [np.float64, np.int64] else col.fillna(0)
)

X = data[feature_cols]
y = data["binary_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[Split] Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"[Split] Train label counts: {Counter(y_train)}")
print(f"[Split] Test  label counts: {Counter(y_test)}")


# =============================================================
# SECTION 5 — SCALING
# =============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# =============================================================
# SECTION 6 — SMOTE: CLASS IMBALANCE FIX
# =============================================================

print("\n[SMOTE] Resampling to balance classes...")
print(f"  Before: {Counter(y_train)}")

sm = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

print(f"  After : {Counter(y_train_res)}")


# =============================================================
# SECTION 7 — STACKING ENSEMBLE (FAST — NO SVM)
# =============================================================

print("\n[Model] Building stacking ensemble (RF + GradientBoosting)...")

# SVM removed — too slow for large datasets
# Reduced estimators and cv=3 for speed with minimal accuracy loss
base_models = [
    ("rf", RandomForestClassifier(
               n_estimators=100,       # reduced from 200
               random_state=42,
               class_weight="balanced",
               n_jobs=-1)),
    ("gb", GradientBoostingClassifier(
               n_estimators=50,        # reduced from 100
               random_state=42,
               learning_rate=0.1,
               max_depth=5)),
]

stacked_model = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=3,               # reduced from 5 — cuts time by 40%
    passthrough=False,
    n_jobs=-1,
)

print("[Model] Training stacking ensemble (10-15 minutes)...")
stacked_model.fit(X_train_res, y_train_res)

stacked_preds = stacked_model.predict(X_test_scaled)
stacked_proba = stacked_model.predict_proba(X_test_scaled)

stacked_acc = accuracy_score(y_test, stacked_preds)
print(f"\n[Stacked] Accuracy: {round(stacked_acc * 100, 2)}%")
print("\n[Stacked] Classification Report:")
print(classification_report(y_test, stacked_preds,
                             target_names=["Normal", "Attack"],
                             zero_division=0))


# =============================================================
# SECTION 8 — STANDALONE RANDOM FOREST (for SHAP + importance)
# =============================================================

print("[Model] Training standalone Random Forest for SHAP analysis...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
rf_model.fit(X_train_res, y_train_res)
rf_preds = rf_model.predict(X_test_scaled)
rf_acc   = accuracy_score(y_test, rf_preds)
print(f"[RF] Accuracy: {round(rf_acc * 100, 2)}%")


# =============================================================
# SECTION 9 — CONFUSION MATRIX
# =============================================================

cm = confusion_matrix(y_test, stacked_preds)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Attack"],
            yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Stacked Ensemble — Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()
print("[Plot] Confusion matrix saved.")


# =============================================================
# SECTION 10 — FEATURE IMPORTANCE
# =============================================================

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n[Features] Top 15 most important features:")
print(feature_importance.head(15).to_string())

plt.figure(figsize=(10, 6))
feature_importance.head(15).plot(kind="bar", color="steelblue")
plt.title("Top 15 Features for Intrusion Detection")
plt.ylabel("Importance Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
plt.close()
print("[Plot] Feature importance saved.")


# =============================================================
# SECTION 11 — SHAP EXPLAINABILITY
# =============================================================

print("\n[SHAP] Generating explanations (200 test samples)...")

n_shap = min(200, len(X_test_scaled))
X_shap_df = pd.DataFrame(X_test_scaled[:n_shap], columns=X.columns)

explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_shap_df)

# Handle both old shap (list) and new shap (array) output formats
if isinstance(shap_values, list):
    sv = shap_values[1]   # old format — index 1 = attack class
else:
    sv = shap_values       # new format — already the right array

plt.figure()
shap.summary_plot(sv, X_shap_df, plot_type="bar",
                  feature_names=X.columns.tolist(), show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_bar.png"))
plt.close()

plt.figure()
shap.summary_plot(sv, X_shap_df,
                  feature_names=X.columns.tolist(), show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm.png"))
plt.close()

print("[SHAP] Summary plots saved.")

print(f"\n[SHAP] Top contributing features for test sample #0:")
shap_row = sv[0] if sv[0].ndim == 1 else sv[0][:, 1]
top_contributors = pd.Series(shap_row, index=X.columns).abs().sort_values(ascending=False)
print(top_contributors.head(5).to_string())


# =============================================================
# SECTION 12 — ISOLATION FOREST (UNSUPERVISED ANOMALY DETECTION)
# =============================================================

print("\n[IsoForest] Training unsupervised anomaly detector...")

iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
iso.fit(X_train_res)

anomaly_labels = iso.predict(X_test_scaled)
anomaly_mask   = anomaly_labels == -1

print(f"[IsoForest] Detected {anomaly_mask.sum()} anomalies "
      f"({round(anomaly_mask.mean() * 100, 1)}% of test set)")

anomaly_df = X_test.copy()
anomaly_df["isolation_score"] = iso.score_samples(X_test_scaled)
anomaly_df["is_anomaly"]      = anomaly_mask

anomaly_df[anomaly_df["is_anomaly"]].sort_values(
    "isolation_score"
).head(10).to_csv(os.path.join(OUTPUT_DIR, "top_anomalies.csv"), index=False)
print("[IsoForest] Top anomalies saved.")


# =============================================================
# SECTION 13 — DBSCAN PORT SCAN DETECTION
# =============================================================

print("\n[DBSCAN] Detecting port scanning behaviour...")

scan_feature_candidates = ["ct_srv_dst", "packet_count", "bytes_sent"]
scan_cols = [c for c in scan_feature_candidates if c in data.columns]

if len(scan_cols) >= 2:
    # Sample 10,000 rows max — DBSCAN kills RAM on full dataset
    scan_data = data[scan_cols].fillna(0)
    if len(scan_data) > 10000:
        scan_data = scan_data.sample(n=10000, random_state=42)
        print("[DBSCAN] Using 10,000 sample rows to avoid memory issues.")

    scan_scaled = StandardScaler().fit_transform(scan_data)

    db = DBSCAN(eps=0.8, min_samples=10, n_jobs=-1)
    db_labels = db.fit_predict(scan_scaled)

    port_scan_mask     = db_labels == -1
    port_scan_suspects = scan_data[port_scan_mask]

    print(f"[DBSCAN] Suspected port scanners: {port_scan_mask.sum()} records")
    print(port_scan_suspects[scan_cols].head(5).to_string())

    port_scan_suspects[scan_cols].to_csv(
        os.path.join(OUTPUT_DIR, "port_scan_suspects.csv"), index=False
    )
    print("[DBSCAN] Port scan suspects saved.")
else:
    print("[DBSCAN] Skipped — required scan features not in dataset.")


# =============================================================
# SECTION 14 — CONCEPT DRIFT DETECTION
# =============================================================

print("\n[Drift] Running Kolmogorov-Smirnov drift detection...")

def detect_drift(ref, new, feature_names, threshold=0.05):
    drifted = []
    for i, feat in enumerate(feature_names):
        _, p = stats.ks_2samp(ref[:, i], new[:, i])
        if p < threshold:
            drifted.append((feat, round(p, 6)))
    return sorted(drifted, key=lambda x: x[1])

mid = len(X_test_scaled) // 2
drifted = detect_drift(
    X_test_scaled[:mid],
    X_test_scaled[mid:],
    list(X.columns)
)

print(f"[Drift] Features with distribution shift: {len(drifted)}")
if drifted:
    print("  WARNING — features showing drift (consider retraining):")
    for feat, p in drifted[:10]:
        print(f"    {feat}: p={p}")
else:
    print("  No significant drift detected.")


# =============================================================
# SECTION 15 — MULTI-CLASS ATTACK CATEGORY EVALUATION
# =============================================================

print("\n[Multi-class] Evaluating per-attack-category performance...")

if "attack_cat" in data.columns:
    mc_data = data.copy()
    mc_data["attack_cat"] = mc_data["attack_cat"].fillna("Normal").astype(str).str.strip()

    cat_counts = mc_data["attack_cat"].value_counts()
    valid_cats = cat_counts[cat_counts >= 50].index
    mc_data = mc_data[mc_data["attack_cat"].isin(valid_cats)].copy()

    le_cat = LabelEncoder()
    y_mc   = le_cat.fit_transform(mc_data["attack_cat"])
    mc_features = [c for c in mc_data.columns if c not in DROP_COLS]
    X_mc   = mc_data[mc_features].fillna(0)

    X_tr_mc, X_te_mc, y_tr_mc, y_te_mc = train_test_split(
        X_mc, y_mc, test_size=0.2, random_state=42, stratify=y_mc
    )

    sc_mc = StandardScaler()
    X_tr_mc_s = sc_mc.fit_transform(X_tr_mc)
    X_te_mc_s = sc_mc.transform(X_te_mc)

    min_class = min(Counter(y_tr_mc).values())
    if min_class >= 6:
        sm_mc = SMOTE(random_state=42, k_neighbors=min(5, min_class - 1))
        X_tr_mc_s, y_tr_mc = sm_mc.fit_resample(X_tr_mc_s, y_tr_mc)

    rf_mc = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    rf_mc.fit(X_tr_mc_s, y_tr_mc)
    preds_mc = rf_mc.predict(X_te_mc_s)

    mc_report = classification_report(
        y_te_mc, preds_mc,
        target_names=le_cat.classes_,
        zero_division=0
    )
    print("\n[Multi-class] Per-category classification report:")
    print(mc_report)

    cm_mc = confusion_matrix(y_te_mc, preds_mc)
    plt.figure(figsize=(max(8, len(le_cat.classes_)), max(6, len(le_cat.classes_) - 1)))
    sns.heatmap(cm_mc, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=le_cat.classes_,
                yticklabels=le_cat.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Multi-class Attack Category — Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "multiclass_confusion.png"))
    plt.close()
    print("[Plot] Multi-class confusion matrix saved.")
else:
    mc_report = "attack_cat column not available."
    print("[Multi-class] Skipped — attack_cat column not found.")


# =============================================================
# SECTION 16 — TRAFFIC DISTRIBUTION PLOT
# =============================================================

plt.figure(figsize=(7, 4))
counts = data["binary_label"].value_counts()
bars   = plt.bar(["Normal (0)", "Attack (1)"], counts.values,
                 color=["steelblue", "tomato"])
for bar, val in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + counts.values.max() * 0.01,
             f"{val:,}", ha="center", va="bottom", fontsize=11)
plt.title("Network Traffic Distribution")
plt.ylabel("Sample Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "traffic_distribution.png"))
plt.close()
print("[Plot] Traffic distribution saved.")


# =============================================================
# SECTION 17 — ATTACK PROBABILITY SCORES
# =============================================================

print("\n[Scores] Attack probability for first 5 test samples:")
for i, prob in enumerate(stacked_proba[:5]):
    label = "Attack" if stacked_preds[i] == 1 else "Normal"
    print(f"  Sample {i}: Normal={prob[0]:.3f}  Attack={prob[1]:.3f}  → {label}")


# =============================================================
# SECTION 18 — FULL REPORT
# =============================================================

report_path = os.path.join(OUTPUT_DIR, "report.txt")
with open(report_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("  AI-BASED NETWORK INTRUSION DETECTION SYSTEM REPORT\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Dataset shape         : {data.shape}\n")
    f.write(f"Features used         : {len(feature_cols)}\n")
    f.write(f"Train samples         : {X_train.shape[0]}\n")
    f.write(f"Test samples          : {X_test.shape[0]}\n")
    f.write(f"After SMOTE (train)   : {len(X_train_res)}\n\n")

    f.write("-" * 60 + "\n")
    f.write("BINARY CLASSIFICATION (Normal vs Attack)\n")
    f.write("-" * 60 + "\n")
    f.write(f"Random Forest Accuracy : {round(rf_acc * 100, 2)}%\n")
    f.write(f"Stacked Ensemble Acc   : {round(stacked_acc * 100, 2)}%\n\n")
    f.write("Stacked Ensemble — Classification Report:\n")
    f.write(classification_report(y_test, stacked_preds,
                                   target_names=["Normal", "Attack"],
                                   zero_division=0))

    f.write("\n" + "-" * 60 + "\n")
    f.write("MULTI-CLASS ATTACK CATEGORY REPORT\n")
    f.write("-" * 60 + "\n")
    f.write(mc_report if isinstance(mc_report, str) else mc_report)

    f.write("\n" + "-" * 60 + "\n")
    f.write("DRIFT DETECTION\n")
    f.write("-" * 60 + "\n")
    if drifted:
        f.write(f"Features showing drift: {len(drifted)}\n")
        for feat, p in drifted:
            f.write(f"  {feat}: p={p}\n")
    else:
        f.write("No significant drift detected.\n")

    f.write("\n" + "-" * 60 + "\n")
    f.write("TOP 15 FEATURES BY IMPORTANCE\n")
    f.write("-" * 60 + "\n")
    f.write(feature_importance.head(15).to_string())
    f.write("\n")

print(f"\n[Done] Report written to {report_path}")
print("[Done] All outputs saved in the 'outputs/' folder.")
print("\nOutput files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  outputs/{f}")
