# train_final.py
# Final training with the new 6,426 record dataset

import sqlite3
import json
import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

os.makedirs("models", exist_ok=True)
METADATA_PATH = os.path.join("models", "training_metadata.json")


class EnsembleRiskPredictor:
    def __init__(self, rf_model, iso_model, scaler, feature_names):
        self.rf = rf_model
        self.iso = iso_model
        self.scaler = scaler
        self.feature_names = feature_names

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        rf_proba = self.rf.predict_proba(X_scaled)[:, 1]
        iso_pred = self.iso.predict(X_scaled)
        iso_score = (iso_pred == -1).astype(float)
        return 0.7 * rf_proba + 0.3 * iso_score


def run_training(database_path="academic_portal.db", min_samples=30):
    result = {
        "success": False,
        "message": "",
        "samples": 0,
        "anomaly_ratio": 0.0,
        "rf_f1": None,
        "iso_f1": None,
        "ensemble_f1": None,
        "trained_at": None,
        "features": [],
    }

    if not os.path.isfile(database_path):
        result["message"] = f"Database not found: {database_path}"
        return result

    conn = sqlite3.connect(database_path)
    
    query = """
    SELECT
        l.user_id,
        strftime('%H', l.login_time) AS hour,
        COALESCE(u.role, 'student') AS role,
        CASE
            WHEN l.ip_address = (
                SELECT ip_address FROM login_logs
                WHERE user_id = l.user_id AND login_time < l.login_time
                ORDER BY login_time DESC LIMIT 1
            ) THEN 0 ELSE 1
        END AS ip_changed,
        CASE
            WHEN l.device = (
                SELECT device FROM login_logs
                WHERE user_id = l.user_id AND login_time < l.login_time
                ORDER BY login_time DESC LIMIT 1
            ) THEN 0 ELSE 1
        END AS device_changed,
        CASE
            WHEN l.ip_address LIKE '192.168%' THEN 'College_WiFi'
            WHEN l.ip_address LIKE '202.70%'  THEN 'WorldLink'
            WHEN l.ip_address LIKE '103.1%'   THEN 'Ncell'
            WHEN l.ip_address LIKE '27.34%'   THEN 'NTC'
            ELSE 'Other'
        END AS isp_type,
        CASE
            WHEN l.status = 'blocked' OR l.action = 'block' THEN 1
            WHEN l.action = 'otp_required' THEN 1
            ELSE 0
        END AS is_anomaly
    FROM login_logs l
    LEFT JOIN users u ON l.user_id = u.id
    WHERE l.status != 'otp_sent'
      AND l.user_id IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    result["samples"] = len(df)
    
    if len(df) < min_samples:
        result["message"] = f"Not enough data ({len(df)} records)."
        return result

    # Encode categorical variables
    role_map = {'student': 0, 'teacher': 1, 'admin': 2}
    df['role_enc'] = df['role'].map(role_map).fillna(0)
    
    isp_map = {'College_WiFi': 0, 'WorldLink': 1, 'Ncell': 2, 'NTC': 3, 'Other': 4}
    df['isp_enc'] = df['isp_type'].map(isp_map).fillna(4)

    features = ["hour", "ip_changed", "device_changed", "role_enc", "isp_enc"]

    X = df[features].fillna(0).astype(float)
    y = df["is_anomaly"].astype(int)
    
    anomaly_count = y.sum()
    normal_count = len(y) - anomaly_count
    anomaly_ratio = anomaly_count / len(y)
    
    result["anomaly_ratio"] = round(float(anomaly_ratio), 4)
    result["features"] = features

    print("\n" + "=" * 60)
    print("📊 DATA SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(df)}")
    print(f"Normal: {normal_count} ({normal_count/len(df)*100:.1f}%)")
    print(f"Anomalies: {anomaly_count} ({anomaly_ratio*100:.1f}%)")
    print(f"Imbalance ratio: {normal_count/max(1, anomaly_count):.1f}:1")

    if anomaly_count < 10:
        result["message"] = f"Too few anomalies ({anomaly_count}). Need at least 10."
        return result

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)} samples (anomalies: {y_train.sum()})")
    print(f"Test: {len(X_test)} samples (anomalies: {y_test.sum()})")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE if needed
    if anomaly_ratio < 0.15 and anomaly_count >= 5:
        try:
            k = min(3, y_train.sum() - 1) if y_train.sum() > 1 else 1
            sm = SMOTE(random_state=42, k_neighbors=max(1, k))
            X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)
            print(f"✅ SMOTE applied: {len(X_train_scaled)} → {len(X_train_bal)} samples")
        except Exception as e:
            print(f"⚠️ SMOTE failed: {e}")
            X_train_bal, y_train_bal = X_train_scaled, y_train
    else:
        print("ℹ️ Using original data (sufficient anomalies)")
        X_train_bal, y_train_bal = X_train_scaled, y_train

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_bal, y_train_bal)
    rf_preds = rf.predict(X_test_scaled)
    
    rf_f1 = float(f1_score(y_test, rf_preds, zero_division=0))
    rf_precision = float(precision_score(y_test, rf_preds, zero_division=0))
    rf_recall = float(recall_score(y_test, rf_preds, zero_division=0))
    print(f"\n✅ Random Forest: F1={rf_f1:.4f}, P={rf_precision:.4f}, R={rf_recall:.4f}")

    # --- Isolation Forest ---
    normal_idx = np.where(y_train == 0)[0]
    X_normal = X_train_scaled[normal_idx]
    
    if len(X_normal) >= 20:
        iso = IsolationForest(
            n_estimators=100,
            contamination=0.05,  # Lower for rare anomalies
            random_state=42
        )
        iso.fit(X_normal)
        iso_preds = (iso.predict(X_test_scaled) == -1).astype(int)
        iso_f1 = float(f1_score(y_test, iso_preds, zero_division=0))
        iso_precision = float(precision_score(y_test, iso_preds, zero_division=0))
        iso_recall = float(recall_score(y_test, iso_preds, zero_division=0))
        print(f"✅ Isolation Forest: F1={iso_f1:.4f}, P={iso_precision:.4f}, R={iso_recall:.4f}")
    else:
        print("⚠️ Not enough normal samples for Isolation Forest.")
        iso = IsolationForest(n_estimators=50, contamination=0.1, random_state=42)
        iso.fit(X_train_scaled)
        iso_preds = (iso.predict(X_test_scaled) == -1).astype(int)
        iso_f1 = 0.0

    # --- Ensemble ---
    ensemble = EnsembleRiskPredictor(rf, iso, scaler, features)
    ensemble_preds = (ensemble.predict_proba(pd.DataFrame(X_test, columns=features)) > 0.5).astype(int)
    
    ensemble_f1 = float(f1_score(y_test, ensemble_preds, zero_division=0))
    ensemble_precision = float(precision_score(y_test, ensemble_preds, zero_division=0))
    ensemble_recall = float(recall_score(y_test, ensemble_preds, zero_division=0))
    print(f"✅ Ensemble: F1={ensemble_f1:.4f}, P={ensemble_precision:.4f}, R={ensemble_recall:.4f}")

    # Save models
    joblib.dump(ensemble, "models/ensemble_risk.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(rf, "models/anomaly_model.pkl")
    joblib.dump(iso, "models/isolation_forest.pkl")
    joblib.dump(features, "models/feature_names.pkl")

    result.update({
        "success": True,
        "message": "Model trained successfully with 6,426 records!",
        "rf_f1": round(rf_f1, 4),
        "rf_precision": round(rf_precision, 4),
        "rf_recall": round(rf_recall, 4),
        "iso_f1": round(iso_f1, 4),
        "iso_precision": round(iso_precision, 4),
        "iso_recall": round(iso_recall, 4),
        "ensemble_f1": round(ensemble_f1, 4),
        "ensemble_precision": round(ensemble_precision, 4),
        "ensemble_recall": round(ensemble_recall, 4),
        "trained_at": datetime.now().isoformat(),
        "confusion_matrix": confusion_matrix(y_test, ensemble_preds).tolist(),
    })
    
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    
    return result


def load_training_metadata():
    if not os.path.isfile(METADATA_PATH):
        return None
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def main():
    print("=" * 60)
    print("🚀 FINAL MODEL TRAINING")
    print("📊 Dataset: 6,426 records with 24% anomalies")
    print("=" * 60)
    
    outcome = run_training()
    
    if outcome["success"]:
        print(f"\n✅ {outcome['message']}")
        print(f"📊 Records: {outcome['samples']}")
        print(f"📈 Anomaly ratio: {outcome['anomaly_ratio'] * 100:.2f}%")
        print(f"\n--- Model Performance ---")
        print(f"  RF F1: {outcome['rf_f1']:.4f}")
        print(f"  RF Precision: {outcome['rf_precision']:.4f}")
        print(f"  RF Recall: {outcome['rf_recall']:.4f}")
        print(f"  IF F1: {outcome['iso_f1']:.4f}")
        print(f"  Ensemble F1: {outcome['ensemble_f1']:.4f}")
        print(f"\n✅ Metadata saved at: {METADATA_PATH}")
    else:
        print(f"\n❌ {outcome['message']}")


if __name__ == "__main__":
    main()