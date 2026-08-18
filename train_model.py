# train_model_fixed.py
# Improved training with better handling of extreme class imbalance

import sqlite3
import json
import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

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
    """Train ensemble with extreme class imbalance handling."""
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
        "confusion_matrix": None,
    }

    if not os.path.isfile(database_path):
        result["message"] = f"Database not found: {database_path}"
        return result

    conn = sqlite3.connect(database_path)
    
    query = """
    SELECT
        l.user_id,
        strftime('%H', l.login_time) AS hour,
        l.ip_address,
        l.device,
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
        result["message"] = f"Not enough data ({len(df)} records). Need {min_samples}."
        return result

    # Encode categorical variables
    role_map = {'student': 0, 'teacher': 1, 'admin': 2}
    df['role_enc'] = df['role'].map(role_map).fillna(0)
    
    isp_map = {'College_WiFi': 0, 'WorldLink': 1, 'Ncell': 2, 'NTC': 3, 'Other': 4}
    df['isp_enc'] = df['isp_type'].map(isp_map).fillna(4)

    # Features
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
    print(f"Anomalies: {anomaly_count} ({anomaly_ratio*100:.2f}%)")
    print(f"Imbalance ratio: {normal_count/anomaly_count:.1f}:1")

    if anomaly_count < 10:
        result["message"] = f"Too few anomalies ({anomaly_count}). Need at least 10."
        return result

    # --- Split data with stratification ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTrain: {len(X_train)} samples (anomalies: {y_train.sum()})")
    print(f"Test: {len(X_test)} samples (anomalies: {y_test.sum()})")

    # --- Scale features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Apply SMOTE with careful parameters ---
    # Only apply if anomaly_count > 3
    if anomaly_count >= 5:
        try:
            # Use SMOTE with k_neighbors limited by available anomalies
            k_neighbors = min(3, y_train.sum() - 1) if y_train.sum() > 1 else 1
            sm = SMOTE(random_state=42, k_neighbors=max(1, k_neighbors))
            X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)
            print(f"✅ SMOTE applied: {len(X_train_scaled)} → {len(X_train_bal)} samples")
        except Exception as e:
            print(f"⚠️ SMOTE failed: {e}. Using original data.")
            X_train_bal, y_train_bal = X_train_scaled, y_train
    else:
        X_train_bal, y_train_bal = X_train_scaled, y_train

    # --- Train Random Forest with balanced class weight ---
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
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

    # --- Train Isolation Forest on normal data only ---
    normal_indices = np.where(y_train == 0)[0]  # Use original y_train, not balanced
    X_normal = X_train_scaled[normal_indices]
    
    if len(X_normal) >= 20:
        iso = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )
        iso.fit(X_normal)
        iso_preds = (iso.predict(X_test_scaled) == -1).astype(int)
        iso_f1 = float(f1_score(y_test, iso_preds, zero_division=0))
        iso_precision = float(precision_score(y_test, iso_preds, zero_division=0))
        iso_recall = float(recall_score(y_test, iso_preds, zero_division=0))
        print(f"✅ Isolation Forest: F1={iso_f1:.4f}, P={iso_precision:.4f}, R={iso_recall:.4f}")
    else:
        print("⚠️ Not enough normal samples for Isolation Forest. Skipping...")
        iso = IsolationForest(n_estimators=50, contamination=0.1, random_state=42)
        iso.fit(X_train_scaled)
        iso_preds = (iso.predict(X_test_scaled) == -1).astype(int)
        iso_f1 = 0.0

    # --- Ensemble ---
    ensemble = EnsembleRiskPredictor(rf, iso, scaler, features)
    
    # Calculate ensemble predictions
    rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
    iso_pred = iso.predict(X_test_scaled)
    iso_score = (iso_pred == -1).astype(float)
    
    # Weighted average
    ensemble_proba = 0.7 * rf_proba + 0.3 * iso_score
    ensemble_preds = (ensemble_proba > 0.5).astype(int)
    
    ensemble_f1 = float(f1_score(y_test, ensemble_preds, zero_division=0))
    ensemble_precision = float(precision_score(y_test, ensemble_preds, zero_division=0))
    ensemble_recall = float(recall_score(y_test, ensemble_preds, zero_division=0))
    print(f"✅ Ensemble: F1={ensemble_f1:.4f}, P={ensemble_precision:.4f}, R={ensemble_recall:.4f}")

    # --- Save models ---
    joblib.dump(ensemble, "models/ensemble_risk.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(rf, "models/anomaly_model.pkl")
    joblib.dump(iso, "models/isolation_forest.pkl")
    joblib.dump(features, "models/feature_names.pkl")

    # --- Save metadata ---
    trained_at = datetime.now().isoformat()
    result.update({
        "success": True,
        "message": "Model trained successfully.",
        "rf_f1": round(rf_f1, 4),
        "rf_precision": round(rf_precision, 4),
        "rf_recall": round(rf_recall, 4),
        "iso_f1": round(iso_f1, 4),
        "ensemble_f1": round(ensemble_f1, 4),
        "ensemble_precision": round(ensemble_precision, 4),
        "ensemble_recall": round(ensemble_recall, 4),
        "trained_at": trained_at,
        "confusion_matrix": confusion_matrix(y_test, ensemble_preds).tolist(),
        "classification_report": classification_report(y_test, ensemble_preds, zero_division=0)
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
    print("EduShield Model Training (Extreme Imbalance Handling)")
    print("=" * 60)
    
    outcome = run_training()
    
    if outcome["success"]:
        print(f"\n✅ {outcome['message']}")
        print(f"📊 Training samples: {outcome['samples']}")
        print(f"📈 Anomaly ratio: {outcome['anomaly_ratio'] * 100:.2f}%")
        print(f"\n--- Results ---")
        print(f"  Random Forest F1: {outcome['rf_f1']:.4f}")
        print(f"  Isolation Forest F1: {outcome['iso_f1']:.4f}")
        print(f"  Ensemble F1: {outcome['ensemble_f1']:.4f}")
    else:
        print(f"\n❌ {outcome['message']}")


if __name__ == "__main__":
    main()