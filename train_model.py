# train_model.py
# Trains Random Forest + Isolation Forest ensemble, saves scaler and feature names.

import sqlite3
import json
import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
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

    def explain(self, X_instance):
        try:
            import shap_explain
            result = shap_explain.explain_risk(self, X_instance)
            if result:
                return {c['label']: c['shap_value'] for c in result['top_factors']}
        except Exception as e:
            print(f"SHAP explain failed: {e}")
        return {}


def _save_metadata(metadata):
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_training_metadata():
    if not os.path.isfile(METADATA_PATH):
        return None
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def run_training(database_path="academic_portal.db", min_samples=50):
    """Train ensemble model and return metrics dict for the admin dashboard."""
    result = {
        "success": False,
        "message": "",
        "samples": 0,
        "anomaly_ratio": 0.0,
        "rf_f1": None,
        "iso_f1": None,
        "trained_at": None,
        "features": [],
    }

    if not os.path.isfile(database_path):
        result["message"] = f"Database not found: {database_path}"
        return result

    conn = sqlite3.connect(database_path)
    query = """
    SELECT
        l.risk_score            AS old_risk_score,
        l.status,
        l.action,
        COALESCE(u.role, 'student') AS role,
        strftime('%H', l.login_time) AS hour,
        l.ip_address,
        CASE
            WHEN l.ip_address LIKE '192.168%%' THEN 'College_WiFi'
            WHEN l.ip_address LIKE '202.70%%'  THEN 'WorldLink'
            WHEN l.ip_address LIKE '103.1%%'   THEN 'Ncell'
            WHEN l.ip_address LIKE '27.34%%'   THEN 'NTC'
            ELSE 'Other'
        END AS isp_type,
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
        END AS device_changed
    FROM login_logs l
    LEFT JOIN users u ON l.user_id = u.id
    WHERE l.status != 'otp_sent'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    result["samples"] = len(df)
    if len(df) < min_samples:
        result["message"] = (
            f"Not enough login data ({len(df)} records). "
            f"Need at least {min_samples} records to train."
        )
        _save_metadata({**result, "trained_at": datetime.now().isoformat()})
        return result

    df["is_anomaly"] = (
        (df["status"] == "blocked") | (df["action"] == "otp_required")
    ).astype(int)

    role_enc = LabelEncoder()
    df["role_enc"] = role_enc.fit_transform(df["role"].fillna("student"))

    isp_enc = LabelEncoder()
    df["isp_enc"] = isp_enc.fit_transform(df["isp_type"].fillna("Other"))

    features = [
        "hour", "ip_changed", "device_changed", "old_risk_score", "role_enc", "isp_enc",
    ]

    X = df[features].fillna(0).astype(float)
    y = df["is_anomaly"]
    result["anomaly_ratio"] = round(float(y.mean()), 4)
    result["features"] = features

    if y.nunique() < 2:
        result["message"] = "Need both normal and anomalous login records to train."
        _save_metadata({**result, "trained_at": datetime.now().isoformat()})
        return result

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train_bal)
    rf_preds = rf.predict(X_test_scaled)
    rf_f1 = float(f1_score(y_test, rf_preds))

    X_normal = X_train_scaled[y_train_bal == 0]
    if len(X_normal) < 2:
        result["message"] = "Not enough normal samples for Isolation Forest."
        _save_metadata({**result, "trained_at": datetime.now().isoformat()})
        return result

    iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    iso.fit(X_normal)
    iso_preds = (iso.predict(X_test_scaled) == -1).astype(int)
    iso_f1 = float(f1_score(y_test, iso_preds))

    ensemble = EnsembleRiskPredictor(rf, iso, scaler, features)
    joblib.dump(ensemble, "models/ensemble_risk.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(role_enc, "models/role_encoder.pkl")
    joblib.dump(isp_enc, "models/isp_encoder.pkl")
    joblib.dump(features, "models/feature_names.pkl")
    joblib.dump(rf, "models/anomaly_model.pkl")
    joblib.dump(iso, "models/isolation_forest.pkl")

    trained_at = datetime.now().isoformat()
    result.update({
        "success": True,
        "message": "Model trained and saved successfully.",
        "rf_f1": round(rf_f1, 4),
        "iso_f1": round(iso_f1, 4),
        "trained_at": trained_at,
    })
    _save_metadata(result)
    return result


def main():
    print("Loading login data from database...")
    outcome = run_training()
    print(outcome["message"] or f"Trained on {outcome['samples']} records.")
    if outcome.get("rf_f1") is not None:
        print(f"Random Forest F1: {outcome['rf_f1']}")
        print(f"Isolation Forest F1: {outcome['iso_f1']}")
    if outcome["success"]:
        print("All models saved to models/ folder.")


if __name__ == "__main__":
    main()
