# shap_explain.py — SHAP-based explainability for the login risk Random Forest model

import io
import base64
from typing import Any, Optional

import numpy as np
import pandas as pd

FEATURE_LABELS = {
    'hour': 'Unusual login hour',
    'ip_changed': 'IP address changed',
    'device_changed': 'New device fingerprint',
    'old_risk_score': 'Elevated rule-based risk',
    'role_enc': 'Role mismatch',
    'isp_enc': 'Unusual ISP (not primary)',
}

_explainer_cache: dict = {}


def _get_tree_explainer(rf_model):
    import shap
    key = id(rf_model)
    if key not in _explainer_cache:
        _explainer_cache[key] = shap.TreeExplainer(rf_model)
    return _explainer_cache[key]


def _positive_class_shap(shap_values, row_index: int = 0) -> np.ndarray:
    """Extract SHAP vector for the positive (anomaly) class."""
    if isinstance(shap_values, list):
        arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        return np.asarray(arr)[row_index].flatten()
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[row_index, :, 1 if arr.shape[2] > 1 else 0].flatten()
    if arr.ndim == 2:
        return arr[row_index].flatten()
    return arr.flatten()


def _scalar_base_value(expected) -> float:
    ev = np.asarray(expected)
    if ev.ndim == 0:
        return float(ev.item())
    if ev.ndim == 1:
        return float(ev[1].item() if len(ev) > 1 else ev[0].item())
    return float(ev.reshape(-1)[1 if ev.size > 1 else 0].item())


def explain_risk(ensemble, X_df: pd.DataFrame) -> Optional[dict[str, Any]]:
    """
    Compute SHAP contributions for one login feature row.
    Returns percentages, labels, and a short summary for storage/UI.
    """
    if ensemble is None or X_df is None or len(X_df) == 0:
        return None
    try:
        X_scaled = ensemble.scaler.transform(X_df)
        explainer = _get_tree_explainer(ensemble.rf)
        shap_values = explainer.shap_values(X_scaled)
        sv = _positive_class_shap(shap_values, 0)

        base_value = _scalar_base_value(explainer.expected_value)

        abs_sv = np.abs(sv)
        total = float(abs_sv.sum()) or 1.0

        contributions = []
        for i, fname in enumerate(ensemble.feature_names):
            shap_val = float(np.asarray(sv[i]).item())
            contributions.append({
                'feature': fname,
                'label': FEATURE_LABELS.get(fname, fname),
                'shap_value': round(shap_val, 5),
                'pct': round(float(abs_sv[i] / total * 100), 1),
            })
        contributions.sort(key=lambda x: x['pct'], reverse=True)

        top = contributions[:5]
        summary = ', '.join(f"{c['label']} (+{c['pct']}%)" for c in top[:3])

        return {
            'method': 'shap_tree_explainer',
            'base_value': round(base_value, 5),
            'predicted_logit': round(base_value + float(np.sum(sv)), 5),
            'contributions': contributions,
            'top_factors': top,
            'summary': summary,
            'labels': [c['label'] for c in top[:3]],
        }
    except Exception as e:
        print(f"SHAP explain error: {e}")
        return None


def merge_geo_shap(shap_data: Optional[dict], geo_labels: list) -> dict:
    """Attach rule/geo factors alongside ML SHAP output for audit storage."""
    payload = shap_data.copy() if shap_data else {'method': 'rule_only', 'contributions': [], 'labels': []}
    payload['geo_factors'] = geo_labels or []
    if geo_labels and 'summary' in payload:
        geo_part = '; '.join(geo_labels[:2])
        payload['summary'] = f"{payload.get('summary', '')} | Geo: {geo_part}".strip(' |')
    elif geo_labels:
        payload['summary'] = f"Geo: {', '.join(geo_labels)}"
    return payload


def aggregate_shap_contributions(shap_list: list) -> tuple[list[str], list[float]]:
    """Average |SHAP| % per feature across multiple login explanations."""
    from collections import defaultdict
    totals: dict[str, list[float]] = defaultdict(list)
    for shap_data in shap_list:
        if not shap_data or not shap_data.get('contributions'):
            continue
        for c in shap_data['contributions']:
            totals[c['feature']].append(c['pct'])
    if not totals:
        return [], []
    avgs = sorted(((f, sum(vs) / len(vs)) for f, vs in totals.items()), key=lambda x: x[1], reverse=True)
    labels = [FEATURE_LABELS.get(f, f) for f, _ in avgs]
    values = [round(v, 1) for _, v in avgs]
    return labels, values


def shap_bar_chart_base64(shap_data: dict) -> Optional[str]:
    """Render horizontal bar chart of SHAP % contributions as base64 PNG."""
    if not shap_data or not shap_data.get('contributions'):
        return None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        items = shap_data['contributions'][:6]
        labels = [f"{c['label']} ({c['pct']}%)" for c in reversed(items)]
        values = [c['pct'] for c in reversed(items)]
        colors = ['#e74c3c' if c['shap_value'] > 0 else '#3498db' for c in reversed(items)]

        fig, ax = plt.subplots(figsize=(8, max(3, len(items) * 0.55)))
        ax.barh(labels, values, color=colors)
        ax.set_xlabel('Risk contribution (%)')
        ax.set_title('SHAP feature importance (|SHAP| normalized)')
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('ascii')
    except Exception as e:
        print(f"SHAP chart error: {e}")
        return None
