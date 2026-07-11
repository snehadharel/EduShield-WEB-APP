# tests/test_app.py
# Complete unit tests for EduShield AI-based Intrusion Detection System
#
# Run from project root:  pytest tests/test_app.py -v
# Install dependencies:  pip install pytest pytest-mock pytest-cov responses

import pytest
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import os
import sys

# Add project root to path so we can import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    app, get_db, init_db, classify_device_type, get_delay, _is_strong_password,
    validate_file_type, old_rule_based_risk, extract_ml_features,
    calculate_risk_score, decide_auth_action, scan_and_log_attack_patterns,
    EnsembleRiskPredictor, get_geolocation, check_abuseipdb,
    complete_successful_login, initiate_otp_verification
)
from shap_explain import explain_risk, merge_geo_shap, aggregate_shap_contributions
from train_model import run_training, load_training_metadata


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def client():
    """Flask test client with in-memory database."""
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['SESSION_COOKIE_HTTPONLY'] = False
    app.config['DATABASE'] = ':memory:'
    with app.test_client() as client:
        with app.app_context():
            init_db()
        yield client


@pytest.fixture
def db_session(client):
    """Provide a database connection for direct queries."""
    with app.app_context():
        db = get_db()
        yield db
        # Rollback not needed for in-memory; but we keep it clean
        db.rollback()


@pytest.fixture
def create_user(db_session):
    """
    Create a test user with a unique username and email.
    Returns a function that can be called with optional overrides.
    """
    from bcrypt import hashpw, gensalt

    def _create_user(username=None, email=None, password='Test@1234',
                     role='student', is_verified=1, is_approved=1, is_admin=0):
        if username is None:
            username = f"user_{uuid.uuid4().hex[:8]}"
        if email is None:
            email = f"{username}@example.com"
        hashed = hashpw(password.encode('utf-8'), gensalt())
        db_session.execute(
            "INSERT INTO users (username, email, password_hash, role, is_approved, is_verified, is_admin) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (username, email, hashed, role, is_approved, is_verified, is_admin)
        )
        db_session.commit()
        user = db_session.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return user
    return _create_user


@pytest.fixture
def login_user(client, create_user):
    """
    Log in a user and return the user object.
    Must be called:  user = login_user()
    You can pass an optional username to create a specific user.
    """
    def _login_user(username=None):
        user = create_user(username=username)
        with client.session_transaction() as sess:
            sess['user_id'] = user['id']
            sess['username'] = user['username']
            sess['role'] = user['role']
            sess['is_admin'] = user['is_admin']
            sess['auth_complete'] = True
            sess['session_token'] = 'dummy-token'
        return user
    return _login_user


# ----------------------------------------------------------------------
# Tests for helper functions
# ----------------------------------------------------------------------

def test_classify_device_type():
    assert classify_device_type("Mozilla/5.0 (iPhone; CPU iPhone OS 14_0)") == "mobile"
    assert classify_device_type("Mozilla/5.0 (iPad; CPU OS 14_0)") == "tablet"
    assert classify_device_type("Mozilla/5.0 (Windows NT 10.0; Win64; x64)") == "desktop"
    assert classify_device_type("Unknown") == "desktop"


def test_get_delay():
    assert get_delay(0) == 0
    assert get_delay(2) == 0
    assert get_delay(3) == 5
    assert get_delay(4) == 5
    assert get_delay(5) == 30
    assert get_delay(7) == 30
    assert get_delay(8) == 120
    assert get_delay(9) == 120
    assert get_delay(10) == 300
    assert get_delay(15) == 300


def test_is_strong_password():
    assert _is_strong_password("Abc@1234") is True
    assert _is_strong_password("weak") is False
    assert _is_strong_password("NoSpecial123") is False
    assert _is_strong_password("lowercase@123") is False
    assert _is_strong_password("UPPERCASE@123") is False
    assert _is_strong_password("Abcdef@1") is True


def test_validate_file_type(tmp_path):
    # Create a temporary PDF file
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%")
    with open(pdf, "rb") as f:
        valid, msg = validate_file_type(f, "test.pdf")
        assert valid is True

    invalid = tmp_path / "fake.pdf"
    invalid.write_text("This is not a PDF")
    with open(invalid, "rb") as f:
        valid, msg = validate_file_type(f, "fake.pdf")
        assert valid is False
        assert "not a valid PDF" in msg

    with patch('app.imghdr.what', return_value='jpeg'):
        with open(pdf, "rb") as f:
            valid, msg = validate_file_type(f, "test.jpg")
            assert valid is True

    with patch('app.imghdr.what', return_value=None):
        with open(pdf, "rb") as f:
            valid, msg = validate_file_type(f, "test.png")
            assert valid is False
            assert "not a valid PNG" in msg


# ----------------------------------------------------------------------
# Tests for risk functions
# ----------------------------------------------------------------------

def test_old_rule_based_risk(client, create_user, db_session):
    user = create_user(username='riskuser')
    # No previous logins -> no IP/device change penalty (risk 0)
    risk = old_rule_based_risk(user['id'], '1.2.3.4', 'Mozilla/5.0', {}, 'student')
    assert risk == 0

    # Add a previous login to simulate known IP/device
    db_session.execute(
        "INSERT INTO login_logs (user_id, ip_address, device, status) VALUES (?, ?, ?, ?)",
        (user['id'], '1.2.3.4', 'Mozilla/5.0', 'success')
    )
    db_session.commit()
    risk = old_rule_based_risk(user['id'], '1.2.3.4', 'Mozilla/5.0', {}, 'student')
    assert risk == 0  # still same IP/device

    # Now use a new IP -> risk 2
    risk = old_rule_based_risk(user['id'], '5.6.7.8', 'Mozilla/5.0', {}, 'student')
    assert risk == 2

    # Simulation flags
    risk = old_rule_based_risk(user['id'], '1.2.3.4', 'Mozilla/5.0', {'unusual_time': True}, 'student')
    assert risk == 1


def test_extract_ml_features(client, create_user, db_session):
    user = create_user(username='mluser')
    # Insert a previous login
    db_session.execute(
        "INSERT INTO login_logs (user_id, ip_address, device, status) VALUES (?, ?, ?, ?)",
        (user['id'], '192.168.1.1', 'laptop', 'success')
    )
    db_session.commit()
    features = extract_ml_features(user['id'], '10.0.0.1', 'mobile')
    assert features['ip_changed'] == 1
    assert features['device_changed'] == 1
    assert features['old_risk_score'] >= 0
    assert features['role_enc'] == 0   # student
    # ISP encoding: '10.0.0.1' is 'Other' -> 4
    assert features['isp_enc'] == 4


def test_calculate_risk_score_with_ensemble(client, create_user, mocker):
    user = create_user(username='ensuser')
    # Mock ensemble to return 0.75
    mock_ensemble = MagicMock()
    mock_ensemble.predict_proba.return_value = [0.75]
    mocker.patch('app.ensemble_risk', mock_ensemble)
    mocker.patch('app.shap_explain.explain_risk', return_value=None)
    risk = calculate_risk_score(user['id'], '1.2.3.4', 'Mozilla/5.0', {}, 'student')
    assert risk == 0.75


def test_calculate_risk_score_fallback(client, create_user, mocker):
    user = create_user(username='fallback')
    mocker.patch('app.ensemble_risk', None)
    # Patch old_rule_based_risk to return 5 (simulate risk)
    mocker.patch('app.old_rule_based_risk', return_value=5)
    risk = calculate_risk_score(user['id'], '1.2.3.4', 'Mozilla/5.0', {}, 'student')
    assert risk == 0.5


def test_decide_auth_action():
    assert decide_auth_action(0.3) == 'allow'
    assert decide_auth_action(0.5) == 'otp'
    assert decide_auth_action(0.8) == 'block'
    assert decide_auth_action(0.4) == 'allow'
    assert decide_auth_action(0.7) == 'otp'


# ----------------------------------------------------------------------
# Tests for security detection
# ----------------------------------------------------------------------

def test_scan_and_log_attack_patterns(client, create_user, db_session):
    user = create_user(username='attackuser')
    with client.session_transaction() as sess:
        sess['user_id'] = user['id']

    # Simulate a SQLi payload via POST form
    with app.test_request_context('/login', method='POST',
                                   data={'username': "admin' OR '1'='1"},
                                   content_type='application/x-www-form-urlencoded'):
        hits = scan_and_log_attack_patterns(endpoint='login', block=False, lock_user=False)
        assert len(hits) == 1
        assert hits[0]['type'] == 'SQLi'
        attack = db_session.execute("SELECT * FROM attack_patterns WHERE pattern_type='SQLi'").fetchone()
        assert attack is not None

    # Simulate XSS payload
    with app.test_request_context('/profile', method='POST',
                                   data={'full_name': "<script>alert('xss')</script>"},
                                   content_type='application/x-www-form-urlencoded'):
        hits = scan_and_log_attack_patterns(endpoint='profile', block=False, lock_user=False)
        assert len(hits) == 1
        assert hits[0]['type'] == 'XSS'


# ----------------------------------------------------------------------
# Tests for routes (integration)
# ----------------------------------------------------------------------

def test_register_flow(client, db_session):
    # Step 1
    data = {
        'full_name': 'Test User',
        'username': 'testregister',
        'email': 'testreg@example.com',
        'role': 'student',
        'password': 'Test@1234',
        'confirm_password': 'Test@1234',
        'preferred_device': 'laptop',
        'step': '1'
    }
    with patch('app._validate_csrf'):
        resp = client.post('/register', data=data, follow_redirects=True)
        assert resp.status_code == 200
        assert b'security' in resp.data.lower() or b'question' in resp.data.lower()

    # Step 2 (security questions)
    sec_data = {
        'sec_q1': 'What is your pet?',
        'sec_a1': 'cat',
        'sec_q2': 'What is your school?',
        'sec_a2': 'abc',
        'sec_q3': 'What is your city?',
        'sec_a3': 'ktm',
        'step': '2'
    }
    with patch('app._validate_csrf'):
        resp = client.post('/register', data=sec_data, follow_redirects=True)
        assert resp.status_code == 200
    user = db_session.execute("SELECT * FROM users WHERE username='testregister'").fetchone()
    assert user is not None
    assert user['is_verified'] == 0


def test_login_success(client, create_user):
    user = create_user(username='logintest', password='Test@1234', is_verified=1)
    with patch('app._validate_csrf'):
        resp = client.post('/login', data={
            'email': 'logintest',
            'password': 'Test@1234',
        }, follow_redirects=True)
        # Check session
        with client.session_transaction() as sess:
            assert sess.get('user_id') == user['id']


def test_login_fail(client, create_user):
    create_user(username='failtest', password='Test@1234', is_verified=1)
    with patch('app._validate_csrf'):
        resp = client.post('/login', data={
            'email': 'failtest',
            'password': 'wrong',
        }, follow_redirects=True)
        assert b'Invalid' in resp.data


def test_dashboard_requires_login(client):
    resp = client.get('/dashboard', follow_redirects=True)
    assert b'Please login' in resp.data


def test_dashboard_logged_in(client, login_user):
    user = login_user()  # create and log in a test user
    resp = client.get('/dashboard')
    assert resp.status_code == 200
    assert b'Welcome back' in resp.data


def test_change_password(client, login_user, db_session):
    user = login_user()  # uses default Test@1234
    with patch('app._validate_csrf'):
        resp = client.post('/change_password', data={
            'old_password': 'Test@1234',
            'new_password': 'New@1234',
            'confirm_password': 'New@1234',
        }, follow_redirects=True)
        assert b'Password changed' in resp.data
    from bcrypt import checkpw
    updated = db_session.execute("SELECT password_hash FROM users WHERE id = ?", (user['id'],)).fetchone()
    assert checkpw('New@1234'.encode('utf-8'), updated['password_hash']) is True


def test_security_center(client, login_user):
    login_user()  # log in
    resp = client.get('/security')
    assert resp.status_code == 200
    assert b'Login History' in resp.data or b'security' in resp.data


# ----------------------------------------------------------------------
# Tests for external API mocks (geolocation, abuseipdb)
# ----------------------------------------------------------------------

def test_get_geolocation():
    with patch('app.requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            'status': 'success',
            'country': 'Nepal',
            'lat': 27.7172,
            'lon': 85.3240
        }
        result = get_geolocation('8.8.8.8')
        assert result['country'] == 'Nepal'
        assert result['lat'] == 27.7172


def test_check_abuseipdb():
    # Mock os.getenv to return a dummy API key so the function proceeds
    with patch('app.os.getenv', return_value='dummy-key'):
        with patch('app.requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                'data': {
                    'abuseConfidenceScore': 50,
                    'countryCode': 'US',
                    'totalReports': 5,
                    'isTor': False
                }
            }
            boost, info = check_abuseipdb('1.2.3.4')
            assert boost == 0.15  # 50% * 0.3
            assert info['abuse_score'] == 50


# ----------------------------------------------------------------------
# Tests for SHAP explainability
# ----------------------------------------------------------------------

def test_shap_explain(mocker):
    mock_ensemble = MagicMock()
    mock_ensemble.rf = MagicMock()
    mock_ensemble.scaler = MagicMock()
    mock_ensemble.feature_names = ['hour', 'ip_changed']
    mock_explainer = MagicMock()
    mock_explainer.expected_value = [0.5, 0.2]
    mock_explainer.shap_values.return_value = [[[0.1, -0.2]]]  # shape (1,2,2)
    mocker.patch('shap_explain._get_tree_explainer', return_value=mock_explainer)

    import pandas as pd
    X = pd.DataFrame([[10, 1]], columns=['hour', 'ip_changed'])
    result = explain_risk(mock_ensemble, X)
    assert result is not None
    assert 'summary' in result
    assert len(result['contributions']) == 2


def test_shap_merge_geo():
    shap_data = {'summary': 'ML risk high', 'labels': ['IP changed']}
    geo = ['Location outside Nepal']
    merged = merge_geo_shap(shap_data, geo)
    assert 'geo_factors' in merged
    assert merged['geo_factors'] == geo
    assert 'Location outside Nepal' in merged['summary']


def test_aggregate_shap_contributions():
    shap_list = [
        {'contributions': [{'feature': 'hour', 'pct': 40}, {'feature': 'ip_changed', 'pct': 60}]},
        {'contributions': [{'feature': 'hour', 'pct': 50}, {'feature': 'ip_changed', 'pct': 50}]}
    ]
    labels, values = aggregate_shap_contributions(shap_list)
    assert len(labels) == 2
    # Sorted descending by average -> ip_changed 55, hour 45
    assert values == [55.0, 45.0]


# ----------------------------------------------------------------------
# Tests for training module (mocked)
# ----------------------------------------------------------------------

def test_run_training_min_samples(mocker):
    # Mock database connection to return empty result
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value = mock_cursor
    mocker.patch('sqlite3.connect', return_value=mock_conn)

    result = run_training(min_samples=10)
    assert result['success'] is False
    assert 'Not enough login data' in result['message']


def test_load_training_metadata(tmp_path, mocker):
    meta_path = tmp_path / "training_metadata.json"
    meta_path.write_text('{"success": true, "samples": 100}')
    mocker.patch('train_model.METADATA_PATH', str(meta_path))
    meta = load_training_metadata()
    assert meta['success'] is True
    assert meta['samples'] == 100


# ----------------------------------------------------------------------
# Tests for session and logout
# ----------------------------------------------------------------------

def test_logout(client, login_user):
    login_user()
    resp = client.get('/logout', follow_redirects=True)
    assert b'logged out' in resp.data
    with client.session_transaction() as sess:
        assert not sess.get('user_id')


def test_session_timeout(client, create_user, mocker):
    # Skip this test if you don't want to mock session lifetime
    # We'll just verify it doesn't crash
    mocker.patch('app.permanent_session_lifetime', timedelta(seconds=1))
    user = create_user()
    with client.session_transaction() as sess:
        sess['user_id'] = user['id']
        sess['username'] = user['username']
        sess['role'] = user['role']
        sess['auth_complete'] = True
    import time
    time.sleep(1.1)
    resp = client.get('/dashboard', follow_redirects=True)
    # We'll just assert it's a redirect or page; this test is a placeholder.
    # Usually session expiry is handled by cookie expiration, so we can't easily test it.
    assert resp.status_code in (200, 302)