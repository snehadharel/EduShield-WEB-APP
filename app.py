# app.py - Artificial Intelligence-based Intrusion Detection System
import sqlite3
import bcrypt
import random
import string
import os
import re
import secrets
import json
import math
import urllib.request
import urllib.error
import ipaddress
import time
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, g, flash, abort, jsonify, Response
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import pyotp
import qrcode
import base64
from io import BytesIO
import numpy as np
import pandas as pd
import smtplib
import imghdr
import zipfile
from io import BytesIO
from email.message import EmailMessage
from werkzeug.utils import secure_filename
import requests
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO
import warnings
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
import sendgrid
from sendgrid.helpers.mail import Mail

# Try to import shap_explain (optional)
try:
    import shap_explain
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap_explain module not found. SHAP explanations disabled.")
    class DummyShap:
        def explain_risk(self, *args, **kwargs): return None
        def merge_geo_shap(self, *args, **kwargs): return None
        def shap_bar_chart_base64(self, *args, **kwargs): return None
        def aggregate_shap_contributions(self, *args, **kwargs): return [], []
    shap_explain = DummyShap()

warnings.filterwarnings("ignore", category=UserWarning)

def validate_file_type(file_stream, filename):
    """
    Validate that the actual file content matches its extension.
    Returns (is_valid, error_message).
    """
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in ALLOWED_DOC_EXTENSIONS:
        return False, f"Extension '{ext}' is not allowed."

    # Read first 1024 bytes for magic header checks
    file_stream.seek(0)
    header = file_stream.read(1024)
    file_stream.seek(0)  # Reset stream for later reading

    if not header:
        return False, "Uploaded file is empty."

    # ---- Image validation (JPEG, PNG) ----
    if ext in ['jpg', 'jpeg']:
        img_type = imghdr.what(file_stream)
        file_stream.seek(0)
        if img_type not in ['jpeg', 'jpg']:
            return False, "File is not a valid JPEG image."
        return True, "OK"

    if ext == 'png':
        img_type = imghdr.what(file_stream)
        file_stream.seek(0)
        if img_type != 'png':
            return False, "File is not a valid PNG image."
        return True, "OK"

    # ---- PDF validation ----
    if ext == 'pdf':
        if not header.startswith(b'%PDF'):
            return False, "File is not a valid PDF document."
        return True, "OK"

    # ---- DOCX validation (must be a valid ZIP containing word/document.xml) ----
    if ext == 'docx':
        if not header.startswith(b'PK\x03\x04'):
            return False, "File is not a valid DOCX archive."
        try:
            file_stream.seek(0)
            with zipfile.ZipFile(BytesIO(file_stream.read())) as zf:
                if 'word/document.xml' not in zf.namelist():
                    return False, "Invalid DOCX file: missing word/document.xml."
        except zipfile.BadZipFile:
            return False, "Invalid DOCX file: corrupted ZIP archive."
        finally:
            file_stream.seek(0)
        return True, "OK"

    # ---- DOC (legacy binary format) ----
    # We cannot reliably magic‑check .doc, but we accept it.
    if ext == 'doc':
        return True, "OK"

    # ---- TXT ----
    if ext == 'txt':
        return True, "OK"

    # Fallback
    return False, f"Unsupported file type '{ext}'."

# ---------- SQLite datetime adapter ----------
def _adapt_datetime(dt):
    return dt.isoformat()

def _convert_datetime(s):
    return datetime.fromisoformat(s.decode())

sqlite3.register_adapter(datetime, _adapt_datetime)
sqlite3.register_converter("timestamp", _convert_datetime)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
app.permanent_session_lifetime = timedelta(minutes=10)

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=(os.getenv('SESSION_COOKIE_SECURE', '0') == '1'),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024
)

ALLOWED_ROLES = {'student', 'teacher'}
ALLOWED_LOGIN_DEVICES = {'mobile', 'laptop', 'desktop', 'tablet', 'other'}
ALLOWED_DOC_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt', 'jpg', 'jpeg', 'png'}

limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day", "50 per hour"], storage_uri="memory://")

serializer = URLSafeTimedSerializer(app.secret_key)

# ---------- Helper: UTC to Nepal time ----------
def utc_to_nepal_time(utc_dt):
    if utc_dt is None:
        return None
    if isinstance(utc_dt, str):
        try:
            utc_dt = datetime.fromisoformat(utc_dt)
        except:
            return utc_dt
    return utc_dt + timedelta(hours=5, minutes=45)

# ---------- EnsembleRiskPredictor class ----------
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
        if SHAP_AVAILABLE:
            result = shap_explain.explain_risk(self, X_instance)
            if result:
                return {c['label']: c['shap_value'] for c in result['top_factors']}
        return {}

# ---------- Load ML ensemble model ----------
def _load_ensemble_model():
    try:
        return joblib.load('models/ensemble_risk.pkl')
    except Exception:
        pass
    try:
        rf = joblib.load('models/anomaly_model.pkl')
        iso = joblib.load('models/isolation_forest.pkl')
        scaler = joblib.load('models/scaler.pkl')
        features = joblib.load('models/feature_names.pkl')
        return EnsembleRiskPredictor(rf, iso, scaler, features)
    except Exception as e:
        print(f"Ensemble model not loaded: {e}. Using fallback rule-based risk.")
        return None

ensemble_risk = _load_ensemble_model()
if ensemble_risk is not None:
    print("Ensemble risk model loaded successfully.")

def reload_ensemble_model():
    global ensemble_risk
    ensemble_risk = _load_ensemble_model()
    return ensemble_risk is not None

DATABASE = 'academic_portal.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE, detect_types=sqlite3.PARSE_DECLTYPES)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# ---------- Database initialization (complete, includes device_fingerprint) ----------
def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()

        # Users table with device_fingerprint
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0,
                preferred_device TEXT,
                security_q1 TEXT,
                security_q2 TEXT,
                security_q3 TEXT,
                security_a1 TEXT,
                security_a2 TEXT,
                security_a3 TEXT,
                is_locked INTEGER DEFAULT 0,
                locked_at TIMESTAMP,
                is_banned INTEGER DEFAULT 0,
                session_token TEXT,
                is_approved INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                device_fingerprint TEXT
            )
        ''')
        for col in ['is_banned', 'session_token', 'is_approved', 'device_fingerprint']:
            try:
                cursor.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass

# Add TOTP columns (safe to run even if they already exist)
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN totp_secret TEXT")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE users ADD COLUMN totp_enabled INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
# Add TOTP backup codes column
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN totp_backup_codes TEXT")
        except sqlite3.OperationalError:
            pass
        # Add is_verified column
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN is_verified INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass

# Add password_changed_at column
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN password_changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        except sqlite3.OperationalError:
            pass
        # Login logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS login_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                ip_address TEXT,
                device TEXT,
                login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT,
                risk_score INTEGER,
                action TEXT,
                alert_sent INTEGER DEFAULT 0,
                risk_factors TEXT,
                geo_country TEXT,
                geo_lat REAL,
                geo_lon REAL,
                shap_json TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        for col in ['risk_factors', 'geo_country', 'geo_lat', 'geo_lon', 'shap_json']:
            try:
                cursor.execute(f"ALTER TABLE login_logs ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass

        # Failed attempts tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS failed_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ip_failed_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ip_address TEXT NOT NULL,
                attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Attack patterns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attack_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                ip_address TEXT,
                endpoint TEXT,
                method TEXT,
                param_key TEXT,
                param_value TEXT,
                pattern_type TEXT,
                pattern TEXT,
                severity INTEGER DEFAULT 1,
                blocked INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        # User activity logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                action TEXT,
                details TEXT,
                ip_address TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        # Academic portal tables (courses, enrollments, etc.)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS profiles (
                user_id INTEGER PRIMARY KEY,
                full_name TEXT,
                phone TEXT,
                address TEXT,
                profile_pic TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS courses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_code TEXT UNIQUE,
                course_name TEXT,
                description TEXT,
                teacher_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (teacher_id) REFERENCES users (id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS enrollments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                course_id INTEGER,
                enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES users (id),
                FOREIGN KEY (course_id) REFERENCES courses (id),
                UNIQUE(student_id, course_id)
            )
        ''')
        # Password history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS password_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')    
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS materials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_id INTEGER,
                title TEXT,
                file_url TEXT,
                description TEXT,
                uploaded_by INTEGER,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (course_id) REFERENCES courses (id),
                FOREIGN KEY (uploaded_by) REFERENCES users (id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                content TEXT,
                posted_by INTEGER,
                target_role TEXT,
                posted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (posted_by) REFERENCES users (id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS routines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_id INTEGER,
                day_of_week TEXT,
                start_time TEXT,
                end_time TEXT,
                room TEXT,
                FOREIGN KEY (course_id) REFERENCES courses (id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chatbot_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_keyword TEXT UNIQUE,
                answer_student TEXT,
                answer_teacher TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_id INTEGER,
                title TEXT NOT NULL,
                description TEXT,
                due_date TIMESTAMP,
                total_marks INTEGER DEFAULT 100,
                created_by INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (course_id) REFERENCES courses (id),
                FOREIGN KEY (created_by) REFERENCES users (id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assignment_id INTEGER,
                student_id INTEGER,
                submission_text TEXT,
                file_url TEXT,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                grade INTEGER,
                feedback TEXT,
                graded_at TIMESTAMP,
                FOREIGN KEY (assignment_id) REFERENCES assignments (id),
                FOREIGN KEY (student_id) REFERENCES users (id),
                UNIQUE(assignment_id, student_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_id INTEGER,
                title TEXT NOT NULL,
                description TEXT,
                exam_date DATE,
                start_time TIME,
                duration_minutes INTEGER,
                total_marks INTEGER DEFAULT 100,
                created_by INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (course_id) REFERENCES courses (id),
                FOREIGN KEY (created_by) REFERENCES users (id)
            )
        ''')
        try:
            cursor.execute("ALTER TABLE exams ADD COLUMN question_paper_url TEXT")
        except sqlite3.OperationalError:
            pass

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exam_grades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exam_id INTEGER NOT NULL,
                student_id INTEGER NOT NULL,
                marks_obtained INTEGER,
                FOREIGN KEY (exam_id) REFERENCES exams (id),
                FOREIGN KEY (student_id) REFERENCES users (id),
                UNIQUE(exam_id, student_id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                document_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trusted_networks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                network TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_behavior_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                url TEXT NOT NULL,
                method TEXT,
                status_code INTEGER,
                time_spent INTEGER,
                referrer TEXT,
                ip_address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_id INTEGER,
                student_id INTEGER,
                date DATE,
                status TEXT CHECK(status IN ('present', 'absent', 'late')),
                marked_by INTEGER,
                marked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (course_id) REFERENCES courses (id),
                FOREIGN KEY (student_id) REFERENCES users (id),
                FOREIGN KEY (marked_by) REFERENCES users (id),
                UNIQUE(course_id, student_id, date)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trusted_devices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                device_fingerprint TEXT NOT NULL,
                device_label TEXT,
                user_agent TEXT,
                ip_address TEXT,
                is_trusted INTEGER DEFAULT 1,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                UNIQUE(user_id, device_fingerprint)
            )
        ''')

        # Admin audit logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS admin_audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_id INTEGER NOT NULL,
                admin_username TEXT NOT NULL,
                action TEXT NOT NULL,
                target_user_id INTEGER,
                target_username TEXT,
                details TEXT,
                ip_address TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (admin_id) REFERENCES users (id)
    )
''')
        # Notifications table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            link TEXT,
            is_read INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
    )
''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
        cursor.execute("INSERT OR IGNORE INTO system_settings (key, value) VALUES ('ai_security_enabled', '1')")
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exam_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exam_id INTEGER NOT NULL,
                question_text TEXT NOT NULL,
                option_a TEXT NOT NULL,
                option_b TEXT NOT NULL,
                option_c TEXT NOT NULL,
                option_d TEXT NOT NULL,
                correct_option TEXT NOT NULL CHECK(correct_option IN ('A','B','C','D')),
                marks INTEGER DEFAULT 1,
                sort_order INTEGER DEFAULT 0,
                FOREIGN KEY (exam_id) REFERENCES exams (id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exam_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exam_id INTEGER NOT NULL,
                student_id INTEGER NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                submitted_at TIMESTAMP,
                status TEXT DEFAULT 'in_progress' CHECK(status IN ('in_progress','submitted','auto_submitted')),
                score INTEGER,
                FOREIGN KEY (exam_id) REFERENCES exams (id),
                FOREIGN KEY (student_id) REFERENCES users (id),
                UNIQUE(exam_id, student_id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exam_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id INTEGER NOT NULL,
                question_id INTEGER NOT NULL,
                selected_option TEXT,
                is_correct INTEGER DEFAULT 0,
                FOREIGN KEY (attempt_id) REFERENCES exam_attempts (id),
                FOREIGN KEY (question_id) REFERENCES exam_questions (id),
                UNIQUE(attempt_id, question_id)
            )
        ''')
        for col_sql in [
            "ALTER TABLE exams ADD COLUMN is_online INTEGER DEFAULT 0",
        ]:
            try:
                cursor.execute(col_sql)
            except sqlite3.OperationalError:
                pass

        # Default chatbot responses
        default_responses = [
            ('attendance', 'Check attendance in Dashboard. Contact teacher for discrepancies.', 'Upload attendance from course page.'),
            ('exam', 'Exam schedules in Notices.', 'Publish exam schedules from Exams page.'),
            ('assignment', 'Assignments listed under each course.', 'Create assignments from course page.'),
            ('login problem', 'Check credentials or contact admin.', 'Same as students.'),
            ('marks', 'Marks visible on course page after teacher publishes.', 'Upload marks from course management.'),
            ('course', 'Enroll from Courses page.', 'Create courses from Courses page.'),
            ('routine', 'Available under Routine menu.', 'Add routine entries from Routine page.'),
            ('notice', 'Check Notices page.', 'Post notices from Post Notice page.'),
            ('profile', 'Update profile from Profile page.', 'Same as students.'),
            ('password', 'Change password from Change Password page.', 'Same as students.')
        ]
        for kw, ans_st, ans_t in default_responses:
            cursor.execute('INSERT OR IGNORE INTO chatbot_data (question_keyword, answer_student, answer_teacher) VALUES (?, ?, ?)', (kw, ans_st, ans_t))

        # Admin user
        admin = cursor.execute("SELECT * FROM users WHERE username = 'admin'").fetchone()
        if not admin:
            password_hash = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt())
            cursor.execute(
                "INSERT INTO users (username, email, password_hash, role, is_admin, is_approved) VALUES (?, ?, ?, ?, ?, ?)",
                ('admin', 'admin.academics@gmail.com', password_hash, 'teacher', 1, 1)
            )
            admin_id = cursor.lastrowid
            cursor.execute("INSERT OR IGNORE INTO profiles (user_id, full_name) VALUES (?, ?)", (admin_id, 'Administrator'))

        db.commit()
        print("Database initialized with device_fingerprint column.")

# ---------- Helper functions ----------
def classify_device_type(user_agent):
    ua = (user_agent or "").lower()
    if "tablet" in ua or "ipad" in ua:
        return "tablet"
    if "mobi" in ua or "android" in ua or "iphone" in ua:
        return "mobile"
    return "desktop"

def old_rule_based_risk(user_id, ip_address, user_agent, simulation_flags, role):
    risk = 0
    db = get_db()
    cursor = db.cursor()
    user = cursor.execute("SELECT preferred_device FROM users WHERE id = ?", (user_id,)).fetchone()
    preferred_device = (user["preferred_device"] if user else None) or ""
    current_device_type = classify_device_type(user_agent)
    prev_logins = cursor.execute(
        "SELECT DISTINCT ip_address, device FROM login_logs WHERE user_id = ? AND status = 'success'",
        (user_id,)
    ).fetchall()
    if simulation_flags.get('new_ip', False):
        risk += 2
    else:
        if prev_logins and not any(login['ip_address'] == ip_address for login in prev_logins):
            risk += 2
    if simulation_flags.get('new_device', False):
        risk += 2
    else:
        if prev_logins and not any(login['device'] == user_agent for login in prev_logins):
            risk += 2
    if preferred_device and preferred_device in ALLOWED_LOGIN_DEVICES:
        if preferred_device != "other" and preferred_device != current_device_type:
            risk += 2
    current_hour = datetime.now().hour
    if simulation_flags.get('unusual_time', False):
        risk += 1
    else:
        if current_hour < 6 or current_hour > 22:
            risk += 1
    if simulation_flags.get('multiple_failed', False):
        risk += 3
    else:
        one_hour_ago = datetime.now() - timedelta(hours=1)
        failed_count = cursor.execute(
            "SELECT COUNT(*) as count FROM failed_attempts WHERE user_id = ? AND attempt_time > ?",
            (user_id, one_hour_ago)
        ).fetchone()['count']
        if failed_count >= 3:
            risk += 3
    if role == 'teacher':
        risk += 1
    return min(risk, 10)

def extract_ml_features(user_id, ip_address, user_agent):
    db = get_db()
    last = db.execute("""
        SELECT ip_address, device FROM login_logs
        WHERE user_id = ? AND status = 'success'
        ORDER BY login_time DESC LIMIT 1
    """, (user_id,)).fetchone()
    ip_changed = 0
    device_changed = 0
    if last:
        ip_changed = 1 if last['ip_address'] != ip_address else 0
        device_changed = 1 if last['device'] != user_agent else 0
    current_hour = datetime.now().hour
    user = db.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
    role_enc = 1 if user and user['role'] == 'teacher' else 0
    isp_type = 'Other'
    if ip_address.startswith('192.168'):
        isp_type = 'College_WiFi'
    elif ip_address.startswith('202.70'):
        isp_type = 'WorldLink'
    elif ip_address.startswith('103.1'):
        isp_type = 'Ncell'
    elif ip_address.startswith('27.34'):
        isp_type = 'NTC'
    isp_enc = {'College_WiFi':0, 'WorldLink':1, 'Ncell':2, 'NTC':3, 'Other':4}.get(isp_type, 4)
    old_risk = old_rule_based_risk(user_id, ip_address, user_agent, {}, user['role'] if user else 'student')
    return {
        'hour': current_hour,
        'ip_changed': ip_changed,
        'device_changed': device_changed,
        'old_risk_score': old_risk,
        'role_enc': role_enc,
        'isp_enc': isp_enc
    }

def get_top_risk_factors(features_dict, ml_score):
    reasons = []
    if features_dict['isp_enc'] not in [0,1,2]:
        reasons.append("Unusual ISP (not primary)")
    if features_dict['device_changed']:
        reasons.append("New device fingerprint")
    if features_dict['ip_changed']:
        reasons.append("IP address changed")
    hour = features_dict['hour']
    if hour < 6 or hour > 22:
        reasons.append("Anomalous login time")
    if features_dict['old_risk_score'] >= 3:
        reasons.append("Elevated rule‑based risk")
    if ml_score > 0.7:
        reasons.append("High ML anomaly probability")
    return reasons[:3]

def build_ml_feature_frame(user_id, ip_address, user_agent):
    features = extract_ml_features(user_id, ip_address, user_agent)
    X = pd.DataFrame([[
        features['hour'],
        features['ip_changed'],
        features['device_changed'],
        features['old_risk_score'],
        features['role_enc'],
        features['isp_enc']
    ]], columns=['hour', 'ip_changed', 'device_changed', 'old_risk_score', 'role_enc', 'isp_enc'])
    return X, features

def calculate_risk_score(user_id, ip_address, user_agent, simulation_flags, role):
    if ensemble_risk is None:
        rule = old_rule_based_risk(user_id, ip_address, user_agent, simulation_flags, role)
        session.pop('last_shap_explanation', None)
        session['last_risk_explanation'] = get_top_risk_factors(
            extract_ml_features(user_id, ip_address, user_agent), min(1.0, rule / 10.0)
        )
        return min(1.0, rule / 10.0)
    X, features = build_ml_feature_frame(user_id, ip_address, user_agent)
    ml_risk = ensemble_risk.predict_proba(X)[0]
    if SHAP_AVAILABLE:
        shap_data = shap_explain.explain_risk(ensemble_risk, X)
        if shap_data:
            session['last_shap_explanation'] = shap_data
            session['last_risk_explanation'] = shap_data['labels']
        else:
            session.pop('last_shap_explanation', None)
            explanation = get_top_risk_factors(features, ml_risk)
            session['last_risk_explanation'] = explanation
    else:
        session.pop('last_shap_explanation', None)
        explanation = get_top_risk_factors(features, ml_risk)
        session['last_risk_explanation'] = explanation
    return ml_risk

def format_risk_factors_text(shap_payload, geo_labels=None):
    if shap_payload:
        parts = [shap_payload.get('summary', '')]
        geo = geo_labels or shap_payload.get('geo_factors') or []
        if geo:
            parts.append('Geo: ' + ', '.join(geo))
        return ' | '.join(p for p in parts if p)
    if geo_labels:
        return 'Geo: ' + ', '.join(geo_labels)
    return None

def log_attempt(user_id, ip, device, status, risk_score, action, alert_sent=0, risk_factors=None,
                geo_country=None, geo_lat=None, geo_lon=None, shap_json=None):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO login_logs (user_id, ip_address, device, status, risk_score, action, alert_sent,
                                risk_factors, geo_country, geo_lat, geo_lon, shap_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, ip, device, status, risk_score, action, alert_sent, risk_factors,
          geo_country, geo_lat, geo_lon, shap_json))
    db.commit()
    return cursor.lastrowid

def log_activity(user_id, username, action, details, ip_address=None):
    if ip_address is None:
        ip_address = request.remote_addr
    db = get_db()
    db.execute(
        "INSERT INTO user_activity_logs (user_id, username, action, details, ip_address) VALUES (?, ?, ?, ?, ?)",
        (user_id, username, action, details, ip_address)
    )
    db.commit()

def log_admin_action(admin_id, admin_username, action, target_user_id=None, target_username=None, details=None, ip_address=None):
    """Log an admin action for audit trail."""
    db = get_db()
    if ip_address is None:
        ip_address = request.remote_addr
    db.execute('''
        INSERT INTO admin_audit_logs (admin_id, admin_username, action, target_user_id, target_username, details, ip_address)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (admin_id, admin_username, action, target_user_id, target_username, details, ip_address))
    db.commit()

def send_notification(user_id, message, link=None):
    """Insert a notification for a specific user."""
    db = get_db()
    db.execute('''
        INSERT INTO notifications (user_id, message, link)
        VALUES (?, ?, ?)
    ''', (user_id, message, link))
    db.commit()

def send_admin_alert(subject, body):
    """Send admin alert using SendGrid HTTP API."""
    api_key = os.getenv('SENDGRID_API_KEY')
    if not api_key:
        print("SENDGRID_API_KEY not configured.")
        return
    
    from_email = os.getenv('SMTP_USER', 'admin.academics@gmail.com')
    to_email = os.getenv('ALERT_RECIPIENT', from_email)
    
    try:
        sg = sendgrid.SendGridAPIClient(api_key=api_key)
        message = Mail(
            from_email=from_email,
            to_emails=to_email,
            subject=subject,
            html_content=f"<p>{body.replace(chr(10), '<br>')}</p>"
        )
        response = sg.send(message)
        if response.status_code in [202, 200]:
            print(f"✅ Admin alert sent: {subject}")
        else:
            print(f"❌ Admin alert failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Admin alert error: {e}")
    


def send_user_alert(user_email, subject, body):
    """Send email using SendGrid HTTP API (works on Render)."""
    if not user_email:
        return False
    
    api_key = os.getenv('SENDGRID_API_KEY')
    if not api_key:
        print("SENDGRID_API_KEY not configured.")
        return False
    
    from_email = os.getenv('SMTP_USER', 'admin.academics@gmail.com')
    
    try:
        sg = sendgrid.SendGridAPIClient(api_key=api_key)
        
        # Clean the body - handle special characters
        import html
        clean_body = html.escape(body)
        html_body = clean_body.replace('\n', '<br>')
        
        # Wrap in proper HTML with meta charset
        full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
</head>
<body>
<p>{html_body}</p>
</body>
</html>"""
        
        message = Mail(
            from_email=from_email,
            to_emails=user_email,
            subject=subject,
            html_content=full_html
        )
        response = sg.send(message)
        if response.status_code in [202, 200]:
            print(f"✅ Email sent to {user_email}")
            return True
        else:
            print(f"❌ Email failed: {response.status_code} - {response.body}")
            return False
    except Exception as e:
        print(f"❌ Email error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def check_and_alert_new_login(user_id, ip_address, user_agent, geo):
    """Send an alert if the login is from a new device, IP, or country."""
    db = get_db()
    
    # 1. Check trusted devices
    trusted_devices = db.execute(
        "SELECT device_fingerprint FROM trusted_devices WHERE user_id = ? AND is_trusted = 1",
        (user_id,)
    ).fetchall()
    trusted_fingerprints = [row['device_fingerprint'] for row in trusted_devices]
    
    # 2. Get recent successful logins (last 5)
    recent = db.execute(
        "SELECT ip_address, geo_country FROM login_logs "
        "WHERE user_id = ? AND status = 'success' "
        "ORDER BY login_time DESC LIMIT 5",
        (user_id,)
    ).fetchall()
    
    # If no recent logins, this is the first login – we consider it new
    if not recent:
        is_new = True
    else:
        recent_ips = [row['ip_address'] for row in recent]
        recent_countries = [row['geo_country'] for row in recent]
        # Check if device fingerprint is new (and we have one)
        fingerprint = session.get('device_fingerprint')  # we can store it in session during login
        # Actually we don't store fingerprint in session; we need to get it from the login form
        # We'll add device_fingerprint to session in complete_successful_login before this call
        device_fingerprint = session.get('device_fingerprint', None)
        is_new = (
            (device_fingerprint and device_fingerprint not in trusted_fingerprints) or
            (ip_address not in recent_ips) or
            (geo['country'] not in recent_countries)
        )
    
    if not is_new:
        return  # No need to alert
    
    # --- Compose email ---
    user = db.execute("SELECT username, email FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        return
    
    subject = "🔐 New login to your EduShield account"
    body = f"""Dear {user['username']},

We noticed a new login to your EduShield account from a device or location we haven't seen before.

📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🌐 IP Address: {ip_address}
📍 Location: {geo['country']}
📱 Device: {user_agent}

If this was you, you can ignore this email. We recommend that you:
- Check your active sessions in the Security Center: {url_for('security_center', _external=True)}
- If you don't recognise this login, click the link below to logout all other devices:
  {url_for('logout_other_sessions', _external=True)}

Stay secure,
EduShield Team
"""
    # Send the alert
    send_user_alert(user['email'], subject, body)

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

def check_brute_force(user_id):
    db = get_db()
    fifteen_min_ago = datetime.now() - timedelta(minutes=15)
    attempts = db.execute(
        "SELECT COUNT(*) as count FROM failed_attempts WHERE user_id = ? AND attempt_time > ?",
        (user_id, fifteen_min_ago)
    ).fetchone()['count']
    return attempts >= 5

def check_ip_bruteforce(ip_address):
    db = get_db()
    fifteen_min_ago = datetime.now() - timedelta(minutes=15)
    attempts = db.execute(
        "SELECT COUNT(*) as count FROM ip_failed_attempts WHERE ip_address = ? AND attempt_time > ?",
        (ip_address, fifteen_min_ago)
    ).fetchone()['count']
    return attempts >= 15

def get_delay(fail_count):
    """Return the delay in seconds based on the number of failed attempts."""
    if fail_count >= 10:
        return 300   # 5 minutes
    elif fail_count >= 8:
        return 120   # 2 minutes
    elif fail_count >= 5:
        return 30    # 30 seconds
    elif fail_count >= 3:
        return 5     # 5 seconds
    else:
        return 0

def get_user_fail_count(user_id):
    """Get number of failed attempts for a user in the last 15 minutes."""
    db = get_db()
    fifteen_min_ago = datetime.now() - timedelta(minutes=15)
    count = db.execute(
        "SELECT COUNT(*) as cnt FROM failed_attempts WHERE user_id = ? AND attempt_time > ?",
        (user_id, fifteen_min_ago)
    ).fetchone()['cnt']
    return count

def get_ip_fail_count(ip_address):
    """Get number of failed attempts from an IP in the last 15 minutes."""
    db = get_db()
    fifteen_min_ago = datetime.now() - timedelta(minutes=15)
    count = db.execute(
        "SELECT COUNT(*) as cnt FROM ip_failed_attempts WHERE ip_address = ? AND attempt_time > ?",
        (ip_address, fifteen_min_ago)
    ).fetchone()['cnt']
    return count

def log_ip_failure(ip_address):
    if ip_address:
        db = get_db()
        db.execute("INSERT INTO ip_failed_attempts (ip_address) VALUES (?)", (ip_address,))
        db.commit()

def is_device_trusted(user_id, device_fingerprint):
    if not device_fingerprint:
        return False
    db = get_db()
    row = db.execute(
        "SELECT id FROM trusted_devices WHERE user_id = ? AND device_fingerprint = ? AND is_trusted = 1",
        (user_id, device_fingerprint),
    ).fetchone()
    return row is not None

def register_trusted_device(user_id, device_fingerprint, user_agent, ip_address, device_label=None):
    if not device_fingerprint:
        return
    db = get_db()
    label = device_label or classify_device_type(user_agent)
    existing = db.execute(
        "SELECT id FROM trusted_devices WHERE user_id = ? AND device_fingerprint = ?",
        (user_id, device_fingerprint),
    ).fetchone()
    if existing:
        db.execute(
            """UPDATE trusted_devices
               SET last_used = CURRENT_TIMESTAMP, user_agent = ?, ip_address = ?, is_trusted = 1
               WHERE id = ?""",
            (user_agent, ip_address, existing['id']),
        )
    else:
        db.execute(
            """INSERT INTO trusted_devices
               (user_id, device_fingerprint, device_label, user_agent, ip_address, is_trusted)
               VALUES (?, ?, ?, ?, ?, 1)""",
            (user_id, device_fingerprint, label, user_agent, ip_address),
        )
    db.commit()

def _parse_exam_datetime(exam_date, start_time):
    if isinstance(exam_date, str):
        exam_date = datetime.fromisoformat(exam_date).date() if 'T' in exam_date else datetime.strptime(exam_date[:10], '%Y-%m-%d').date()
    if isinstance(start_time, str):
        parts = start_time.split(':')
        start_time = datetime.strptime(f"{parts[0]}:{parts[1]}", '%H:%M').time()
    return datetime.combine(exam_date, start_time)

def get_exam_window(exam):
    start = _parse_exam_datetime(exam['exam_date'], exam['start_time'])
    duration = exam['duration_minutes'] or 60
    end = start + timedelta(minutes=duration)
    now = datetime.now()
    return start, end, start <= now <= end

def calculate_behavior_risk(user_id):
    db = get_db()
    logs = db.execute('''
        SELECT url, method, time_spent, created_at, referrer
        FROM user_behavior_logs
        WHERE user_id = ? AND created_at > datetime('now', '-1 hour')
        ORDER BY created_at DESC
        LIMIT 50
    ''', (user_id,)).fetchall()
    if len(logs) < 5:
        return 0.0
    risk = 0.0
    timestamps = [log['created_at'] for log in logs]
    if len(timestamps) > 1:
        try:
            first = datetime.fromisoformat(timestamps[-1])
            last = datetime.fromisoformat(timestamps[0])
            duration = (last - first).total_seconds()
            if duration > 0:
                req_per_sec = len(timestamps) / duration
                if req_per_sec > 2:
                    risk += 0.4
                elif req_per_sec > 1:
                    risk += 0.2
        except:
            pass
    short_pages = [log for log in logs if log['time_spent'] is not None and log['time_spent'] < 1]
    if len(short_pages) > len(logs) * 0.3:
        risk += 0.3
    admin_pages = ['/admin', '/ids_dashboard', '/admin/attack_logs', '/admin/activity_logs', '/admin/trusted_networks']
    for log in logs:
        if any(log['url'].endswith(page) for page in admin_pages):
            if not log['referrer'] or 'dashboard' not in log['referrer']:
                risk += 0.2
                break
    sensitive_urls = ['/profile', '/change_password', '/profile/documents']
    sensitive_count = sum(1 for log in logs if any(url in log['url'] for url in sensitive_urls))
    if sensitive_count > 10:
        risk += 0.3
    return min(risk, 1.0)

# ---------- Authentication decorators ----------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or not session.get('auth_complete'):
            if session.get('otp_user_id'):
                flash('Please enter the OTP sent to your email to continue.', 'warning')
                return redirect(url_for('verify_otp'))
            flash('Please login to access this page.', 'error')
            return redirect(url_for('login'))
        db = get_db()
        user = db.execute("SELECT session_token, is_locked, is_banned, is_approved FROM users WHERE id = ?", (session['user_id'],)).fetchone()
        if not user:
            session.clear()
            flash('User not found. Please login again.', 'error')
            return redirect(url_for('login'))
        if user['session_token'] != session.get('session_token'):
            session.clear()
            flash('You have been logged out because another session was started.', 'error')
            return redirect(url_for('login'))
        if user['is_locked'] or user['is_banned']:
            session.clear()
            flash('Your account is locked or banned. Contact admin.', 'error')
            return redirect(url_for('login'))
        if session.get('role') == 'teacher' and not user['is_approved']:
            session.clear()
            flash('Your teacher account is pending admin approval.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or not session.get('is_admin'):
            flash('Admin access required.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def behavior_check_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = session.get('user_id')
        if not user_id:
            flash('Please login to access this page.', 'error')
            return redirect(url_for('login'))
        if session.get('reauthenticate_until'):
            try:
                if datetime.now() < datetime.fromisoformat(session['reauthenticate_until']):
                    return f(*args, **kwargs)
            except:
                pass
        risk = calculate_behavior_risk(user_id)
        if risk > 0.5:
            session['reauthenticate_redirect'] = request.url
            flash('Unusual activity detected. Please re‑authenticate to continue.', 'warning')
            return redirect(url_for('reauthenticate'))
        return f(*args, **kwargs)
    return decorated_function

# ---------- CSRF protection ----------
def _get_csrf_token():
    token = session.get('_csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['_csrf_token'] = token
    return token

def _validate_csrf():
    sent = request.form.get('csrf_token', '')
    if not sent and request.is_json:
        sent = request.json.get('csrf_token', '')
    expected = session.get('_csrf_token', '')
    if not sent or not expected or not secrets.compare_digest(sent, expected):
        abort(400, description="CSRF token missing or invalid.")

@app.context_processor
def inject_globals():
    from datetime import datetime
    return {
        'csrf_token': _get_csrf_token,
        'timedelta': timedelta,
        'now': datetime.now()
    }

# ---------- Security headers ----------
@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

def _is_strong_password(pw):
    return bool(
        pw
        and len(pw) >= 8
        and re.search(r'[a-z]', pw)
        and re.search(r'[A-Z]', pw)
        and re.search(r'\d', pw)
        and re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', pw)
    )

# ---------- SQL Injection & XSS Detection ----------
SQLI_PATTERNS = [
    r"('(?:\s|%20)*or(?:\s|%20)+\d+=\d+)",
    r"(\bor\b\s+1=1\b)",
    r"(\bunion\b\s+(\ball\b\s+)?\bselect\b)",
    r"(--|#)",
    r"(\bselect\b.+\bfrom\b)",
    r"(\binsert\b.+\binto\b)",
    r"(\bupdate\b.+\bset\b)",
    r"(\bdelete\b.+\bfrom\b)",
    r"(\bdrop\b\s+\btable\b)",
]
XSS_PATTERNS = [
    r"<\s*script\b",
    r"javascript\s*:",
    r"on\w+\s*=",
    r"<\s*iframe\b",
    r"<\s*img\b[^>]*\bonerror\b",
]

def scan_and_log_attack_patterns(endpoint=None, block=False, extra_sources=None, lock_user=True):
    sources = []
    for key, value in (request.form or {}).items():
        sources.append((key, value))
    for key, value in (request.args or {}).items():
        sources.append((key, value))
    if request.is_json:
        try:
            payload = request.get_json(silent=True) or {}
            if isinstance(payload, dict):
                for k, v in payload.items():
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        sources.append((str(k), "" if v is None else str(v)))
        except Exception:
            pass
    if extra_sources:
        sources.extend(extra_sources.items())
    if not sources:
        return []
    ip = request.remote_addr
    user_id = session.get("user_id")
    ep = endpoint or (request.endpoint or "unknown")
    method = request.method
    db = get_db()
    cur = db.cursor()
    hits = []
    for key, value in sources:
        if not isinstance(value, str):
            continue
        val = value.strip()
        if not val:
            continue
        for pattern in SQLI_PATTERNS:
            if re.search(pattern, val, re.IGNORECASE):
                cur.execute(
                    """INSERT INTO attack_patterns
                       (user_id, ip_address, endpoint, method, param_key, param_value, pattern_type, pattern, severity, blocked)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (user_id, ip, ep, method, key, val[:500], "SQLi", pattern[:200], 3, 1 if block else 0)
                )
                hits.append({"key": key, "type": "SQLi", "pattern": pattern})
                db.execute("INSERT INTO ip_failed_attempts (ip_address) VALUES (?)", (ip,))
                db.commit()
                if lock_user and user_id:
                    user = db.execute("SELECT is_admin FROM users WHERE id = ?", (user_id,)).fetchone()
                    if user and not user['is_admin']:
                        db.execute("UPDATE users SET is_locked = 1, locked_at = CURRENT_TIMESTAMP WHERE id = ?", (user_id,))
                        db.commit()
                        send_admin_alert("Account locked due to SQLi attempt", f"User: {session.get('username', 'Unknown')}\nIP: {ip}\nEndpoint: {ep}\nPayload: {val[:200]}")
                        user_email = db.execute("SELECT email FROM users WHERE id = ?", (user_id,)).fetchone()['email']
                        send_user_alert(user_email, "Your account has been locked", "Your account has been locked due to a detected security threat. Please contact the administrator to unlock it.")
                        session.clear()
                        flash("Your account has been locked due to a detected attack attempt. Contact admin.", "error")
                        abort(403, "Account locked due to security violation.")
                if block:
                    abort(400, "Potential SQL injection detected – request blocked.")
        for pattern in XSS_PATTERNS:
            if re.search(pattern, val, re.IGNORECASE):
                cur.execute(
                    """INSERT INTO attack_patterns
                       (user_id, ip_address, endpoint, method, param_key, param_value, pattern_type, pattern, severity, blocked)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (user_id, ip, ep, method, key, val[:500], "XSS", pattern[:200], 2, 1 if block else 0)
                )
                hits.append({"key": key, "type": "XSS", "pattern": pattern})
                db.commit()
                if lock_user and user_id:
                    user = db.execute("SELECT is_admin FROM users WHERE id = ?", (user_id,)).fetchone()
                    if user and not user['is_admin']:
                        db.execute("UPDATE users SET is_locked = 1, locked_at = CURRENT_TIMESTAMP WHERE id = ?", (user_id,))
                        db.commit()
                        send_admin_alert("Account locked due to XSS attempt", f"User: {session.get('username', 'Unknown')}\nIP: {ip}\nEndpoint: {ep}\nPayload: {val[:200]}")
                        user_email = db.execute("SELECT email FROM users WHERE id = ?", (user_id,)).fetchone()['email']
                        send_user_alert(user_email, "Your account has been locked", "Your account has been locked because a security violation (XSS attempt) was detected. Please contact the administrator to unlock it.")
                        session.clear()
                        flash("Your account has been locked due to a detected attack attempt. Contact admin.", "error")
                        abort(403, "Account locked due to security violation.")
                if block:
                    abort(400, "XSS payload detected – request blocked.")
    db.commit()
    return hits

# ---------- Geolocation ----------
_geo_cache = {}
def get_geolocation(ip):
    if ip.startswith('127.') or ip.startswith('192.168') or ip == '::1':
        return {'country': 'Nepal', 'lat': 27.7172, 'lon': 85.3240}
    now = datetime.now()
    for cached_ip in list(_geo_cache.keys()):
        if _geo_cache[cached_ip]['expires'] < now:
            del _geo_cache[cached_ip]
    if ip in _geo_cache:
        return _geo_cache[ip]['data']
    try:
        response = requests.get(f"http://ip-api.com/json/{ip}?fields=status,country,lat,lon", timeout=3)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                result = {
                    'country': data['country'],
                    'lat': data.get('lat', 27.7172),
                    'lon': data.get('lon', 85.3240)
                }
                _geo_cache[ip] = {'data': result, 'expires': now + timedelta(hours=1)}
                return result
    except Exception as e:
        print(f"Geolocation API error: {e}")
    return {'country': 'Nepal', 'lat': 27.7172, 'lon': 85.3240}

def get_ai_security_status():
    db = get_db()
    row = db.execute("SELECT value FROM system_settings WHERE key = 'ai_security_enabled'").fetchone()
    return row['value'] == '1' if row else True

def set_ai_security_status(enabled):
    db = get_db()
    db.execute("INSERT OR REPLACE INTO system_settings (key, value) VALUES ('ai_security_enabled', ?)",
               (str(1 if enabled else 0),))
    db.commit()

def get_ai_metrics():
    db = get_db()
    threats = db.execute("SELECT COUNT(*) as cnt FROM login_logs WHERE status = 'blocked' AND login_time > datetime('now', '-1 day')").fetchone()['cnt']
    attacks = db.execute("SELECT COUNT(*) as cnt FROM attack_patterns WHERE event_time > datetime('now', '-1 day')").fetchone()['cnt']
    threats_detected = threats + attacks
    risk_events = db.execute("SELECT COUNT(*) as cnt FROM login_logs WHERE risk_score > 0 AND login_time > datetime('now', '-1 day')").fetchone()['cnt']
    suspicious = db.execute("SELECT COUNT(*) as cnt FROM login_logs WHERE (risk_score > 50 OR action = 'otp_required') AND login_time > datetime('now', '-1 day')").fetchone()['cnt']
    return {
        'threats_detected': threats_detected,
        'risk_events': risk_events,
        'suspicious_logins': suspicious
    }

def is_ip_trusted(ip):
    db = get_db()
    networks = db.execute("SELECT network FROM trusted_networks").fetchall()
    if not networks:
        return False
    try:
        ip_obj = ipaddress.ip_address(ip)
    except ValueError:
        return False
    for row in networks:
        try:
            net = ipaddress.ip_network(row['network'], strict=False)
            if ip_obj in net:
                return True
        except ValueError:
            continue
    return False

def apply_geofence_risk(ip_address, user_role, risk_score):
    if is_ip_trusted(ip_address):
        adjusted = max(0.0, risk_score - 0.4)
        explanations = ["Trusted college network - risk reduced"]
        return adjusted, explanations
    geo = get_geolocation(ip_address)
    adjusted = risk_score
    explanations = []
    if geo['country'] != 'Nepal':
        adjusted = min(1.0, adjusted + 0.3)
        explanations.append(f"Login from {geo['country']} (outside Nepal) – high risk")
    else:
        lat, lon = geo['lat'], geo['lon']
        if not (27.65 <= lat <= 27.75 and 85.25 <= lon <= 85.45):
            adjusted = min(1.0, adjusted + 0.15)
            explanations.append("Login from outside Kathmandu Valley – moderate risk")
        else:
            explanations.append("Geolocation matches Nepal/Kathmandu Valley")
    if user_role == 'teacher' and adjusted >= 0.3:
        adjusted = min(1.0, adjusted + 0.1)
        explanations.append("Teacher account – additional verification recommended")
    return adjusted, explanations

# ---------- Adaptive authentication & threat intelligence ----------
RISK_ALLOW_MAX = 0.40   # low risk: allow directly
RISK_OTP_MAX = 0.70     # medium: OTP; high (>70%): block

def decide_auth_action(risk_score):
    """Unified risk-based auth for all roles: low→allow, medium→OTP, high→block."""
    if risk_score > RISK_OTP_MAX:
        return 'block'
    if risk_score > RISK_ALLOW_MAX:
        return 'otp'
    return 'allow'

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two coordinates."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))

def check_impossible_travel(user_id, current_lat, current_lon):
    """Flag logins >500 km from last login within <2 hours."""
    db = get_db()
    last = db.execute("""
        SELECT geo_lat, geo_lon, login_time FROM login_logs
        WHERE user_id = ? AND status IN ('success', 'otp_sent')
          AND geo_lat IS NOT NULL AND geo_lon IS NOT NULL
        ORDER BY login_time DESC LIMIT 1
    """, (user_id,)).fetchone()
    if not last:
        return 0.0, None
    login_time = last['login_time']
    if isinstance(login_time, str):
        login_time = datetime.fromisoformat(login_time)
    elapsed_hours = (datetime.now() - login_time).total_seconds() / 3600
    if elapsed_hours <= 0:
        return 0.0, None
    dist_km = haversine_km(last['geo_lat'], last['geo_lon'], current_lat, current_lon)
    if dist_km > 500 and elapsed_hours < 2:
        msg = f"Impossible travel: {dist_km:.0f} km in {elapsed_hours * 60:.0f} min"
        return 0.30, msg
    return 0.0, None

_abuse_cache = {}

def check_abuseipdb(ip_address):
    """Query AbuseIPDB for IP reputation. Returns (risk_boost 0–0.3, info_dict)."""
    if ip_address.startswith('127.') or ip_address.startswith('192.168') or ip_address == '::1':
        return 0.0, None
    api_key = os.getenv('ABUSEIPDB_API_KEY')
    if not api_key:
        return 0.0, None
    now = datetime.now()
    for cached_ip in list(_abuse_cache.keys()):
        if _abuse_cache[cached_ip]['expires'] < now:
            del _abuse_cache[cached_ip]
    if ip_address in _abuse_cache:
        entry = _abuse_cache[ip_address]
        return entry['risk_boost'], entry['info']
    try:
        response = requests.get(
            'https://api.abuseipdb.com/api/v2/check',
            headers={'Key': api_key, 'Accept': 'application/json'},
            params={'ipAddress': ip_address, 'maxAgeInDays': 90},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json().get('data', {})
            score = data.get('abuseConfidenceScore', 0)
            risk_boost = min(0.30, (score / 100.0) * 0.30)
            info = {
                'abuse_score': score,
                'country': data.get('countryCode'),
                'total_reports': data.get('totalReports', 0),
                'is_tor': data.get('isTor', False),
            }
            _abuse_cache[ip_address] = {
                'risk_boost': risk_boost,
                'info': info,
                'expires': now + timedelta(hours=24),
            }
            return risk_boost, info
    except Exception as e:
        print(f"AbuseIPDB error: {e}")
    return 0.0, None

def evaluate_full_risk(user_id, ip_address, user_agent, simulation_flags, role, risk_extra=0.0):
    """ML risk + impossible travel + AbuseIPDB + geofence for any role."""
    geo = get_geolocation(ip_address)
    risk_score = calculate_risk_score(user_id, ip_address, user_agent, simulation_flags, role)
    risk_score = min(1.0, risk_score + risk_extra)
    shap_raw = session.pop('last_shap_explanation', None)
    session.pop('last_risk_explanation', None)
    extra_factors = []
    travel_boost, travel_msg = check_impossible_travel(user_id, geo['lat'], geo['lon'])
    if travel_boost:
        risk_score = min(1.0, risk_score + travel_boost)
        extra_factors.append(travel_msg)
    abuse_boost, abuse_info = check_abuseipdb(ip_address)
    if abuse_boost:
        risk_score = min(1.0, risk_score + abuse_boost)
        if abuse_info:
            extra_factors.append(
                f"AbuseIPDB reputation {abuse_info['abuse_score']}% "
                f"({abuse_info['total_reports']} reports)"
            )
    risk_score, geo_explanations = apply_geofence_risk(ip_address, role, risk_score)
    all_geo = geo_explanations + extra_factors
    shap_payload = shap_explain.merge_geo_shap(shap_raw, all_geo) if SHAP_AVAILABLE else None
    explanation_str = format_risk_factors_text(shap_payload, all_geo)
    shap_json_str = json.dumps(shap_payload) if shap_payload else None
    return risk_score, geo, shap_payload, all_geo, abuse_info, explanation_str, shap_json_str

def _build_display_reasons(shap_payload, geo_explanations, fingerprint_changed=False):
    display_reasons = shap_payload.get('summary', '') if shap_payload else ''
    if geo_explanations:
        display_reasons = (display_reasons + ' | ' if display_reasons else '') + ', '.join(geo_explanations)
    if fingerprint_changed:
        display_reasons = (display_reasons + ' | ' if display_reasons else '') + 'New device fingerprint'
    return display_reasons

def complete_successful_login(user, ip_address, user_agent, risk_percent, action,
                              explanation_str=None, geo=None, shap_json_str=None, activity_detail=None,
                              redirect_to='dashboard', device_fingerprint=None, trust_device=False):
    db = get_db()
    db.execute("DELETE FROM failed_attempts WHERE user_id = ?", (user['id'],))
    db.commit()
    fp = device_fingerprint or session.pop('pending_device_fingerprint', None)
    if trust_device or session.pop('pending_trust_device', False):
        register_trusted_device(user['id'], fp, user_agent, ip_address)
    if fp:
        db.execute("UPDATE users SET device_fingerprint = ? WHERE id = ?", (fp, user['id']))
        db.commit()
    session_token = secrets.token_urlsafe(32)
    db.execute("UPDATE users SET session_token = ? WHERE id = ?", (session_token, user['id']))
    db.commit()
    session.clear()
    session.permanent = True
    session['user_id'] = user['id']
    session['username'] = user['username']
    session['role'] = user['role']
    session['is_admin'] = user['is_admin']
    session['last_activity'] = datetime.now().isoformat()
    session['session_token'] = session_token
    session['auth_complete'] = True
    # Store device fingerprint for the alert check
    session['device_fingerprint'] = device_fingerprint
    geo = geo or get_geolocation(ip_address)
    log_attempt(user['id'], ip_address, user_agent, 'success', risk_percent, action,
                risk_factors=explanation_str, geo_country=geo['country'],
                geo_lat=geo['lat'], geo_lon=geo['lon'], shap_json=shap_json_str)
    
    # ========== ADD THIS BLOCK ==========
    # Send alert if this is a new device/IP/country
    check_and_alert_new_login(user['id'], ip_address, user_agent, geo)
    # ===================================
    
    detail = activity_detail or f"Login successful from {ip_address}"
    log_activity(user['id'], user['username'], 'login', detail)
    if redirect_to == 'dashboard':
        flash('Login successful!', 'success')
    return redirect(url_for(redirect_to))

def initiate_otp_verification(user, ip_address, user_agent, risk_score, explanation_str,
                              shap_json_str=None, geo=None, recovery=False, admin_login=False,
                              device_fingerprint=None, trust_device=False):
    db = get_db()
    
    # --- FIX: Ensure geo is never None ---
    if geo is None:
        geo = get_geolocation(ip_address)
    
    # 1. Get or generate TOTP secret
    user_record = db.execute("SELECT totp_secret, totp_enabled, username FROM users WHERE id = ?", (user['id'],)).fetchone()
    secret = user_record['totp_secret']
    if not secret:
        secret = pyotp.random_base32()
        db.execute("UPDATE users SET totp_secret = ? WHERE id = ?", (secret, user['id']))
        db.commit()
        enabled = False
    else:
        enabled = user_record['totp_enabled'] == 1

    # 2. Store data in session (no numeric OTP anymore)
    session.clear()
    session.permanent = True
    session['pending_device_fingerprint'] = device_fingerprint
    session['pending_trust_device'] = trust_device
    session['totp_user_id'] = user['id']
    session['totp_risk_score'] = risk_score
    session['totp_ip'] = ip_address
    session['totp_user_agent'] = user_agent
    session['totp_explanation'] = explanation_str
    session['totp_shap_json'] = shap_json_str
    session['totp_geo'] = geo          # now geo is guaranteed to be a dict
    session['totp_recovery'] = recovery
    session['totp_admin_login'] = admin_login
    session['totp_enabled'] = enabled
    session['totp_secret'] = secret

    # 3. Generate provisioning URI (for QR code)
    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(user_record['username'], issuer_name="EduShield")
    session['totp_provisioning_uri'] = provisioning_uri

    # Log the event (status 'otp_sent' remains for compatibility)
    log_attempt(user['id'], ip_address, user_agent, 'otp_sent', int(risk_score * 100),
                'totp_required', risk_factors=explanation_str,
                geo_country=geo['country'], geo_lat=geo['lat'], geo_lon=geo['lon'],
                shap_json=shap_json_str)

    flash('Enter the 6‑digit code from your authenticator app.', 'info')
    return redirect(url_for('verify_otp'))

# ---------- Request logging middleware ----------
@app.before_request
def before_request_logging():
    if request.endpoint in ('static', 'login', 'logout', 'verify_otp', 'verify_teacher_otp',
                            'security_questions', 'forgot_password', 'reset_password',
                            'verify_security_questions', 'reauthenticate'):
        return
    if 'user_id' not in session:
        return
    request.start_time = time.time()
    g.behavior_log_id = None
    db = get_db()
    try:
        cursor = db.execute('''
            INSERT INTO user_behavior_logs (user_id, url, method, status_code, time_spent, referrer, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session['user_id'], request.url, request.method, 0, 0, request.referrer, request.remote_addr))
        g.behavior_log_id = cursor.lastrowid
        db.commit()
    except Exception as e:
        print(f"Behavior log insert error: {e}")

@app.after_request
def after_request_update_time(response):
    if hasattr(request, 'start_time') and hasattr(g, 'behavior_log_id') and g.behavior_log_id:
        elapsed_ms = int((time.time() - request.start_time) * 1000)
        db = get_db()
        try:
            db.execute('UPDATE user_behavior_logs SET time_spent = ?, status_code = ? WHERE id = ?',
                       (elapsed_ms, response.status_code, g.behavior_log_id))
            db.commit()
        except Exception as e:
            print(f"Behavior log update error: {e}")
    return response

# ---------- Routes ----------
@app.route('/')
def index():
    if session.get('user_id') and session.get('auth_complete'):
        return redirect(url_for('dashboard'))
    return render_template('landing.html', now=datetime.now())

@app.route('/register', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def register():
    if request.method == 'GET':
        # Clear any leftover session data (include reg_username)
        for key in ['reg_full_name', 'reg_username', 'reg_email', 'reg_role', 'reg_password_hash', 'reg_device']:
            session.pop(key, None)
        return render_template('register.html')   # Step 1

    # POST handling
    _validate_csrf()
    scan_and_log_attack_patterns(endpoint="register", block=False)

    step = request.form.get('step')
    if step == '1' or step is None:
        # Process step 1 (account details)
        full_name = request.form.get('full_name', '').strip()

        # --- NEW: username field validation ---
        username = request.form.get('username', '').strip()
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_.]{2,29}$', username):
            flash('Username must be 3-30 characters, start with a letter, and contain only letters, numbers, underscores, or dots.', 'error')
            return redirect(url_for('register'))
        db = get_db()
        existing = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            flash('Username already taken. Please choose another.', 'error')
            return redirect(url_for('register'))
        session['reg_username'] = username

        email = request.form.get('email', '').strip().lower()
        role = request.form.get('role', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')

        if not all([full_name, email, role, password, confirm]):
            flash('All fields are required.', 'error')
            return redirect(url_for('register'))
        if password != confirm:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('register'))
        if not _is_strong_password(password):
            flash('Password must be at least 8 characters with uppercase, lowercase, a number, and a special character.', 'error')
            return redirect(url_for('register'))

        # Store data in session
        session['reg_full_name'] = full_name
        session['reg_email'] = email
        session['reg_role'] = role
        session['reg_password_hash'] = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        session['reg_device'] = request.form.get('preferred_device', 'laptop')
        session['reg_fingerprint'] = request.form.get('device_fingerprint', '')

        if role == 'student':
            return render_template('security_questions.html')   # Step 2
        else:
            # Teacher: skip security questions, go directly to final registration
            return finalize_registration(bypass_questions=True)

    elif step == '2':
        # Process security questions (student only)
        return finalize_registration(bypass_questions=False)

    else:
        flash('Invalid registration step.', 'error')
        return redirect(url_for('register'))
    
def finalize_registration(bypass_questions=False):
    full_name = session.get('reg_full_name')
    email = session.get('reg_email')
    role = session.get('reg_role')
    preferred_device = session.get('reg_device', 'laptop')
    password_hash = session.get('reg_password_hash')
    device_fingerprint = session.get('reg_fingerprint', '')
    username = session.get('reg_username')

    if not all([full_name, email, role, password_hash, username]):
        flash('Registration session expired. Please start over.', 'error')
        return redirect(url_for('register'))

    sec_q1 = sec_a1 = sec_q2 = sec_a2 = sec_q3 = sec_a3 = None
    if role == 'student' and not bypass_questions:
        sec_q1 = request.form.get('sec_q1', '').strip()
        sec_a1 = request.form.get('sec_a1', '').strip().lower()
        sec_q2 = request.form.get('sec_q2', '').strip()
        sec_a2 = request.form.get('sec_a2', '').strip().lower()
        sec_q3 = request.form.get('sec_q3', '').strip()
        sec_a3 = request.form.get('sec_a3', '').strip().lower()
        if not all([sec_q1, sec_a1, sec_q2, sec_a2, sec_q3, sec_a3]):
            flash('All security questions are required.', 'error')
            return redirect(url_for('register'))

    # Check username uniqueness
    db = get_db()
    existing = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        flash('Username already taken. Please choose another.', 'error')
        return redirect(url_for('register'))

    # Hash answers
    a1_hash = bcrypt.hashpw(sec_a1.encode('utf-8'), bcrypt.gensalt()) if sec_a1 else None
    a2_hash = bcrypt.hashpw(sec_a2.encode('utf-8'), bcrypt.gensalt()) if sec_a2 else None
    a3_hash = bcrypt.hashpw(sec_a3.encode('utf-8'), bcrypt.gensalt()) if sec_a3 else None

    is_approved = 1 if role == 'student' else 0

    try:
        db.execute('''
            INSERT INTO users
            (username, email, password_hash, role, preferred_device,
             security_q1, security_a1, security_q2, security_a2, security_q3, security_a3,
             is_approved, device_fingerprint)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (username, email, password_hash, role, preferred_device,
              sec_q1, a1_hash, sec_q2, a2_hash, sec_q3, a3_hash,
              is_approved, device_fingerprint,))
        user_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        db.execute("INSERT OR IGNORE INTO profiles (user_id, full_name) VALUES (?, ?)", (user_id, full_name))
        db.commit()
        log_activity(user_id, username, 'account_registered', f"Registered as {role}")

        # --- SEND VERIFICATION EMAIL ---
        token = serializer.dumps(email, salt='email-verify-salt')
        verify_link = url_for('verify_email', token=token, _external=True)
        subject = "Verify your email – EduShield"
        body = f"""Dear {full_name or username},

Thank you for registering on EduShield.

Please verify your email address by clicking the link below (valid for 24 hours):

{verify_link}

If you did not register on EduShield, please ignore this email.

Stay secure,
EduShield Team
"""
        send_user_alert(email, subject, body)

        # Clear session data
        for key in ['reg_full_name', 'reg_username', 'reg_email', 'reg_role', 'reg_password_hash', 'reg_device', 'reg_fingerprint']:
            session.pop(key, None)

        if role == 'teacher':
            send_admin_alert("New Teacher Registration Pending Approval",
                             f"Username: {username}\nEmail: {email}\nFull Name: {full_name}")
            flash('Registration successful! Your teacher account requires admin approval. Please also verify your email before you can log in.', 'info')
        else:
            flash('Registration successful! A verification link has been sent to your email. Please verify your account before logging in.', 'success')
        return redirect(url_for('login'))

    except sqlite3.IntegrityError:
        flash('Username or email already exists.', 'error')
        return redirect(url_for('register'))
    
@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login():
    if request.method == 'POST':
        _validate_csrf()
        scan_and_log_attack_patterns(endpoint="login", block=False)
        
        identifier = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        ip_address = request.remote_addr
        user_agent = request.headers.get('User-Agent', 'Unknown')
        device_fingerprint = request.form.get('device_fingerprint', '')
        
        # ---------- Progressive Delay Logic ----------
        ip_fail_count = get_ip_fail_count(ip_address)
        delay = get_delay(ip_fail_count)
        
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE email = ? OR username = ?", (identifier, identifier)).fetchone()
        
        if user:
            user_fail_count = get_user_fail_count(user['id'])
            delay = max(delay, get_delay(user_fail_count))
            
            # Lock account if 15+ failures in 15 minutes
            if user_fail_count >= 15:
                db.execute("UPDATE users SET is_locked = 1, locked_at = CURRENT_TIMESTAMP WHERE id = ?", (user['id'],))
                db.commit()
                flash('Too many failed attempts. Your account has been locked for security. Contact admin.', 'error')
                return redirect(url_for('login'))
        else:
            user_fail_count = 0
        
        if delay > 0:
            import time
            flash(f'Too many failed attempts. Please wait {delay} seconds before trying again.', 'warning')
            time.sleep(delay)
            # Re‑fetch user in case something changed during sleep (unlikely but safe)
            if user:
                user = db.execute("SELECT * FROM users WHERE id = ?", (user['id'],)).fetchone()
        
        # ---------- Continue with login checks ----------
        if not user:
            log_ip_failure(ip_address)
            log_attempt(None, ip_address, user_agent, 'failed', 0, 'unknown_user')
            flash('Invalid email/username or password.', 'error')
            return redirect(url_for('login'))
        
        # Email verification
        if not user['is_verified']:
            flash('Please verify your email before logging in. Check your inbox for the verification link.', 'error')
            return redirect(url_for('login'))
        
        if user['is_banned']:
            flash('Your account has been blocked by an administrator. Contact support.', 'error')
            return redirect(url_for('login'))
        
        if user['is_locked']:
            flash('Your account is locked due to multiple security failures. Contact admin.', 'error')
            return redirect(url_for('login'))
        
        if user['role'] == 'teacher' and not user['is_admin'] and not user['is_approved']:
            flash('Your teacher account is pending admin approval. You will be notified once approved.', 'warning')
            return redirect(url_for('login'))
        
        # Security Questions flow for students after 3 failures (only if not already handled by delay)
        if user_fail_count >= 3 and user['role'] == 'student':
            session['sec_user_id'] = user['id']
            session['sec_questions'] = [user['security_q1'], user['security_q2'], user['security_q3']]
            session['sec_answers'] = [user['security_a1'], user['security_a2'], user['security_a3']]
            flash('Too many failed attempts. Please answer security questions.', 'warning')
            return redirect(url_for('security_questions'))
        
        # Password check
        if not bcrypt.checkpw(password.encode('utf-8'), user['password_hash']):
            log_ip_failure(ip_address)
            db.execute("INSERT INTO failed_attempts (user_id) VALUES (?)", (user['id'],))
            db.commit()
            log_attempt(user['id'], ip_address, user_agent, 'failed', 0, 'none')
            flash('Invalid email/username or password.', 'error')
            return redirect(url_for('login'))
        
        # ---------- Successful login path ----------
        trust_device = 'trust_device' in request.form
        
        # Device fingerprint check
        fingerprint_changed = False
        risk_extra = 0.0
        stored_fp = user['device_fingerprint'] if 'device_fingerprint' in user.keys() else None
        try:
            if is_device_trusted(user['id'], device_fingerprint):
                fingerprint_changed = False
                risk_extra = 0.0
            elif stored_fp and device_fingerprint and stored_fp != device_fingerprint:
                fingerprint_changed = True
                risk_extra = 0.3
                flash('New device detected. Login risk increased.', 'warning')
        except Exception as e:
            print(f"Fingerprint error: {e}")
        
        # Simulation flags (for testing)
        simulation_flags = {
            'new_ip': 'simulate_new_ip' in request.form,
            'new_device': 'simulate_new_device' in request.form,
            'unusual_time': 'simulate_unusual_time' in request.form,
            'multiple_failed': 'simulate_multiple_failed' in request.form
        }
        
        # Risk evaluation
        risk_score, geo, shap_payload, geo_explanations, abuse_info, explanation_str, shap_json_str = (
            evaluate_full_risk(user['id'], ip_address, user_agent, simulation_flags, user['role'], risk_extra)
        )
        risk_percent = int(risk_score * 100)
        display_reasons = _build_display_reasons(shap_payload, geo_explanations, fingerprint_changed)
        auth_action = decide_auth_action(risk_score)
        
        # Admin accounts: always require OTP
        if user['is_admin']:
            return initiate_otp_verification(
                user, ip_address, user_agent, risk_score, explanation_str, shap_json_str, geo,
                admin_login=True,
                device_fingerprint=device_fingerprint, trust_device=trust_device,
            )
        
        if auth_action == 'block':
            log_attempt(user['id'], ip_address, user_agent, 'blocked', risk_percent, 'block',
                        risk_factors=explanation_str, geo_country=geo['country'], geo_lat=geo['lat'],
                        geo_lon=geo['lon'], shap_json=shap_json_str)
            send_admin_alert(
                f"High-risk login blocked ({user['role']})",
                f"User: {user['username']}\nIP: {ip_address}\nRisk: {risk_percent}%\n"
                f"Fingerprint changed: {fingerprint_changed}\nFactors: {explanation_str}"
            )
            flash(f'Login blocked due to high risk ({risk_percent}%). {display_reasons}', 'error')
            return redirect(url_for('login'))
        
        if auth_action == 'otp':
            return initiate_otp_verification(
                user, ip_address, user_agent, risk_score, explanation_str, shap_json_str, geo,
                device_fingerprint=device_fingerprint, trust_device=trust_device,
            )
        
        # Allow login
        return complete_successful_login(
            user, ip_address, user_agent, risk_percent, 'allow',
            explanation_str=explanation_str, geo=geo, shap_json_str=shap_json_str,
            activity_detail=f"Login successful from {ip_address} (fingerprint changed: {fingerprint_changed})",
            device_fingerprint=device_fingerprint, trust_device=trust_device,
        )
    
    return render_template('login.html')

@app.route('/security_questions', methods=['GET', 'POST'])
def security_questions():
    if 'sec_user_id' not in session:
        flash('No verification in progress.', 'error')
        return redirect(url_for('login'))
    user_id = session['sec_user_id']
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if not user:
        session.pop('sec_user_id', None)
        flash('User not found.', 'error')
        return redirect(url_for('login'))
    if user['role'] == 'teacher':
        return initiate_otp_verification(
            user, request.remote_addr, request.headers.get('User-Agent', 'Unknown'),
            0.5, 'Security-question recovery OTP', recovery=True
        )
    if request.method == 'POST':
        _validate_csrf()
        answers = [
            request.form.get('answer_1', '').strip().lower(),
            request.form.get('answer_2', '').strip().lower(),
            request.form.get('answer_3', '').strip().lower()
        ]
        correct = 0
        if answers[0] and bcrypt.checkpw(answers[0].encode('utf-8'), session['sec_answers'][0]):
            correct += 1
        if answers[1] and bcrypt.checkpw(answers[1].encode('utf-8'), session['sec_answers'][1]):
            correct += 1
        if answers[2] and bcrypt.checkpw(answers[2].encode('utf-8'), session['sec_answers'][2]):
            correct += 1
        if correct >= 2:
            db.execute("DELETE FROM failed_attempts WHERE user_id=?", (user_id,))
            db.commit()
            session.pop('sec_user_id', None)
            session.pop('sec_questions', None)
            session.pop('sec_answers', None)
            flash('Verification successful. Please login again.', 'success')
            return redirect(url_for('login'))
        else:
            db.execute("UPDATE users SET is_locked=1, locked_at=CURRENT_TIMESTAMP WHERE id=?", (user_id,))
            db.commit()
            send_admin_alert("Account locked due to failed security questions", f"User: {user['username']}\nIP: {request.remote_addr}\nFailed to answer enough security questions.")
            send_user_alert(user['email'], "Your account has been locked", "Your account has been locked because you failed to answer your security questions correctly. Please contact the administrator to unlock it.")
            session.clear()
            flash('Security verification failed. Your account has been locked. Contact admin.', 'error')
            return redirect(url_for('login'))
    return render_template('security_questions.html', questions=session.get('sec_questions', ['','','']))

@app.route('/verify_email/<token>')
def verify_email(token):
    try:
        email = serializer.loads(token, salt='email-verify-salt', max_age=86400)  # 24 hours
    except (SignatureExpired, BadSignature):
        flash('The verification link is invalid or has expired. Please request a new one.', 'error')
        return redirect(url_for('login'))
    
    db = get_db()
    user = db.execute("SELECT id, is_verified FROM users WHERE email = ?", (email,)).fetchone()
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('login'))
    
    if user['is_verified']:
        flash('Your email is already verified. You can now log in.', 'info')
        return redirect(url_for('login'))
    
    db.execute("UPDATE users SET is_verified = 1 WHERE id = ?", (user['id'],))
    db.commit()
    log_activity(user['id'], email, 'email_verified', "Email address verified")
    
    flash('Your email has been verified successfully! You can now log in.', 'success')
    return redirect(url_for('login'))

@app.route('/resend_verification', methods=['GET', 'POST'])
def resend_verification():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        if not email:
            flash('Please enter your email address.', 'error')
            return redirect(url_for('resend_verification'))
        
        db = get_db()
        user = db.execute("SELECT id, username, email, is_verified FROM users WHERE email = ?", (email,)).fetchone()
        if not user:
            flash('No account found with that email address.', 'error')
            return redirect(url_for('resend_verification'))
        if user['is_verified']:
            flash('Your email is already verified. Please log in.', 'info')
            return redirect(url_for('login'))
        
        # Generate new token
        token = serializer.dumps(email, salt='email-verify-salt')
        verify_link = url_for('verify_email', token=token, _external=True)
        subject = "Verify your email – EduShield"
        body = f"Dear {user['username']},\n\nPlease verify your email by clicking the link below (valid for 24 hours):\n\n{verify_link}\n\nIf you did not request this, ignore this email."
        send_user_alert(email, subject, body)
        flash('A new verification link has been sent to your email address.', 'success')
        return redirect(url_for('login'))
    
    return render_template('resend_verification.html')


@app.route('/verify_otp', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def verify_otp():
    import json

    if 'totp_user_id' not in session:
        flash('No active verification. Please login again.', 'error')
        return redirect(url_for('login'))

    user_id = session['totp_user_id']
    db = get_db()
    user = db.execute("SELECT id, username, email, totp_secret, totp_enabled, totp_backup_codes FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('login'))

    totp = pyotp.TOTP(user['totp_secret'])

    # --- Helper to finalize login (reused for both TOTP and backup codes) ---
    def finalize_login(is_backup=False):
        user_full = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        otp_ip = session['totp_ip']
        otp_user_agent = session['totp_user_agent']
        otp_risk_score = session['totp_risk_score']
        explanation = session.pop('totp_explanation', None)
        shap_json_str = session.pop('totp_shap_json', None)
        geo = session.pop('totp_geo', None)
        admin_login = session.pop('totp_admin_login', False)
        session.pop('totp_provisioning_uri', None)
        session.pop('totp_secret', None)
        session.pop('totp_enabled', None)

        action = 'allow_with_totp' if not is_backup else 'allow_with_backup_code'
        dest = 'admin_dashboard' if admin_login else 'dashboard'
        return complete_successful_login(
            user_full, otp_ip, otp_user_agent, int(otp_risk_score * 100), action,
            explanation_str=explanation, geo=geo, shap_json_str=shap_json_str,
            activity_detail=f"Login with {'backup code' if is_backup else 'TOTP'} from {otp_ip}",
            redirect_to=dest,
            device_fingerprint=session.get('pending_device_fingerprint'),
            trust_device=session.get('pending_trust_device', False),
        )

    # --- POST: Verify code ---
    if request.method == 'POST':
        _validate_csrf()
        entered_code = request.form.get('otp', '').strip()

        # 1. Try TOTP verification first
        if totp.verify(entered_code, valid_window=1):
            # --- First time TOTP setup? Generate backup codes ---
            if not user['totp_enabled']:
                # Generate 10 backup codes (8 characters each)
                plain_codes = [secrets.token_hex(4) for _ in range(10)]
                hashed_codes = []
                for code in plain_codes:
                    hashed = bcrypt.hashpw(code.encode('utf-8'), bcrypt.gensalt())
                    hashed_codes.append(hashed.decode('utf-8'))
                
                # Save hashed codes to DB and enable TOTP
                db.execute("UPDATE users SET totp_enabled = 1, totp_backup_codes = ? WHERE id = ?",
                           (json.dumps(hashed_codes), user_id))
                db.commit()

                # Store plain codes in session and redirect to show them
                session['new_backup_codes'] = plain_codes
                flash('TOTP enabled! Please save your backup codes.', 'success')
                
                # Since we have a valid TOTP, we can keep the session data
                # and redirect to the recovery codes page
                return redirect(url_for('show_recovery_codes'))

            # --- Existing TOTP user (normal login) ---
            return finalize_login(is_backup=False)

        # 2. Try Backup Code verification (if TOTP failed)
        backup_codes_json = user['totp_backup_codes']
        if backup_codes_json:
            backup_list = json.loads(backup_codes_json)
            found_idx = -1
            for i, hashed in enumerate(backup_list):
                if bcrypt.checkpw(entered_code.encode('utf-8'), hashed.encode('utf-8')):
                    found_idx = i
                    break

            if found_idx != -1:
                # Remove the used code from the list
                backup_list.pop(found_idx)
                db.execute("UPDATE users SET totp_backup_codes = ? WHERE id = ?",
                           (json.dumps(backup_list), user_id))
                db.commit()

                flash('Backup code accepted. Please set up a new authenticator app next time you log in.', 'info')
                return finalize_login(is_backup=True)

        # 3. Neither TOTP nor backup code matched
        flash('Invalid code. Please try again.', 'error')
        return redirect(url_for('verify_otp'))

    # --- GET: Render template ---
    provisioning_uri = session.get('totp_provisioning_uri')
    is_first_time = not user['totp_enabled']

    qr_base64 = None
    if is_first_time and provisioning_uri:
        qr = qrcode.make(provisioning_uri)
        buffered = BytesIO()
        qr.save(buffered, format="PNG")
        qr_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return render_template('verify_otp.html',
                           is_first_time=is_first_time,
                           qr_base64=qr_base64,
                           secret=user['totp_secret'],
                           username=user['username'])

@app.route('/recovery_codes')
def show_recovery_codes():
    if request.method == 'POST':
        return redirect(url_for('dashboard'))
    if 'new_backup_codes' not in session:
        flash('No recovery codes available.', 'error')
        return redirect(url_for('dashboard'))
    
    codes = session.pop('new_backup_codes', [])
    return render_template('recovery_codes.html', codes=codes)

@app.route('/verify_teacher_otp', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def verify_teacher_otp():
    if 'teacher_otp' not in session:
        flash('No OTP verification in progress.', 'error')
        return redirect(url_for('login'))
    otp_generated_at = session.get('teacher_otp_generated_at')
    if otp_generated_at:
        elapsed = datetime.now() - datetime.fromisoformat(otp_generated_at)
        if elapsed.total_seconds() > 300:
            session.pop('teacher_otp', None)
            session.pop('teacher_otp_generated_at', None)
            flash('OTP has expired. Please login again.', 'error')
            return redirect(url_for('login'))
    if request.method == 'POST':
        _validate_csrf()
        entered_otp = request.form.get('otp', '').strip()
        if entered_otp == session.get('teacher_otp'):
            user_id = session.get('teacher_user_id')
            if not user_id:
                flash('Invalid verification session. Please login again.', 'error')
                return redirect(url_for('login'))
            db = get_db()
            user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            if not user:
                flash('User not found.', 'error')
                return redirect(url_for('login'))
            session_token = secrets.token_urlsafe(32)
            db.execute("UPDATE users SET session_token = ? WHERE id = ?", (session_token, user['id']))
            db.commit()
            teacher_ip = session.get('teacher_ip', request.remote_addr)
            teacher_user_agent = session.get('teacher_user_agent', request.headers.get('User-Agent', 'Unknown'))
            session.clear()
            session.permanent = True
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            session['is_admin'] = user['is_admin']
            session['last_activity'] = datetime.now().isoformat()
            session['session_token'] = session_token
            log_attempt(user_id, teacher_ip, teacher_user_agent, 'success', 0, 'allow_with_email_otp')
            log_activity(user_id, user['username'], 'login', f"Login successful with email OTP from {teacher_ip}")
            for key in ['teacher_otp', 'teacher_otp_generated_at', 'teacher_user_id', 'teacher_ip', 'teacher_user_agent']:
                session.pop(key, None)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid OTP.', 'error')
            return redirect(url_for('verify_teacher_otp'))
    return render_template('verify_teacher_otp.html')

@app.route('/reauthenticate', methods=['GET', 'POST'])
@login_required
def reauthenticate():
    if request.method == 'POST':
        _validate_csrf()
        password = request.form.get('password')
        db = get_db()
        user = db.execute("SELECT password_hash FROM users WHERE id = ?", (session['user_id'],)).fetchone()
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash']):
            session['reauthenticate_until'] = (datetime.now() + timedelta(minutes=30)).isoformat()
            session.pop('reauthenticate_redirect', None)
            flash('Re‑authentication successful. You may continue.', 'success')
            return redirect(session.get('reauthenticate_redirect', url_for('dashboard')))
        else:
            flash('Incorrect password. Access denied.', 'error')
            return redirect(url_for('dashboard'))
    return render_template('reauthenticate.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'GET':
        return render_template('forgot_password.html')
    email = request.form.get('email', '').strip().lower()
    if not email:
        flash('Email is required.', 'error')
        return redirect(url_for('forgot_password'))
    db = get_db()
    user = db.execute("SELECT id, username, email, role FROM users WHERE email = ?", (email,)).fetchone()
    if not user:
        flash('If that email exists in our system, we have sent a password reset link.', 'info')
        return redirect(url_for('login'))
    token = serializer.dumps(user['email'], salt='password-reset-salt')
    reset_link = url_for('reset_password', token=token, _external=True)
    send_user_alert(user['email'], "Password Reset Request", 
                    f"Dear {user['username']},\n\nClick the link below to reset your password (valid for 1 hour):\n{reset_link}\n\nIf you did not request this, ignore this email.")
    flash('Password reset link sent to your email. Please check your inbox.', 'success')
    return redirect(url_for('login'))

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = serializer.loads(token, salt='password-reset-salt', max_age=3600)
    except (SignatureExpired, BadSignature):
        flash('The password reset link is invalid or has expired. Please request a new one.', 'error')
        return redirect(url_for('forgot_password'))
    db = get_db()
    user = db.execute("SELECT id, email FROM users WHERE email = ?", (email,)).fetchone()
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('forgot_password'))
    if request.method == 'GET':
        return render_template('reset_password.html', token=token)
    new_password = request.form.get('new_password', '')
    confirm = request.form.get('confirm_password', '')
    if new_password != confirm:
        flash('Passwords do not match.', 'error')
        return redirect(url_for('reset_password', token=token))
    if not _is_strong_password(new_password):
        flash('Password must be at least 8 characters with uppercase, lowercase, a number, and a special character (!@#$%^&*).', 'error')
        return redirect(url_for('reset_password', token=token))
    new_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
    db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user['id']))
    db.execute("DELETE FROM failed_attempts WHERE user_id = ?", (user['id'],))
    db.execute("UPDATE users SET is_locked = 0, locked_at = NULL WHERE id = ?", (user['id'],))
    db.commit()
    log_attempt(user['id'], request.remote_addr, 'N/A', 'password_reset', 0, 'reset')
    log_activity(user['id'], user['email'], 'password_reset', "Password reset via token")
    flash('Password reset successfully. Please login with your new password.', 'success')
    return redirect(url_for('login'))

@app.route('/verify_security_questions', methods=['GET', 'POST'])
def verify_security_questions():
    if 'reset_user_id' not in session:
        flash('No password reset in progress.', 'error')
        return redirect(url_for('forgot_password'))
    if request.method == 'GET':
        questions = session.get('reset_questions', ['', '', ''])
        return render_template('verify_security_questions.html', questions=questions)
    answers = [
        request.form.get('answer_1', '').strip().lower(),
        request.form.get('answer_2', '').strip().lower(),
        request.form.get('answer_3', '').strip().lower()
    ]
    correct = 0
    for i, ans in enumerate(answers):
        if ans and bcrypt.checkpw(ans.encode('utf-8'), session['reset_answers'][i]):
            correct += 1
    if correct >= 2:
        return redirect(url_for('reset_password'))
    else:
        flash('You answered fewer than 2 questions correctly. Password reset cancelled.', 'error')
        session.pop('reset_user_id', None)
        session.pop('reset_questions', None)
        session.pop('reset_answers', None)
        return redirect(url_for('login'))

@app.route('/profile/documents', methods=['GET', 'POST'])
@login_required
@behavior_check_required
def profile_documents():
    db = get_db()
    if request.method == 'POST':
        _validate_csrf()
        scan_and_log_attack_patterns(endpoint="profile_documents", block=False)
        
        doc_name = request.form.get('document_name', '').strip()
        description = request.form.get('description', '').strip()
        file = request.files.get('file')
        
        if not doc_name or not file:
            flash('Document name and file are required.', 'error')
            return redirect(url_for('profile_documents'))
        
        if '.' not in file.filename:
            flash('File must have an extension.', 'error')
            return redirect(url_for('profile_documents'))
        
        # --- Magic bytes validation ---
        is_valid, msg = validate_file_type(file, file.filename)
        if not is_valid:
            flash(f'Invalid file: {msg}', 'error')
            return redirect(url_for('profile_documents'))
        
        upload_dir = os.path.join('static', 'uploads', 'user_docs', str(session['user_id']))
        os.makedirs(upload_dir, exist_ok=True)
        safe_filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{safe_filename}"
        file_path = os.path.join(upload_dir, unique_filename)
        file.save(file_path)
        
        db.execute(
            "INSERT INTO user_documents (user_id, document_name, file_path, description) VALUES (?, ?, ?, ?)",
            (session['user_id'], doc_name, file_path, description)
        )
        db.commit()
        log_activity(session['user_id'], session['username'], 'document_upload', f"Uploaded {doc_name} ({file.filename})")
        flash('Document uploaded successfully.', 'success')
        return redirect(url_for('profile_documents'))
    
    docs = db.execute("SELECT * FROM user_documents WHERE user_id = ? ORDER BY uploaded_at DESC", (session['user_id'],)).fetchall()
    return render_template('profile_documents.html', documents=docs)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
@behavior_check_required
def profile():
    db = get_db()
    if request.method == 'POST':
        _validate_csrf()
        scan_and_log_attack_patterns(endpoint="profile", block=False, lock_user=True)
        xss_pattern = re.compile(r'<\s*script|javascript\s*:|on\w+\s*=|<\s*iframe|<\s*img[^>]*onerror', re.IGNORECASE)
        full_name = request.form.get('full_name', '')
        phone = request.form.get('phone', '')
        address = request.form.get('address', '')
        if (xss_pattern.search(full_name) or xss_pattern.search(phone) or xss_pattern.search(address)):
            db.execute("UPDATE users SET is_locked=1, locked_at=CURRENT_TIMESTAMP WHERE id=?", (session['user_id'],))
            db.commit()
            send_admin_alert("Account locked due to XSS attempt", f"User: {session['username']}\nIP: {request.remote_addr}\nPayload: {full_name or phone or address}")
            user_email = db.execute("SELECT email FROM users WHERE id = ?", (session['user_id'],)).fetchone()['email']
            send_user_alert(user_email, "Your account has been locked", "Your account has been locked because a security violation (XSS attempt) was detected. Please contact the administrator to unlock it.")
            session.clear()
            flash('XSS attempt detected. Your account has been locked. Contact admin.', 'error')
            return redirect(url_for('login'))
        db.execute("INSERT OR REPLACE INTO profiles (user_id, full_name, phone, address) VALUES (?, ?, ?, ?)",
                   (session['user_id'], full_name, phone, address))
        db.commit()
        log_activity(session['user_id'], session['username'], 'profile_update', f"Name: {full_name}, Phone: {phone}, Address: {address}")
        flash('Profile updated successfully', 'success')
        return redirect(url_for('profile'))
    user = db.execute("SELECT * FROM users WHERE id = ?", (session['user_id'],)).fetchone()
    profile = db.execute("SELECT * FROM profiles WHERE user_id = ?", (session['user_id'],)).fetchone()
    return render_template('profile.html', user=user, profile=profile)

@app.route('/profile/documents/delete/<int:doc_id>', methods=['POST'])
@login_required
def delete_document(doc_id):
    db = get_db()
    doc = db.execute("SELECT * FROM user_documents WHERE id = ? AND user_id = ?", 
                     (doc_id, session['user_id'])).fetchone()
    if not doc:
        flash('Document not found.', 'error')
        return redirect(url_for('profile_documents'))
    
    # Delete the file from disk
    file_path = doc['file_path']
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Could not delete file: {e}")
    
    db.execute("DELETE FROM user_documents WHERE id = ?", (doc_id,))
    db.commit()
    
    log_activity(session['user_id'], session['username'], 'document_deleted', 
                 f"Deleted document: {doc['document_name']}")
    flash('Document deleted successfully.', 'success')
    return redirect(url_for('profile_documents'))

@app.route('/dashboard')
@login_required
def dashboard():
    db = get_db()
    user = db.execute("SELECT email FROM users WHERE id = ?", (session['user_id'],)).fetchone()
    user_email = user['email'] if user else None
    now = datetime.now()
    semester_week = min(max(((now - datetime(now.year, 1, 1)).days // 7) + 1, 1), 20)

    enrolled_courses = []
    if session['role'] == 'student':
        enrolled_courses = db.execute('''
            SELECT c.id, c.course_code, c.course_name, u.username as teacher_name
            FROM enrollments e
            JOIN courses c ON e.course_id = c.id
            JOIN users u ON c.teacher_id = u.id
            WHERE e.student_id = ?
            ORDER BY c.course_name
        ''', (session['user_id'],)).fetchall()

    upcoming_assignments = db.execute('''
        SELECT a.id, a.title, a.due_date, c.course_name, c.course_code
        FROM assignments a
        JOIN courses c ON a.course_id = c.id
        JOIN enrollments e ON c.id = e.course_id
        WHERE e.student_id = ? AND a.due_date > CURRENT_TIMESTAMP
        ORDER BY a.due_date ASC
        LIMIT 5
    ''', (session['user_id'],)).fetchall()

    upcoming_exams = db.execute('''
        SELECT e.id, e.title, e.exam_date, e.start_time, e.duration_minutes, c.course_name, c.course_code
        FROM exams e
        JOIN courses c ON e.course_id = c.id
        JOIN enrollments en ON c.id = en.course_id
        WHERE en.student_id = ? AND e.exam_date >= DATE('now')
        ORDER BY e.exam_date ASC
        LIMIT 5
    ''', (session['user_id'],)).fetchall()

    notices = db.execute('''
        SELECT n.id, n.title, n.content, n.posted_at, u.username as author
        FROM notices n
        JOIN users u ON n.posted_by = u.id
        WHERE n.target_role IN ('student', 'all')
        ORDER BY n.posted_at DESC
        LIMIT 5
    ''').fetchall()

    pending_tasks = db.execute('''
        SELECT COUNT(*) as cnt
        FROM assignments a
        JOIN enrollments e ON a.course_id = e.course_id
        WHERE e.student_id = ? AND a.due_date >= CURRENT_TIMESTAMP
    ''', (session['user_id'],)).fetchone()['cnt'] or 0

    tasks_due_today = db.execute('''
        SELECT COUNT(*) as cnt
        FROM assignments a
        JOIN enrollments e ON a.course_id = e.course_id
        WHERE e.student_id = ? AND DATE(a.due_date) = DATE('now')
    ''', (session['user_id'],)).fetchone()['cnt'] or 0

    grade_row = db.execute('''
        SELECT AVG(CASE WHEN a.total_marks > 0 THEN (s.grade * 1.0 / a.total_marks * 100) ELSE NULL END) as avg_grade
        FROM submissions s
        JOIN assignments a ON s.assignment_id = a.id
        JOIN enrollments e ON a.course_id = e.course_id
        WHERE s.student_id = ? AND e.student_id = ? AND s.grade IS NOT NULL
    ''', (session['user_id'], session['user_id'])).fetchone()
    average_grade = round(grade_row['avg_grade'] or 0, 1)

    attendance_row = db.execute('''
        SELECT COUNT(*) as total,
               SUM(CASE WHEN status IN ('present', 'late') THEN 1 ELSE 0 END) as present
        FROM attendance
        WHERE student_id = ?
    ''', (session['user_id'],)).fetchone()
    attendance_total = attendance_row['total'] or 0
    attendance_percentage = round((attendance_row['present'] or 0) / attendance_total * 100, 1) if attendance_total else 0

    grade_increase = 3.2

    grade_labels = []
    grade_data_values = []
    for course in enrolled_courses:
        avg_course = db.execute('''
            SELECT AVG(CASE WHEN a.total_marks > 0 THEN (s.grade * 1.0 / a.total_marks * 100) ELSE NULL END) as pct
            FROM assignments a
            LEFT JOIN submissions s ON a.id = s.assignment_id AND s.student_id = ?
            WHERE a.course_id = ?
        ''', (session['user_id'], course['id'])).fetchone()
        grade_labels.append(course['course_code'])
        grade_data_values.append(round(avg_course['pct'] or 0, 1))

    week_labels = []
    attendance_weekly = []
    for i in range(6, -1, -1):
        day = (now.date() - timedelta(days=i))
        stats = db.execute('''
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN status IN ('present', 'late') THEN 1 ELSE 0 END) as present
            FROM attendance
            WHERE student_id = ? AND date = ?
        ''', (session['user_id'], day)).fetchone()
        total = stats['total'] or 0
        present = stats['present'] or 0
        week_labels.append(day.strftime('%a'))
        attendance_weekly.append(round((present / total * 100) if total else 0, 1))
    
    ai_enabled = get_ai_security_status()
    ai_metrics = get_ai_metrics()

    # Get user's full name from profiles
    profile = db.execute("SELECT full_name FROM profiles WHERE user_id = ?", (session['user_id'],)).fetchone()
    full_name = profile['full_name'] if profile else session['username']

    return render_template('dashboard.html',
                           user_email=user_email,
                           full_name=full_name,
                           enrolled_courses=enrolled_courses,
                           upcoming_assignments=upcoming_assignments,
                           upcoming_exams=upcoming_exams,
                           notices=notices,
                           now=now,
                           semester_week=semester_week,
                           pending_tasks=pending_tasks,
                           tasks_due_today=tasks_due_today,
                           attendance_percentage=attendance_percentage,
                           average_grade=average_grade,
                           grade_increase=grade_increase,
                           grade_labels=grade_labels,
                           grade_data_values=grade_data_values,
                           week_labels=week_labels,
                           attendance_weekly=attendance_weekly, ai_enabled=ai_enabled,ai_metrics=ai_metrics)

@app.route('/exams/<int:exam_id>/my_marks')
@login_required
def view_my_exam_marks(exam_id):
    if session['role'] != 'student':
        abort(403)
    db = get_db()
    exam = db.execute('''
        SELECT e.*, c.course_name
        FROM exams e
        JOIN courses c ON e.course_id = c.id
        WHERE e.id = ?
    ''', (exam_id,)).fetchone()
    if not exam:
        abort(404)
    enrollment = db.execute('''
        SELECT * FROM enrollments
        WHERE student_id = ? AND course_id = ?
    ''', (session['user_id'], exam['course_id'])).fetchone()
    if not enrollment:
        flash('You are not enrolled in this course.', 'error')
        return redirect(url_for('list_exams'))
    grade = db.execute('''
        SELECT marks_obtained FROM exam_grades
        WHERE exam_id = ? AND student_id = ?
    ''', (exam_id, session['user_id'])).fetchone()
    marks = grade['marks_obtained'] if grade else 'Not graded yet'
    return render_template('student_exam_marks.html', exam=exam, marks=marks)

@app.route('/teacher/grades')
@login_required
def teacher_gradebook_overview():
    if session['role'] != 'teacher':
        abort(403)
    db = get_db()
    courses = db.execute('''
        SELECT id, course_code, course_name,
               (SELECT COUNT(*) FROM enrollments WHERE course_id = c.id) as student_count
        FROM courses c
        WHERE c.teacher_id = ?
        ORDER BY c.course_name
    ''', (session['user_id'],)).fetchall()
    return render_template('teacher_gradebook_overview.html', courses=courses)

@app.route('/my_exam_marks')
@login_required
def my_exam_marks():
    if session['role'] != 'student':
        abort(403)
    db = get_db()
    exams = db.execute('''
        SELECT e.id, e.title, e.exam_date, e.start_time, e.total_marks, 
               c.course_name, 
               eg.marks_obtained
        FROM exams e
        JOIN courses c ON e.course_id = c.id
        JOIN enrollments en ON c.id = en.course_id
        LEFT JOIN exam_grades eg ON e.id = eg.exam_id AND eg.student_id = ?
        WHERE en.student_id = ?
        ORDER BY e.exam_date DESC
    ''', (session['user_id'], session['user_id'])).fetchall()
    exam_list = []
    for row in exams:
        exam = dict(row)
        exam['marks_obtained'] = exam['marks_obtained'] if exam['marks_obtained'] is not None else 'Not graded'
        exam_list.append(exam)
    return render_template('my_exam_marks.html', exams=exam_list)

@app.route('/change_password', methods=['GET', 'POST'])
@login_required
@behavior_check_required
def change_password():
    if request.method == 'POST':
        _validate_csrf()
        old_pw = request.form['old_password']
        new_pw = request.form['new_password']
        confirm = request.form['confirm_password']
        
        if new_pw != confirm:
            flash('Passwords do not match.', 'error')
            return redirect(url_for('change_password'))
        
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE id = ?", (session['user_id'],)).fetchone()
        if not bcrypt.checkpw(old_pw.encode('utf-8'), user['password_hash']):
            flash('Current password is incorrect.', 'error')
            return redirect(url_for('change_password'))
        
        # --- Password History Check ---
        history = db.execute('''
            SELECT password_hash FROM password_history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 5
        ''', (session['user_id'],)).fetchall()
        
        for h in history:
            if bcrypt.checkpw(new_pw.encode('utf-8'), h['password_hash']):
                flash('You have used this password recently. Please choose a new one.', 'error')
                return redirect(url_for('change_password'))
        
        # --- Risk Check ---
        risk = old_rule_based_risk(session['user_id'], request.remote_addr, request.headers.get('User-Agent',''), {}, session['role'])
        if risk >= 3:
            flash('High risk detected. Password change requires additional verification. Please contact admin.', 'error')
            return redirect(url_for('dashboard'))
        
        # --- Update Password ---
        new_hash = bcrypt.hashpw(new_pw.encode('utf-8'), bcrypt.gensalt())
        db.execute("UPDATE users SET password_hash = ?, password_changed_at = CURRENT_TIMESTAMP WHERE id = ?",
                   (new_hash, session['user_id']))
        
        # --- Store in History ---
        db.execute("INSERT INTO password_history (user_id, password_hash) VALUES (?, ?)",
                   (session['user_id'], new_hash))
        
        # Keep only last 5 entries
        db.execute('''
            DELETE FROM password_history
            WHERE user_id = ? AND id NOT IN (
                SELECT id FROM password_history
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 5
            )
        ''', (session['user_id'], session['user_id']))
        
        db.commit()
        log_activity(session['user_id'], session['username'], 'password_changed', "Password changed")
        flash('Password changed successfully!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('change_password.html')

@app.route('/materials/delete/<int:material_id>', methods=['POST'])
@login_required
def delete_material(material_id):
    if session['role'] != 'teacher':
        abort(403)
    db = get_db()
    material = db.execute("SELECT * FROM materials WHERE id = ?", (material_id,)).fetchone()
    if not material:
        flash('Material not found.', 'error')
        return redirect(url_for('list_courses'))
    # Check if the teacher owns the course
    course = db.execute("SELECT teacher_id FROM courses WHERE id = ?", (material['course_id'],)).fetchone()
    if course['teacher_id'] != session['user_id']:
        abort(403)
    # Delete the material
    db.execute("DELETE FROM materials WHERE id = ?", (material_id,))
    db.commit()
    log_activity(session['user_id'], session['username'], 'material_deleted', f"Deleted material ID {material_id}")
    flash('Material deleted successfully.', 'success')
    return redirect(url_for('course_materials', course_id=material['course_id']))


@app.route('/courses/<int:course_id>/materials')
@login_required
def course_materials(course_id):
    db = get_db()
    
    # Fetch course with teacher username
    course = db.execute('''
        SELECT c.*, u.username as teacher_name
        FROM courses c
        LEFT JOIN users u ON c.teacher_id = u.id
        WHERE c.id = ?
    ''', (course_id,)).fetchone()
    
    if not course:
        abort(404)
    
    # --- Permission checks ---
    if session['role'] == 'teacher':
        # Teacher can only view their own courses
        if course['teacher_id'] != session['user_id']:
            abort(403)
    
    elif session['role'] == 'student':
        # Student must be enrolled
        enrollment = db.execute(
            "SELECT * FROM enrollments WHERE student_id = ? AND course_id = ?",
            (session['user_id'], course_id)
        ).fetchone()
        if not enrollment:
            abort(403)
    
    # Admin: no additional check needed – can view any course
    
    # Fetch materials
    materials = db.execute(
        "SELECT * FROM materials WHERE course_id = ? ORDER BY uploaded_at DESC",
        (course_id,)
    ).fetchall()
    
    return render_template('materials.html', course=course, materials=materials)

@app.route('/courses')
@login_required
def list_courses():
    db = get_db()
    if session['role'] == 'teacher':
        courses = db.execute('''
            SELECT c.*, u.username as teacher_name,
                   (SELECT COUNT(*) FROM enrollments WHERE course_id = c.id) as student_count
            FROM courses c
            JOIN users u ON c.teacher_id = u.id
            WHERE c.teacher_id = ?
        ''', (session['user_id'],)).fetchall()
        return render_template('courses.html', courses=courses, enrolled_ids=[])
    else:
        all_courses = db.execute('''
            SELECT c.*, u.username as teacher_name,
                   (SELECT COUNT(*) FROM enrollments WHERE course_id = c.id) as student_count
            FROM courses c
            JOIN users u ON c.teacher_id = u.id
            ORDER BY c.course_name
        ''').fetchall()
        enrolled = db.execute("SELECT course_id FROM enrollments WHERE student_id = ?", (session['user_id'],)).fetchall()
        enrolled_ids = [e['course_id'] for e in enrolled]
        return render_template('courses.html', courses=all_courses, enrolled_ids=enrolled_ids)
    
@app.route('/assignments/<int:assignment_id>/grade', methods=['GET', 'POST'])
@login_required
@behavior_check_required
def grade_submissions(assignment_id):
    # Only teachers
    if session['role'] != 'teacher':
        abort(403)
    db = get_db()
    
    # Fetch assignment and verify ownership
    assignment = db.execute('''
        SELECT a.*, c.course_name, c.teacher_id
        FROM assignments a
        JOIN courses c ON a.course_id = c.id
        WHERE a.id = ?
    ''', (assignment_id,)).fetchone()
    if not assignment:
        abort(404)
    if assignment['teacher_id'] != session['user_id']:
        abort(403)
    
    # Fetch all submissions for this assignment with student details
    submissions = db.execute('''
        SELECT s.id, s.submission_text, s.file_url, s.submitted_at, s.grade, s.feedback,
               u.id as student_id, u.username, p.full_name
        FROM submissions s
        JOIN users u ON s.student_id = u.id
        LEFT JOIN profiles p ON u.id = p.user_id
        WHERE s.assignment_id = ?
        ORDER BY s.submitted_at ASC
    ''', (assignment_id,)).fetchall()
    
    # Handle POST (bulk grade update)
    if request.method == 'POST':
        _validate_csrf()
        for row in submissions:
            grade_key = f'grade_{row["id"]}'
            feedback_key = f'feedback_{row["id"]}'
            if grade_key in request.form:
                grade_val = request.form[grade_key].strip()
                feedback_val = request.form.get(feedback_key, '').strip()
                if grade_val:
                    try:
                        grade_int = int(grade_val)
                        db.execute('''
                            UPDATE submissions
                            SET grade = ?, feedback = ?, graded_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        ''', (grade_int, feedback_val, row['id']))
                    except ValueError:
                        pass
        db.commit()
        flash('Grades updated successfully.', 'success')
        log_activity(session['user_id'], session['username'], 'assignment_grades_bulk',
                     f"Graded submissions for assignment ID {assignment_id}")
        return redirect(url_for('grade_submissions', assignment_id=assignment_id))
    
    return render_template('grade_submissions.html',
                           assignment=assignment,
                           submissions=submissions)

@app.route('/courses/create', methods=['GET', 'POST'])
@login_required
@behavior_check_required
def create_course():
    if session['role'] != 'teacher':
        abort(403)
    
    db = get_db()
    
    # GET request: fetch course count for stats
    courses_count = db.execute(
        "SELECT COUNT(*) as count FROM courses WHERE teacher_id = ?", 
        (session['user_id'],)
    ).fetchone()['count']
    
    if request.method == 'POST':
        _validate_csrf()
        code = request.form.get('course_code', '').strip()
        name = request.form.get('course_name', '').strip()
        desc = request.form.get('description', '').strip()
        
        # Basic backend validation (redundant but safe)
        if not code or not name:
            flash('Course code and name are required.', 'error')
            return render_template('create_course.html', courses_count=courses_count, now=datetime.now())
        
        if len(code) < 2:
            flash('Course code must be at least 2 characters.', 'error')
            return render_template('create_course.html', courses_count=courses_count, now=datetime.now())
        
        try:
            db.execute(
                "INSERT INTO courses (course_code, course_name, description, teacher_id) VALUES (?, ?, ?, ?)",
                (code, name, desc, session['user_id'])
            )
            db.commit()
            log_activity(session['user_id'], session['username'], 'course_created', f"Course: {name} ({code})")
            flash('Course created successfully', 'success')
            return redirect(url_for('list_courses'))
        except sqlite3.IntegrityError:
            flash('Course code already exists. Please choose a different code.', 'error')
            return render_template('create_course.html', courses_count=courses_count, now=datetime.now())
    
    # GET request (or failed POST)
    return render_template('create_course.html', 
                           courses_count=courses_count, 
                           now=datetime.now())

@app.route('/courses/enroll/<int:course_id>', methods=['POST'])
@login_required
def enroll_course(course_id):
    if session['role'] != 'student':
        abort(403)
    _validate_csrf()
    db = get_db()
    try:
        db.execute("INSERT INTO enrollments (student_id, course_id) VALUES (?, ?)", (session['user_id'], course_id))
        db.commit()
        log_activity(session['user_id'], session['username'], 'course_enrollment', f"Enrolled in course ID {course_id}")
        flash('Successfully enrolled in course!', 'success')
    except sqlite3.IntegrityError:
        flash('You are already enrolled in this course.', 'warning')
    return redirect(url_for('list_courses'))


@app.route('/courses/<int:course_id>/materials/upload', methods=['GET', 'POST'])
@login_required
@behavior_check_required
def upload_material(course_id):
    if session['role'] != 'teacher':
        abort(403)
    db = get_db()
    course = db.execute("SELECT * FROM courses WHERE id = ? AND teacher_id = ?", (course_id, session['user_id'])).fetchone()
    if not course:
        abort(403)
    if request.method == 'POST':
        _validate_csrf()
        title = request.form['title']
        desc = request.form['description']
        url = request.form['file_url']
        db.execute("INSERT INTO materials (course_id, title, file_url, description, uploaded_by) VALUES (?, ?, ?, ?, ?)",
                   (course_id, title, url, desc, session['user_id']))
        db.commit()
        log_activity(session['user_id'], session['username'], 'material_uploaded', f"Material: {title} for course {course['course_name']}")
        flash('Material uploaded', 'success')
        return redirect(url_for('course_materials', course_id=course_id))
    return render_template('upload_materials.html', course=course)

@app.route('/teacher/attendance')
@login_required
def teacher_attendance_dashboard():
    if session['role'] != 'teacher':
        abort(403)
    db = get_db()
    courses = db.execute('''
        SELECT id, course_code, course_name
        FROM courses
        WHERE teacher_id = ?
        ORDER BY course_name
    ''', (session['user_id'],)).fetchall()
    return render_template('teacher_attendance_dashboard.html', courses=courses)

@app.route('/notices')
@login_required
def view_notices():
    db = get_db()
    notices = db.execute('''
        SELECT n.*, u.username as author
        FROM notices n JOIN users u ON n.posted_by = u.id
        WHERE n.target_role = ? OR n.target_role = 'all'
        ORDER BY n.posted_at DESC
    ''', (session['role'],)).fetchall()
    return render_template('notices.html', notices=notices)
@app.route('/notices/delete/<int:notice_id>', methods=['POST'])
@login_required
def delete_notice(notice_id):
    if session['role'] not in ['teacher', 'admin']:
        abort(403)
    db = get_db()
    notice = db.execute("SELECT * FROM notices WHERE id = ?", (notice_id,)).fetchone()
    if not notice:
        flash('Notice not found.', 'error')
        return redirect(url_for('view_notices'))
    
    # Check if user is the author or admin
    if session['role'] != 'admin' and notice['posted_by'] != session['user_id']:
        abort(403)
    
    db.execute("DELETE FROM notices WHERE id = ?", (notice_id,))
    db.commit()
    
    log_activity(session['user_id'], session['username'], 'notice_deleted', f"Deleted notice: {notice['title']}")
    flash('Notice deleted successfully.', 'success')
    return redirect(url_for('view_notices'))

@app.route('/notices/post', methods=['GET', 'POST'])
@login_required
@behavior_check_required
def post_notice():
    if session['role'] not in ['teacher', 'admin']:
        abort(403)
    if request.method == 'POST':
        _validate_csrf()
        title = request.form['title']
        content = request.form['content']
        target = request.form['target_role']
        db = get_db()
        db.execute("INSERT INTO notices (title, content, posted_by, target_role) VALUES (?, ?, ?, ?)",
                   (title, content, session['user_id'], target))
        db.commit()
        
        # Send notifications to targeted users
        if target == 'all':
            users = db.execute("SELECT id FROM users").fetchall()
        elif target == 'student':
            users = db.execute("SELECT id FROM users WHERE role = 'student'").fetchall()
        elif target == 'teacher':
            users = db.execute("SELECT id FROM users WHERE role = 'teacher'").fetchall()
        else:
            users = []
        
        for user in users:
            send_notification(
                user['id'],
                f"📢 New notice: {title}",
                url_for('view_notices')
            )
        
        log_activity(session['user_id'], session['username'], 'notice_posted', f"Title: {title}, Target: {target}")
        flash('Notice posted successfully.', 'success')
        return redirect(url_for('view_notices'))
    
    # GET request – pass `now` for the preview date
    return render_template('post_notice.html', now=datetime.now())

@app.route('/routine')
@login_required
def view_routine():
    db = get_db()
    if session['role'] == 'student':
        routines = db.execute('''
            SELECT r.*, c.course_name, c.course_code
            FROM routines r
            JOIN courses c ON r.course_id = c.id
            JOIN enrollments e ON c.id = e.course_id
            WHERE e.student_id = ?
            ORDER BY 
                CASE day_of_week
                    WHEN 'Monday' THEN 1 WHEN 'Tuesday' THEN 2 WHEN 'Wednesday' THEN 3
                    WHEN 'Thursday' THEN 4 WHEN 'Friday' THEN 5 WHEN 'Saturday' THEN 6
                    WHEN 'Sunday' THEN 7
                END, start_time
        ''', (session['user_id'],)).fetchall()
    else:
        routines = db.execute('''
            SELECT r.*, c.course_name, c.course_code
            FROM routines r
            JOIN courses c ON r.course_id = c.id
            WHERE c.teacher_id = ?
            ORDER BY day_of_week, start_time
        ''', (session['user_id'],)).fetchall()
    return render_template('routine.html', routines=routines)
from datetime import datetime

@app.route('/routine/add', methods=['GET', 'POST'])
@login_required
@behavior_check_required
def add_routine():
    if session['role'] != 'teacher':
        abort(403)
    db = get_db()
    courses = db.execute("SELECT id, course_name, course_code FROM courses WHERE teacher_id = ?", (session['user_id'],)).fetchall()
    
    # Count routines for stats
    routines_count = db.execute("SELECT COUNT(*) as count FROM routines").fetchone()['count']
    
    if request.method == 'POST':
        _validate_csrf()
        course_id = request.form['course_id']
        day = request.form['day_of_week']
        start = request.form['start_time']
        end = request.form['end_time']
        room = request.form['room']
        db.execute("INSERT INTO routines (course_id, day_of_week, start_time, end_time, room) VALUES (?, ?, ?, ?, ?)",
                   (course_id, day, start, end, room))
        db.commit()
        log_activity(session['user_id'], session['username'], 'routine_added', f"Course ID {course_id}, {day} {start}-{end}")
        flash('Routine added successfully.', 'success')
        return redirect(url_for('view_routine'))
    
    return render_template('add_routine.html', 
                           courses=courses, 
                           routines_count=routines_count, 
                           now=datetime.now())


@app.route('/routine/delete/<int:routine_id>', methods=['POST'])
@login_required
def delete_routine(routine_id):
    if session['role'] != 'teacher':
        abort(403)
    db = get_db()
    routine = db.execute("SELECT * FROM routines WHERE id = ?", (routine_id,)).fetchone()
    if not routine:
        flash('Routine entry not found.', 'error')
        return redirect(url_for('view_routine'))
    # Optional: check ownership (course teacher)
    course = db.execute("SELECT teacher_id FROM courses WHERE id = ?", (routine['course_id'],)).fetchone()
    if course['teacher_id'] != session['user_id']:
        abort(403)
    db.execute("DELETE FROM routines WHERE id = ?", (routine_id,))
    db.commit()
    log_activity(session['user_id'], session['username'], 'routine_deleted', f"Deleted routine ID {routine_id}")
    flash('Routine entry deleted.', 'success')
    return redirect(url_for('view_routine'))

@app.route('/assignments')
@login_required
def list_assignments():
    db = get_db()
    now = datetime.now()  # <-- ADD THIS

    if session['role'] == 'teacher':
        assignments = db.execute('''
            SELECT a.*, c.course_name 
            FROM assignments a JOIN courses c ON a.course_id = c.id
            WHERE c.teacher_id = ?
            ORDER BY a.due_date ASC
        ''', (session['user_id'],)).fetchall()
    else:
        assignments = db.execute('''
            SELECT a.*, c.course_name, 
                   (SELECT grade FROM submissions WHERE assignment_id = a.id AND student_id = ?) as my_grade,
                   (SELECT submitted_at FROM submissions WHERE assignment_id = a.id AND student_id = ?) as submitted_at
            FROM assignments a
            JOIN courses c ON a.course_id = c.id
            JOIN enrollments e ON c.id = e.course_id
            WHERE e.student_id = ?
            ORDER BY a.due_date ASC
        ''', (session['user_id'], session['user_id'], session['user_id'])).fetchall()

    return render_template('assignments.html', 
                           assignments=assignments, 
                           role=session['role'],
                           now=now)  # <-- ADD THIS

@app.route('/assignments/create', methods=['GET', 'POST'])
@login_required
@behavior_check_required
def create_assignment():
    if session['role'] != 'teacher':
        abort(403)
    db = get_db()
    courses = db.execute("SELECT id, course_name FROM courses WHERE teacher_id = ?", (session['user_id'],)).fetchall()
    if not courses:
        flash('You need to create a course first.', 'warning')
        return redirect(url_for('create_course'))
    if request.method == 'POST':
        _validate_csrf()
        course_id = request.form['course_id']
        title = request.form['title']
        description = request.form['description']
        due_date = request.form['due_date']
        total_marks = request.form['total_marks']
        db.execute('''
            INSERT INTO assignments (course_id, title, description, due_date, total_marks, created_by)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (course_id, title, description, due_date, total_marks, session['user_id']))
        db.commit()
        
        # --- Send notifications to all enrolled students ---
        course = db.execute("SELECT course_name FROM courses WHERE id = ?", (course_id,)).fetchone()
        course_name = course['course_name'] if course else 'the course'
        
        students = db.execute('''
            SELECT student_id FROM enrollments WHERE course_id = ?
        ''', (course_id,)).fetchall()
        
        for student in students:
            send_notification(
                student['student_id'],
                f"📝 New assignment: {title} in {course_name}",
                url_for('list_assignments')
            )
        
        log_activity(session['user_id'], session['username'], 'assignment_created', f"Assignment: {title} for course ID {course_id}")
        flash('Assignment created successfully', 'success')
        return redirect(url_for('list_assignments'))
    return render_template('create_assignment.html', courses=courses)

@app.route('/assignments/submit/<int:assignment_id>', methods=['GET', 'POST'])
@login_required
def submit_assignment(assignment_id):
    if session['role'] != 'student':
        abort(403)
    db = get_db()
    assignment = db.execute('''
        SELECT a.*, c.id as course_id, c.course_name
        FROM assignments a JOIN courses c ON a.course_id = c.id
        WHERE a.id = ?
    ''', (assignment_id,)).fetchone()
    if not assignment:
        abort(404)
    enrollment = db.execute("SELECT * FROM enrollments WHERE student_id = ? AND course_id = ?", (session['user_id'], assignment['course_id'])).fetchone()
    if not enrollment:
        flash('You are not enrolled in this course.', 'error')
        return redirect(url_for('list_assignments'))

    # Check for existing submission
    existing = db.execute("SELECT * FROM submissions WHERE assignment_id = ? AND student_id = ?",
                         (assignment_id, session['user_id'])).fetchone()

    if request.method == 'POST':
        _validate_csrf()
        submission_text = request.form['submission_text']
        file_url = request.form.get('file_url', '')

        if existing:
            # Update existing submission
            db.execute("""
                UPDATE submissions 
                SET submission_text = ?, file_url = ?, submitted_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            """, (submission_text, file_url, existing['id']))
            log_activity(session['user_id'], session['username'], 'assignment_updated',
                         f"Assignment ID {assignment_id}, text: {submission_text[:100]}")
            flash('Assignment updated successfully.', 'success')
        else:
            # Insert new submission
            db.execute("""
                INSERT INTO submissions (assignment_id, student_id, submission_text, file_url)
                VALUES (?, ?, ?, ?)
            """, (assignment_id, session['user_id'], submission_text, file_url))
            log_activity(session['user_id'], session['username'], 'assignment_submitted',
                         f"Assignment ID {assignment_id}, text: {submission_text[:100]}")
            flash('Assignment submitted successfully.', 'success')
        db.commit()
        return redirect(url_for('list_assignments'))

    # Pass existing submission (or None) to template
    return render_template('submit_assignment.html', assignment=assignment, submission=existing)

@app.route('/gradebook/<int:course_id>', methods=['GET', 'POST'])
@login_required
def gradebook(course_id):
    if session.get('role') != 'teacher':
        abort(403)
    
    db = get_db()
    
    # Verify course ownership
    course = db.execute('''
        SELECT id, course_code, course_name
        FROM courses
        WHERE id = ? AND teacher_id = ?
    ''', (course_id, session['user_id'])).fetchone()
    if not course:
        flash('Course not found or you do not have permission.', 'error')
        return redirect(url_for('teacher_gradebook_overview'))
    
    # Students
    students = db.execute('''
        SELECT u.id, u.username, p.full_name
        FROM enrollments e
        JOIN users u ON e.student_id = u.id
        LEFT JOIN profiles p ON u.id = p.user_id
        WHERE e.course_id = ?
        ORDER BY p.full_name, u.username
    ''', (course_id,)).fetchall()
    
    # Assignments
    assignments = db.execute('''
        SELECT id, title, total_marks
        FROM assignments
        WHERE course_id = ?
        ORDER BY due_date, id
    ''', (course_id,)).fetchall()
    
    # Exams
    exams = db.execute('''
        SELECT id, title, total_marks
        FROM exams
        WHERE course_id = ?
        ORDER BY exam_date, id
    ''', (course_id,)).fetchall()
    
    assessments = []
    for a in assignments:
        assessments.append({'type': 'assignment', 'id': a['id'], 'title': a['title'], 'max_marks': a['total_marks']})
    for e in exams:
        assessments.append({'type': 'exam', 'id': e['id'], 'title': e['title'], 'max_marks': e['total_marks']})
    
    # Grade data
    grade_data = []
    for student in students:
        row = {
            'student_id': student['id'],
            'student_name': student['full_name'] or student['username'],
            'marks': {}
        }
        for ass in assessments:
            if ass['type'] == 'assignment':
                sub = db.execute('''
                    SELECT grade FROM submissions 
                    WHERE assignment_id = ? AND student_id = ?
                ''', (ass['id'], student['id'])).fetchone()
                row['marks'][ass['id']] = sub['grade'] if sub and sub['grade'] is not None else None
            else:  # exam
                grade = db.execute('''
                    SELECT marks_obtained FROM exam_grades 
                    WHERE exam_id = ? AND student_id = ?
                ''', (ass['id'], student['id'])).fetchone()
                row['marks'][ass['id']] = grade['marks_obtained'] if grade and grade['marks_obtained'] is not None else None
        grade_data.append(row)
    
    # Stats
    total_students = len(students)
    all_totals = []
    pending_grades = 0
    for row in grade_data:
        total = 0
        has_null = False
        for ass in assessments:
            mark = row['marks'].get(ass['id'])
            if mark is not None:
                total += mark
            else:
                has_null = True
        all_totals.append(total)
        if has_null:
            pending_grades += 1
    avg_mark = round(sum(all_totals) / total_students, 1) if total_students else 0
    highest_score = max(all_totals) if all_totals else 0
    
    # POST: Save grades
    if request.method == 'POST':
        _validate_csrf()
        for student in students:
            for ass in assessments:
                field_name = f"{ass['type']}_{ass['id']}_{student['id']}"
                value = request.form.get(field_name, '').strip()
                if value == '':
                    if ass['type'] == 'assignment':
                        db.execute('UPDATE submissions SET grade = NULL WHERE assignment_id = ? AND student_id = ?', 
                                  (ass['id'], student['id']))
                    else:
                        db.execute('DELETE FROM exam_grades WHERE exam_id = ? AND student_id = ?', 
                                  (ass['id'], student['id']))
                else:
                    try:
                        mark = int(value)
                        if mark < 0: mark = 0
                        if ass['type'] == 'assignment':
                            existing = db.execute('SELECT id FROM submissions WHERE assignment_id = ? AND student_id = ?', 
                                                 (ass['id'], student['id'])).fetchone()
                            if existing:
                                db.execute('UPDATE submissions SET grade = ?, graded_at = CURRENT_TIMESTAMP WHERE id = ?', 
                                          (mark, existing['id']))
                            else:
                                db.execute('''INSERT INTO submissions (assignment_id, student_id, grade, graded_at) 
                                              VALUES (?, ?, ?, CURRENT_TIMESTAMP)''', 
                                          (ass['id'], student['id'], mark))
                        else:
                            db.execute('INSERT OR REPLACE INTO exam_grades (exam_id, student_id, marks_obtained) VALUES (?, ?, ?)', 
                                      (ass['id'], student['id'], mark))
                    except ValueError:
                        pass
        db.commit()
        flash('Grades updated successfully.', 'success')
        log_activity(session['user_id'], session['username'], 'gradebook_update', 
                    f'Updated grades for course {course_id}')
        return redirect(url_for('gradebook', course_id=course_id))
    
    return render_template('teacher_gradebook.html',
                           course=course,
                           students=students,
                           assessments=assessments,
                           grade_data=grade_data,
                           total_students=total_students,
                           avg_mark=avg_mark,
                           highest_score=highest_score,
                           pending_grades=pending_grades)
    
@app.route('/assignments/grade/<int:submission_id>', methods=['GET', 'POST'])
@login_required
@behavior_check_required
def grade_submission(submission_id):
    if session['role'] != 'teacher':
        abort(403)
    db = get_db()
    submission = db.execute('''
        SELECT s.*, a.title as assignment_title, u.username as student_name, a.total_marks
        FROM submissions s
        JOIN assignments a ON s.assignment_id = a.id
        JOIN users u ON s.student_id = u.id
        WHERE s.id = ?
    ''', (submission_id,)).fetchone()
    if not submission:
        abort(404)
    course = db.execute('''
        SELECT c.teacher_id FROM assignments a
        JOIN courses c ON a.course_id = c.id
        WHERE a.id = ?
    ''', (submission['assignment_id'],)).fetchone()
    if course['teacher_id'] != session['user_id']:
        abort(403)
    if request.method == 'POST':
        _validate_csrf()
        grade = request.form['grade']
        feedback = request.form['feedback']
        db.execute("UPDATE submissions SET grade = ?, feedback = ?, graded_at = CURRENT_TIMESTAMP WHERE id = ?",
                   (grade, feedback, submission_id))
        db.commit()
        
        # --- Notify the student ---
        send_notification(
            submission['student_id'],
            f"✅ '{submission['assignment_title']}' graded: {grade}/{submission['total_marks']}",
            url_for('list_assignments')
        )
        
        log_activity(session['user_id'], session['username'], 'assignment_graded', f"Submission ID {submission_id}, grade {grade}")
        flash('Submission graded', 'success')
        return redirect(url_for('list_assignments'))
    return render_template('grade_submission.html', submission=submission)

@app.route('/exams')
@login_required
def list_exams():
    db = get_db()
    if session['role'] == 'teacher':
        exams = db.execute('''
            SELECT e.*, c.course_name 
            FROM exams e JOIN courses c ON e.course_id = c.id
            WHERE c.teacher_id = ?
            ORDER BY e.exam_date ASC, e.start_time ASC
        ''', (session['user_id'],)).fetchall()
    else:
        exams = db.execute('''
            SELECT e.*, c.course_name 
            FROM exams e
            JOIN courses c ON e.course_id = c.id
            JOIN enrollments en ON c.id = en.course_id
            WHERE en.student_id = ?
            ORDER BY e.exam_date ASC, e.start_time ASC
        ''', (session['user_id'],)).fetchall()
    exam_list = []
    for row in exams:
        item = dict(row)
        if session['role'] == 'student':
            item['is_online'] = int(item.get('is_online') or 0)
            start, end, item['window_open'] = get_exam_window(item)
            now = datetime.now()
            item['window_upcoming'] = now < start
            item['window_closed'] = now > end
            item['q_count'] = db.execute(
                "SELECT COUNT(*) as c FROM exam_questions WHERE exam_id = ?",
                (item['id'],),
            ).fetchone()['c']
            att = db.execute(
                "SELECT status, score FROM exam_attempts WHERE exam_id = ? AND student_id = ?",
                (item['id'], session['user_id']),
            ).fetchone()
            item['attempt_status'] = att['status'] if att else None
            item['attempt_score'] = att['score'] if att else None
            item['has_paper'] = bool(item.get('question_paper_url'))
            item['can_take_online'] = (
                item['is_online']
                and item['q_count'] > 0
                and item['window_open']
                and item['attempt_status'] not in ('submitted', 'auto_submitted')
            )
        exam_list.append(item)
    return render_template('exams.html', exams=exam_list, role=session['role'])

@app.route('/exams/create', methods=['GET', 'POST'])
@login_required
@behavior_check_required
def create_exam():
    if session['role'] != 'teacher':
        abort(403)
    db = get_db()
    courses = db.execute("SELECT id, course_name FROM courses WHERE teacher_id = ?", (session['user_id'],)).fetchall()
    if not courses:
        flash('You need to create a course first.', 'warning')
        return redirect(url_for('create_course'))
    if request.method == 'POST':
        _validate_csrf()
        course_id = request.form['course_id']
        title = request.form['title']
        description = request.form['description']
        exam_date = request.form['exam_date']
        start_time = request.form['start_time']
        duration = request.form['duration_minutes']
        total_marks = request.form['total_marks']
        is_online = 1 if request.form.get('is_online') else 0
        db.execute('''
            INSERT INTO exams (course_id, title, description, exam_date, start_time, duration_minutes, total_marks, created_by, is_online)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (course_id, title, description, exam_date, start_time, duration, total_marks, session['user_id'], is_online))
        db.commit()
        
        # --- Send notifications to all enrolled students ---
        course = db.execute("SELECT course_name FROM courses WHERE id = ?", (course_id,)).fetchone()
        course_name = course['course_name'] if course else 'the course'
        
        students = db.execute('''
            SELECT student_id FROM enrollments WHERE course_id = ?
        ''', (course_id,)).fetchall()
        
        for student in students:
            send_notification(
                student['student_id'],
                f"📅 New exam scheduled: {title} in {course_name}",
                url_for('list_exams')
            )
        
        log_activity(session['user_id'], session['username'], 'exam_created', f"Exam: {title} for course ID {course_id}")
        flash('Exam scheduled successfully', 'success')
        return redirect(url_for('list_exams'))
    return render_template('create_exam.html', courses=courses)

@app.route('/exams/<int:exam_id>/upload_paper', methods=['GET', 'POST'])
@login_required
@behavior_check_required
def upload_question_paper(exam_id):
    if session['role'] != 'teacher':
        abort(403)
    db = get_db()
    exam = db.execute('''
        SELECT e.*, c.course_name 
        FROM exams e JOIN courses c ON e.course_id = c.id 
        WHERE e.id = ?
    ''', (exam_id,)).fetchone()
    if not exam:
        abort(404)
    course = db.execute("SELECT teacher_id FROM courses WHERE id = ?", (exam['course_id'],)).fetchone()
    if course['teacher_id'] != session['user_id']:
        abort(403)
    
    if request.method == 'POST':
        file = request.files.get('question_paper')
        if not file:
            flash('No file selected.', 'error')
            return redirect(url_for('upload_question_paper', exam_id=exam_id))
        
        if '.' not in file.filename:
            flash('File must have an extension.', 'error')
            return redirect(url_for('upload_question_paper', exam_id=exam_id))
        
        ext = file.filename.rsplit('.', 1)[1].lower()
        if ext != 'pdf':
            flash('Only PDF files are allowed.', 'error')
            return redirect(url_for('upload_question_paper', exam_id=exam_id))
        
        # --- Magic bytes validation ---
        is_valid, msg = validate_file_type(file, file.filename)
        if not is_valid:
            flash(f'Invalid PDF file: {msg}', 'error')
            return redirect(url_for('upload_question_paper', exam_id=exam_id))
        
        filename = secure_filename(f"exam_{exam_id}_{file.filename}")
        upload_dir = os.path.join('static', 'uploads', 'exam_papers')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)
        db.execute("UPDATE exams SET question_paper_url = ? WHERE id = ?", (file_path.replace('\\', '/'), exam_id))
        db.commit()
        flash('Question paper uploaded successfully.', 'success')
        return redirect(url_for('list_exams'))
    
    return render_template('upload_question_paper.html', exam=exam)

@app.route('/exams/<int:exam_id>/marks', methods=['GET', 'POST'])
@login_required
@behavior_check_required
def exam_marks(exam_id):
    if session['role'] != 'teacher':
        abort(403)
    db = get_db()
    exam = db.execute("SELECT * FROM exams WHERE id = ?", (exam_id,)).fetchone()
    if not exam:
        abort(404)
    course = db.execute("SELECT teacher_id FROM courses WHERE id = ?", (exam['course_id'],)).fetchone()
    if course['teacher_id'] != session['user_id']:
        abort(403)
    rows = db.execute('''
        SELECT u.id, u.username, p.full_name
        FROM enrollments e
        JOIN users u ON e.student_id = u.id
        LEFT JOIN profiles p ON u.id = p.user_id
        WHERE e.course_id = ?
    ''', (exam['course_id'],)).fetchall()
    students = [dict(row) for row in rows]
    if request.method == 'POST':
        for student in students:
            marks = request.form.get(f'marks_{student["id"]}', '')
            if marks:
                try:
                    marks_int = int(marks)
                    db.execute('''
                        INSERT OR REPLACE INTO exam_grades (exam_id, student_id, marks_obtained)
                        VALUES (?, ?, ?)
                    ''', (exam_id, student['id'], marks_int))
                except ValueError:
                    pass
        db.commit()
        flash('Marks saved successfully.', 'success')
        return redirect(url_for('list_exams'))
    for student in students:
        grade = db.execute('''
            SELECT marks_obtained FROM exam_grades
            WHERE exam_id = ? AND student_id = ?
        ''', (exam_id, student['id'])).fetchone()
        student['marks'] = grade['marks_obtained'] if grade else ''
    return render_template('exam_marks.html', exam=exam, students=students)
@app.route('/settings/email', methods=['POST'])
@login_required
def change_email():
    _validate_csrf()
    new_email = request.form.get('new_email', '').strip().lower()
    password = request.form.get('password', '')
    if not new_email or not password:
        flash('All fields required.', 'error')
        return redirect(url_for('settings'))
    
    db = get_db()
    user = db.execute("SELECT password_hash FROM users WHERE id = ?", (session['user_id'],)).fetchone()
    if not bcrypt.checkpw(password.encode('utf-8'), user['password_hash']):
        flash('Incorrect password.', 'error')
        return redirect(url_for('settings'))
    
    existing = db.execute("SELECT id FROM users WHERE email = ? AND id != ?", (new_email, session['user_id'])).fetchone()
    if existing:
        flash('Email already in use.', 'error')
        return redirect(url_for('settings'))
    
    db.execute("UPDATE users SET email = ? WHERE id = ?", (new_email, session['user_id']))
    db.commit()
    log_activity(session['user_id'], session['username'], 'email_changed', f"New email: {new_email}")
    flash('Email updated successfully.', 'success')
    return redirect(url_for('settings'))

@app.route('/settings/totp/toggle', methods=['POST'])
@login_required
def toggle_totp():
    _validate_csrf()
    import pyotp, json, secrets
    db = get_db()
    user = db.execute("SELECT totp_enabled, totp_secret FROM users WHERE id = ?", (session['user_id'],)).fetchone()
    
    if user['totp_enabled']:
        # Disable
        db.execute("UPDATE users SET totp_enabled = 0, totp_secret = NULL, totp_backup_codes = NULL WHERE id = ?",
                   (session['user_id'],))
        flash('Two‑Factor Authentication disabled.', 'success')
    else:
        # Enable – generate secret and backup codes
        secret = pyotp.random_base32()
        plain_codes = [secrets.token_hex(4) for _ in range(10)]
        hashed_codes = [bcrypt.hashpw(c.encode('utf-8'), bcrypt.gensalt()).decode('utf-8') for c in plain_codes]
        db.execute("""
            UPDATE users 
            SET totp_secret = ?, totp_enabled = 1, totp_backup_codes = ?
            WHERE id = ?
        """, (secret, json.dumps(hashed_codes), session['user_id']))
        db.commit()
        session['new_backup_codes'] = plain_codes
        flash('TOTP enabled! Backup codes generated. Save them now.', 'success')
    
    db.commit()
    return redirect(url_for('settings'))

@app.route('/settings/totp/regenerate', methods=['POST'])
@login_required
def regenerate_totp_codes():
    _validate_csrf()
    import json, secrets
    db = get_db()
    user = db.execute("SELECT totp_enabled FROM users WHERE id = ?", (session['user_id'],)).fetchone()
    if not user['totp_enabled']:
        flash('TOTP is not enabled.', 'error')
        return redirect(url_for('settings'))
    
    plain_codes = [secrets.token_hex(4) for _ in range(10)]
    hashed_codes = [bcrypt.hashpw(c.encode('utf-8'), bcrypt.gensalt()).decode('utf-8') for c in plain_codes]
    db.execute("UPDATE users SET totp_backup_codes = ? WHERE id = ?", (json.dumps(hashed_codes), session['user_id']))
    db.commit()
    session['new_backup_codes'] = plain_codes
    flash('New backup codes generated!', 'success')
    return redirect(url_for('settings'))

@app.template_test('search')
def test_search(value, substring):
    """Return True if substring is in value (case‑sensitive)."""
    return substring in (value or '')

@app.route('/settings')
@login_required
def settings():
    db = get_db()
    user = db.execute("""
        SELECT email, password_changed_at, totp_enabled, totp_secret
        FROM users WHERE id = ?
    """, (session['user_id'],)).fetchone()
    return render_template('settings.html', user=user)

@app.route('/attendance')
@login_required
def view_attendance():
    db = get_db()
    user_role = session['role']
    user_id = session['user_id']

    # --- Admin view ---
    if user_role == 'admin':
        student_id = request.args.get('student_id', type=int)
        if student_id:
            # Admin viewing a specific student
            student = db.execute("SELECT id, username FROM users WHERE id = ? AND role = 'student'", (student_id,)).fetchone()
            if not student:
                flash('Student not found.', 'error')
                return redirect(url_for('view_attendance'))
            # Get enrollments and attendance for that student
            enrollments = db.execute('''
                SELECT c.id as course_id, c.course_code, c.course_name
                FROM enrollments e JOIN courses c ON e.course_id = c.id
                WHERE e.student_id = ?
            ''', (student_id,)).fetchall()
            attendance_data = _build_attendance_data(student_id, enrollments)
            return render_template('admin_student_attendance.html', attendance_data=attendance_data, student=student)
        else:
            # Admin sees a list of all students (with search)
            search = request.args.get('search', '').strip()
            if search:
                students = db.execute('''
                    SELECT id, username, email
                    FROM users
                    WHERE role = 'student' AND (username LIKE ? OR email LIKE ?)
                    ORDER BY username
                ''', (f'%{search}%', f'%{search}%')).fetchall()
            else:
                students = db.execute('''
                    SELECT id, username, email
                    FROM users
                    WHERE role = 'student'
                    ORDER BY username
                ''').fetchall()
            return render_template('admin_attendance_list.html', students=students, search=search)

    # --- Student view ---
    if user_role == 'student':
        enrollments = db.execute('''
            SELECT c.id as course_id, c.course_code, c.course_name
            FROM enrollments e JOIN courses c ON e.course_id = c.id
            WHERE e.student_id = ?
        ''', (user_id,)).fetchall()
        attendance_data = _build_attendance_data(user_id, enrollments)
        return render_template('student_attendance.html', attendance_data=attendance_data)

    # --- Teacher or other roles ---
    flash('Only students and admins can view attendance summaries.', 'error')
    return redirect(url_for('dashboard'))

def _build_attendance_data(student_id, enrollments):
    """Helper to build attendance data for a student."""
    db = get_db()
    attendance_data = []
    for enr in enrollments:
        stats = db.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'present' THEN 1 ELSE 0 END) as present,
                SUM(CASE WHEN status = 'late' THEN 1 ELSE 0 END) as late,
                SUM(CASE WHEN status = 'absent' THEN 1 ELSE 0 END) as absent
            FROM attendance
            WHERE course_id = ? AND student_id = ?
        ''', (enr['course_id'], student_id)).fetchone()
        total = stats['total'] or 0
        present = stats['present'] or 0
        late = stats['late'] or 0
        percentage = round((present + late * 0.5) / total * 100, 1) if total > 0 else 0
        attendance_data.append({
            'course': enr,
            'total': total,
            'present': present,
            'late': late,
            'absent': stats['absent'] or 0,
            'percentage': percentage
        })
    return attendance_data

@app.route('/courses/<int:course_id>/attendance', methods=['GET', 'POST'])
@login_required
def mark_attendance(course_id):
    if session['role'] != 'teacher':
        abort(403)
    db = get_db()
    course = db.execute("SELECT * FROM courses WHERE id = ? AND teacher_id = ?", (course_id, session['user_id'])).fetchone()
    if not course:
        abort(403)
    students = db.execute('''
        SELECT u.id, u.username, p.full_name
        FROM enrollments e JOIN users u ON e.student_id = u.id
        LEFT JOIN profiles p ON u.id = p.user_id
        WHERE e.course_id = ?
    ''', (course_id,)).fetchall()
    if request.method == 'POST':
        _validate_csrf()
        date_str = request.form['date']
        try:
            attendance_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except:
            flash('Invalid date format.', 'error')
            return redirect(url_for('mark_attendance', course_id=course_id))
        for student in students:
            status = request.form.get(f'status_{student["id"]}', 'absent')
            if status not in ('present', 'absent', 'late'):
                status = 'absent'
            db.execute('''
                INSERT OR REPLACE INTO attendance (course_id, student_id, date, status, marked_by)
                VALUES (?, ?, ?, ?, ?)
            ''', (course_id, student['id'], attendance_date, status, session['user_id']))
        db.commit()
        flash('Attendance recorded.', 'success')
        return redirect(url_for('mark_attendance', course_id=course_id))
    today = datetime.now().date()
    existing_att = {}
    for student in students:
        rec = db.execute('''
            SELECT status FROM attendance
            WHERE course_id = ? AND student_id = ? AND date = ?
        ''', (course_id, student['id'], today)).fetchone()
        existing_att[student['id']] = rec['status'] if rec else ''
    return render_template('attendance.html', course=course, students=students, today=today, existing_att=existing_att)

@app.route('/grades')
@login_required
def grades_summary():
    if session['role'] != 'student':
        flash('Only students can view grades.', 'error')
        return redirect(url_for('dashboard'))
    db = get_db()
    courses = db.execute('''
        SELECT c.id, c.course_code, c.course_name
        FROM enrollments e JOIN courses c ON e.course_id = c.id
        WHERE e.student_id = ?
    ''', (session['user_id'],)).fetchall()
    grade_data = []
    total_obtained = 0
    total_possible = 0
    for course in courses:
        assignments = db.execute('''
            SELECT a.id, a.title, a.total_marks, s.grade
            FROM assignments a
            LEFT JOIN submissions s ON a.id = s.assignment_id AND s.student_id = ?
            WHERE a.course_id = ?
        ''', (session['user_id'], course['id'])).fetchall()
        course_obtained = 0
        course_possible = 0
        for a in assignments:
            if a['grade'] is not None:
                course_obtained += a['grade']
                course_possible += a['total_marks']
        percentage = (course_obtained / course_possible * 100) if course_possible > 0 else 0
        if percentage >= 90: letter = 'A'
        elif percentage >= 80: letter = 'B'
        elif percentage >= 70: letter = 'C'
        elif percentage >= 60: letter = 'D'
        else: letter = 'F'
        gpa_points = 4.0 if percentage >= 90 else (3.0 if percentage >= 80 else (2.0 if percentage >= 70 else (1.0 if percentage >= 60 else 0.0)))
        grade_data.append({
            'course': course,
            'obtained': course_obtained,
            'possible': course_possible,
            'percentage': round(percentage, 1),
            'letter': letter,
            'gpa_points': gpa_points
        })
        total_obtained += course_obtained
        total_possible += course_possible
    overall_percentage = (total_obtained / total_possible * 100) if total_possible > 0 else 0
    overall_gpa = sum([g['gpa_points'] for g in grade_data]) / len(grade_data) if grade_data else 0
    return render_template('grades.html', grade_data=grade_data, overall_percentage=round(overall_percentage,1), overall_gpa=round(overall_gpa,2))

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data = request.get_json()
    user_message = data.get('message', '').strip()
    scan_and_log_attack_patterns(endpoint="chat", block=False)
    if not user_message:
        return jsonify({'response': 'Please enter a question.'})
    if len(user_message) > 500:
        return jsonify({'response': 'Your question is too long (max 500 characters).'})
    user_message = user_message.replace('<', '&lt;').replace('>', '&gt;')
    user_role = session['role']
    msg = user_message.lower()
    general_responses = {
        'computer': 'A computer is an electronic device that processes data and performs tasks according to instructions.',
        'python': 'Python is a popular high‑level programming language.',
        'ai': 'AI (Artificial Intelligence) is the simulation of human intelligence in machines.',
        'hello': 'Hello! How can I help you today?',
        'hi': 'Hi there! Ask me anything about the portal or general knowledge.',
    }
    for key, answer in general_responses.items():
        if key in msg:
            return jsonify({'response': answer})
    if 'attendance' in msg:
        if user_role == 'student':
            return jsonify({'response': 'Check your attendance on the Attendance page. Contact your teacher if you find discrepancies.'})
        else:
            return jsonify({'response': 'You can mark attendance from the course page.'})
    if 'exam' in msg:
        return jsonify({'response': 'Exam schedules are posted under the Exams menu.'})
    if 'assignment' in msg:
        return jsonify({'response': 'Assignments are listed under each course.'})
    if 'grade' in msg or 'marks' in msg:
        return jsonify({'response': 'Your grades are available on the course page after the teacher publishes them.'})
    if 'course' in msg:
        return jsonify({'response': 'Browse all courses from the Courses page.'})
    if 'routine' in msg or 'schedule' in msg:
        return jsonify({'response': 'Your class routine is available under the Routine menu.'})
    if 'notice' in msg:
        return jsonify({'response': 'All official notices are posted on the Notices page.'})
    if 'profile' in msg:
        return jsonify({'response': 'Update your personal information from the Profile page.'})
    if 'password' in msg:
        return jsonify({'response': 'Change your password from the Change Password page.'})
    if 'help' in msg:
        return jsonify({'response': 'I can help with: attendance, exams, assignments, grades, courses, routine, notices, profile, password, login issues, OTP, and general knowledge.'})
    return jsonify({'response': "I'm EduShield AI. I can help with portal features and general knowledge."})

# ---------- Admin & IDS routes (remaining) ----------
@app.route('/ids_dashboard')
@login_required
@admin_required
def ids_dashboard():
    db = get_db()
    trend_rows = db.execute("""
        SELECT date(login_time) as date, 
               COUNT(*) as total,
               SUM(CASE WHEN status='blocked' OR action IN ('block','otp_required') THEN 1 ELSE 0 END) as attacks
        FROM login_logs
        WHERE login_time > datetime('now','-7 days')
        GROUP BY date(login_time) ORDER BY date
    """).fetchall()
    trend = [dict(row) for row in trend_rows]
    risk_rows = db.execute("SELECT risk_score, COUNT(*) as cnt FROM login_logs GROUP BY risk_score ORDER BY risk_score").fetchall()
    risk_dist = [dict(row) for row in risk_rows]
    blocked_rows = db.execute("""
        SELECT ip_address, COUNT(*) as attempts
        FROM ip_failed_attempts
        WHERE attempt_time > datetime('now','-1 day')
        GROUP BY ip_address ORDER BY attempts DESC LIMIT 10
    """).fetchall()
    blocked_ips = [dict(row) for row in blocked_rows]
    otp_rows = db.execute("""
        SELECT date(login_time) as date, COUNT(*) as otp_cnt
        FROM login_logs
        WHERE action='otp_required' AND login_time > datetime('now','-7 days')
        GROUP BY date(login_time) ORDER BY date
    """).fetchall()
    otp_freq = [dict(row) for row in otp_rows]
    top_rows = db.execute("""
        SELECT ip_address, COUNT(*) as cnt
        FROM login_logs
        WHERE (status='blocked' OR action IN ('block','block_ip_bruteforce')) 
          AND login_time > datetime('now','-7 days')
        GROUP BY ip_address ORDER BY cnt DESC LIMIT 10
    """).fetchall()
    top_blocked_ips = [dict(row) for row in top_rows]
    alert_rows = db.execute("""
        SELECT l.*, u.username
        FROM login_logs l LEFT JOIN users u ON l.user_id = u.id
        WHERE l.status IN ('blocked','otp_sent') OR l.action IN ('block','otp_required','block_ip_bruteforce')
        ORDER BY l.login_time DESC LIMIT 50
    """).fetchall()
    alerts = [dict(row) for row in alert_rows]
    now = datetime.now()
    return render_template('ids_dashboard.html',
                         trend=trend,
                         risk_dist=risk_dist,
                         blocked_ips=blocked_ips,
                         otp_freq=otp_freq,
                         top_blocked_ips=top_blocked_ips,
                         alerts=alerts,
                         now=now)

@app.route('/admin')
@login_required
@admin_required
@behavior_check_required
def admin_dashboard():
    db = get_db()
    users = db.execute('''
        SELECT id, username, email, role, is_admin, created_at,
               is_locked, locked_at, is_banned, is_approved, device_fingerprint
        FROM users
    ''').fetchall()
    users_list = []
    for user in users:
        user_dict = dict(user)
        if user_dict.get('created_at'):
            user_dict['created_at_local'] = utc_to_nepal_time(user_dict['created_at'])
        else:
            user_dict['created_at_local'] = None
        users_list.append(user_dict)
    logs = db.execute('''
        SELECT l.*, u.username FROM login_logs l LEFT JOIN users u ON l.user_id = u.id 
        ORDER BY l.login_time DESC LIMIT 100
    ''').fetchall()
    suspicious = db.execute('''
        SELECT l.*, u.username FROM login_logs l LEFT JOIN users u ON l.user_id = u.id 
        WHERE l.risk_score >= 3 OR l.action IN ('block', 'otp_required', 'block_ip_bruteforce', 'unknown_user')
        ORDER BY l.login_time DESC LIMIT 50
    ''').fetchall()
    fifteen_min_ago = datetime.now() - timedelta(minutes=15)
    blocked_users = db.execute('''
        SELECT u.id, u.username, u.email, COUNT(fa.id) as failed_count
        FROM users u JOIN failed_attempts fa ON u.id = fa.user_id
        WHERE fa.attempt_time > ? GROUP BY u.id HAVING COUNT(fa.id) >= 5
    ''', (fifteen_min_ago,)).fetchall()
    locked_users = db.execute("SELECT id, username, email, locked_at FROM users WHERE is_locked=1").fetchall()
    total_courses = db.execute("SELECT COUNT(*) as count FROM courses").fetchone()['count']
    total_notices = db.execute("SELECT COUNT(*) as count FROM notices").fetchone()['count']
    stats = db.execute('''
        SELECT 
            COUNT(*) as total_logins,
            SUM(CASE WHEN status='blocked' OR action='block' THEN 1 ELSE 0 END) as blocked,
            SUM(CASE WHEN action='otp_required' THEN 1 ELSE 0 END) as otp_sent,
            SUM(CASE WHEN alert_sent=1 THEN 1 ELSE 0 END) as alerts_sent
        FROM login_logs
    ''').fetchone()
    return render_template('admin.html', 
                         users=users, 
                         logs=logs, 
                         suspicious=suspicious,
                         blocked_users=blocked_users,
                         locked_users=locked_users,
                         total_courses=total_courses,
                         total_notices=total_notices,
                         stats=stats)

@app.route('/admin/fingerprint_history/<int:user_id>')
@login_required
@admin_required
def fingerprint_history(user_id):
    db = get_db()
    logs = db.execute('''
        SELECT login_time, ip_address, device, risk_factors, status
        FROM login_logs
        WHERE user_id = ?
        ORDER BY login_time DESC
        LIMIT 50
    ''', (user_id,)).fetchall()
    user = db.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        abort(404)
    return render_template('fingerprint_history.html', logs=logs, user=user)

@app.route('/admin/unlock/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def unlock_user(user_id):
    _validate_csrf()
    db = get_db()
    user = db.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('admin_dashboard'))
    
    db.execute("DELETE FROM failed_attempts WHERE user_id = ?", (user_id,))
    db.execute("UPDATE users SET is_locked = 0, locked_at = NULL WHERE id = ?", (user_id,))
    db.commit()
    
    log_admin_action(session['user_id'], session['username'], 'unlock_user', user_id, user['username'], f"Unlocked user {user['username']}")
    flash(f"User {user_id} unlocked successfully.", "success")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/toggle_block/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def toggle_block(user_id):
    _validate_csrf()
    db = get_db()
    user = db.execute("SELECT username, is_banned FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('admin_dashboard'))
    new_status = 0 if user['is_banned'] else 1
    db.execute("UPDATE users SET is_banned = ? WHERE id = ?", (new_status, user_id))
    db.commit()
    action = 'block_user' if new_status else 'unblock_user'
    log_admin_action(session['user_id'], session['username'], action, user_id, user['username'], f"{action.capitalize()} user {user['username']}")
    flash(f"User {'blocked' if new_status else 'unblocked'} successfully.", "success")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/toggle_ai_security', methods=['POST'])
@login_required
@admin_required
def toggle_ai_security():
    data = request.get_json()
    enabled = data.get('enabled', False)
    set_ai_security_status(enabled)
    log_admin_action(session['user_id'], session['username'], 'toggle_ai_security', details=f"Set AI Security Engine to {'ON' if enabled else 'OFF'}")
    log_activity(session['user_id'], session['username'], 'toggle_ai_security', f"Set AI Security Engine to {'ON' if enabled else 'OFF'}")
    return jsonify({'success': True, 'status': enabled})

@app.route('/admin/ai_metrics')
@login_required
@admin_required
def ai_metrics():
    return jsonify(get_ai_metrics())

@app.route('/admin/approve_teacher/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def approve_teacher(user_id):
    _validate_csrf()
    db = get_db()
    user = db.execute("SELECT email, username FROM users WHERE id = ? AND role = 'teacher'", (user_id,)).fetchone()
    if not user:
        flash('Teacher not found.', 'error')
        return redirect(url_for('pending_teachers'))
    db.execute("UPDATE users SET is_approved = 1 WHERE id = ?", (user_id,))
    db.commit()
    send_user_alert(user['email'], "Your teacher account has been approved", 
                    f"Dear {user['username']},\n\nYour teacher account has been approved. You can now log in.\n\nLogin here: {url_for('login', _external=True)}")
    log_activity(session['user_id'], session['username'], 'teacher_approved', f"Approved teacher {user['username']} (ID: {user_id})")
    flash(f'Teacher {user["username"]} approved successfully.', 'success')
    return redirect(url_for('pending_teachers'))

def _parse_shap_json(raw):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None

def _recompute_shap_for_log(log_row):
    if ensemble_risk is None or not log_row.get('user_id'):
        return None
    try:
        X, _ = build_ml_feature_frame(log_row['user_id'], log_row['ip_address'] or '', log_row['device'] or '')
        return shap_explain.explain_risk(ensemble_risk, X)
    except Exception:
        return None

@app.route('/admin/risk_explanations')
@login_required
@admin_required
def risk_explanations():
    db = get_db()
    rows = db.execute('''
        SELECT l.*, u.username
        FROM login_logs l
        LEFT JOIN users u ON l.user_id = u.id
        WHERE l.risk_score > 0 OR l.shap_json IS NOT NULL OR l.risk_factors IS NOT NULL
        ORDER BY l.login_time DESC
        LIMIT 100
    ''').fetchall()
    logs = []
    shap_payloads = []
    for row in rows:
        entry = dict(row)
        shap_data = _parse_shap_json(entry.get('shap_json'))
        if not shap_data and entry.get('user_id'):
            shap_data = _recompute_shap_for_log(entry)
        entry['shap'] = shap_data
        entry['shap_summary'] = (shap_data or {}).get('summary') or entry.get('risk_factors') or '—'
        logs.append(entry)
        if shap_data:
            shap_payloads.append(shap_data)
    aggregate_labels, aggregate_values = shap_explain.aggregate_shap_contributions(shap_payloads)
    return render_template(
        'risk_explanations.html',
        logs=logs,
        aggregate_labels=aggregate_labels,
        aggregate_values=aggregate_values,
        shap_available=ensemble_risk is not None,
    )

@app.route('/admin/risk_explanations/<int:log_id>')
@login_required
@admin_required
def risk_explanation_detail(log_id):
    db = get_db()
    log_row = db.execute('''
        SELECT l.*, u.username
        FROM login_logs l
        LEFT JOIN users u ON l.user_id = u.id
        WHERE l.id = ?
    ''', (log_id,)).fetchone()
    if not log_row:
        abort(404)
    entry = dict(log_row)
    shap_data = _parse_shap_json(entry.get('shap_json'))
    if not shap_data:
        # Try to recompute
        shap_data = _recompute_shap_for_log(entry)
        # If recomputation succeeded, save it to the database
        if shap_data:
            shap_json_str = json.dumps(shap_data)
            db.execute("UPDATE login_logs SET shap_json = ? WHERE id = ?", (shap_json_str, log_id))
            db.commit()
            entry['shap_json'] = shap_json_str
    chart_b64 = shap_explain.shap_bar_chart_base64(shap_data) if shap_data else None
    return render_template('risk_explanation_detail.html', log=entry, shap=shap_data, chart_b64=chart_b64)

@app.route('/admin/security_heatmap')
@login_required
@admin_required
def security_heatmap():
    return render_template('security_heatmap.html')

@app.route('/admin/api/security_map_data')
@login_required
@admin_required
def security_map_data():
    db = get_db()
    rows = db.execute('''
        SELECT l.id, l.geo_lat, l.geo_lon, l.risk_score, l.login_time, l.action,
               l.geo_country, l.status, u.username, u.role
        FROM login_logs l
        LEFT JOIN users u ON l.user_id = u.id
        WHERE l.geo_lat IS NOT NULL AND l.geo_lon IS NOT NULL
        ORDER BY l.login_time DESC
        LIMIT 500
    ''').fetchall()
    points = []
    for row in rows:
        r = dict(row)
        if r.get('login_time'):
            r['login_time'] = r['login_time'].isoformat() if hasattr(r['login_time'], 'isoformat') else str(r['login_time'])
        points.append(r)
    return jsonify(points)

@app.route('/admin/attack_logs')
@login_required
@admin_required
def attack_logs():
    db = get_db()
    logs = db.execute('''
        SELECT a.*, u.username 
        FROM attack_patterns a
        LEFT JOIN users u ON a.user_id = u.id
        ORDER BY a.event_time DESC
        LIMIT 200
    ''').fetchall()
    logs_list = []
    for row in logs:
        row_dict = dict(row)
        if row_dict.get('event_time'):
            row_dict['event_time_local'] = utc_to_nepal_time(row_dict['event_time'])
        logs_list.append(row_dict)
    return render_template('attack_logs.html', logs=logs_list)

@app.route('/admin/activity_logs')
@login_required
@admin_required
def activity_logs():
    db = get_db()
    logs = db.execute("SELECT * FROM user_activity_logs ORDER BY timestamp DESC LIMIT 200").fetchall()
    return render_template('activity_logs.html', logs=logs)

@app.route('/admin/export_logs')
@login_required
@admin_required
def export_logs():
    db = get_db()
    rows = db.execute('''
        SELECT l.login_time, u.username, l.ip_address, l.device, l.status, l.risk_score, l.action, l.alert_sent, l.risk_factors, l.geo_country, l.geo_lat, l.geo_lon
        FROM login_logs l LEFT JOIN users u ON l.user_id = u.id
        ORDER BY l.login_time DESC
    ''').fetchall()
    import csv
    from io import StringIO
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['time','username','ip','device','status','risk_score','action','alert_sent','risk_factors','geo_country','geo_lat','geo_lon'])
    for row in rows:
        cw.writerow([row['login_time'], row['username'], row['ip_address'], row['device'], row['status'], row['risk_score'], row['action'], row['alert_sent'], row['risk_factors'], row['geo_country'], row['geo_lat'], row['geo_lon']])
    output = si.getvalue()
    return Response(output, mimetype='text/csv', headers={'Content-Disposition': 'attachment;filename=login_logs.csv'})
@app.route('/admin/trusted_networks')
@login_required
@admin_required
def trusted_networks():
    db = get_db()
    networks_rows = db.execute("SELECT * FROM trusted_networks ORDER BY created_at DESC").fetchall()
    networks = []
    for row in networks_rows:
        net = dict(row)
        if net.get('created_at'):
            net['created_at_local'] = utc_to_nepal_time(net['created_at'])
        else:
            net['created_at_local'] = None
        networks.append(net)
    return render_template('trusted_networks.html', networks=networks)


@app.route('/admin/trusted_networks/add', methods=['POST'])
@login_required
@admin_required
def add_trusted_network():
    _validate_csrf()
    network = request.form.get('network', '').strip()
    description = request.form.get('description', '').strip()
    if not network:
        flash('Network CIDR or IP required', 'error')
        return redirect(url_for('trusted_networks'))
    try:
        ipaddress.ip_network(network, strict=False)
    except ValueError:
        flash('Invalid network format. Use CIDR (e.g., 192.168.0.0/24) or single IP.', 'error')
        return redirect(url_for('trusted_networks'))
    db = get_db()
    db.execute("INSERT INTO trusted_networks (network, description) VALUES (?, ?)", (network, description))
    db.commit()
    log_admin_action(session['user_id'], session['username'], 'add_trusted_network', details=f"Added network {network} ({description})")
    log_activity(session['user_id'], session['username'], 'trusted_network_added', f"Added {network}")
    flash('Trusted network added successfully', 'success')
    return redirect(url_for('trusted_networks'))

@app.route('/admin/trusted_networks/delete/<int:network_id>', methods=['POST'])
@login_required
@admin_required
def delete_trusted_network(network_id):
    _validate_csrf()
    db = get_db()
    net = db.execute("SELECT network, description FROM trusted_networks WHERE id = ?", (network_id,)).fetchone()
    if net:
        log_admin_action(session['user_id'], session['username'], 'delete_trusted_network', details=f"Deleted network {net['network']} ({net['description']})")
    db.execute("DELETE FROM trusted_networks WHERE id = ?", (network_id,))
    db.commit()
    log_activity(session['user_id'], session['username'], 'trusted_network_deleted', f"Deleted ID {network_id}")
    flash('Trusted network removed', 'success')
    return redirect(url_for('trusted_networks'))

@app.route('/admin/behavior_analytics')
@login_required
@admin_required
def behavior_analytics():
    db = get_db()
    recent_logs = db.execute('''
        SELECT b.*, u.username
        FROM user_behavior_logs b
        JOIN users u ON b.user_id = u.id
        ORDER BY b.created_at DESC LIMIT 200
    ''').fetchall()
    logs_list = []
    for row in recent_logs:
        log_dict = dict(row)
        if log_dict.get('created_at'):
            log_dict['created_at_local'] = utc_to_nepal_time(log_dict['created_at'])
        else:
            log_dict['created_at_local'] = None
        logs_list.append(log_dict)
    high_risk_users = db.execute('''
        SELECT user_id, COUNT(*) as high_risk_count
        FROM user_behavior_logs
        WHERE time_spent < 1 OR url LIKE '%/admin%'
        GROUP BY user_id
        HAVING high_risk_count > 10
    ''').fetchall()
    return render_template('behavior_analytics.html', logs=logs_list, high_risk_users=high_risk_users)

def utc_to_nepal_time(utc_dt):
    if utc_dt is None:
        return None
    if isinstance(utc_dt, str):
        try:
            utc_dt = datetime.fromisoformat(utc_dt)
        except:
            return utc_dt
    return utc_dt + timedelta(hours=5, minutes=45)

@app.route('/admin/report')
@login_required
@admin_required
def generate_report():
    db = get_db()
    total_logins = db.execute("SELECT COUNT(*) as count FROM login_logs").fetchone()['count']
    blocked_logins = db.execute("SELECT COUNT(*) as count FROM login_logs WHERE status='blocked' OR action='block'").fetchone()['count']
    otp_sent = db.execute("SELECT COUNT(*) as count FROM login_logs WHERE action='otp_required'").fetchone()['count']
    locked_users = db.execute("SELECT COUNT(*) as count FROM users WHERE is_locked=1").fetchone()['count']
    banned_users = db.execute("SELECT COUNT(*) as count FROM users WHERE is_banned=1").fetchone()['count']
    risk_factors = db.execute("SELECT risk_factors FROM login_logs WHERE risk_factors IS NOT NULL AND risk_factors != '' LIMIT 100").fetchall()
    factor_counts = {}
    for row in risk_factors:
        for factor in row['risk_factors'].split(','):
            f = factor.strip()
            if f:
                factor_counts[f] = factor_counts.get(f, 0) + 1
    top_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    attacks = db.execute("SELECT pattern_type, COUNT(*) as cnt FROM attack_patterns GROUP BY pattern_type").fetchall()
    locked_accounts = db.execute("SELECT username, locked_at FROM users WHERE is_locked=1 ORDER BY locked_at DESC LIMIT 10").fetchall()
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title="Security Report")
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    story = []
    story.append(Paragraph("Academic Portal Security Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    story.append(Spacer(1, 0.3*inch))
    data = [["Metric", "Value"],
            ["Total Logins", str(total_logins)],
            ["Blocked Logins", str(blocked_logins)],
            ["OTP Requests", str(otp_sent)],
            ["Locked Accounts", str(locked_users)],
            ["Banned Users", str(banned_users)]]
    table = Table(data, colWidths=[3*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Top 5 Risk Factors", heading_style))
    if top_factors:
        factor_data = [["Risk Factor", "Occurrences"]] + [[f, str(c)] for f, c in top_factors]
        factor_table = Table(factor_data, colWidths=[3.5*inch, 1*inch])
        factor_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
        story.append(factor_table)
    else:
        story.append(Paragraph("No risk factors recorded.", normal_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Detected XSS/SQLi Attacks", heading_style))
    if attacks:
        attack_data = [["Type", "Count"]] + [[row['pattern_type'], str(row['cnt'])] for row in attacks]
        attack_table = Table(attack_data, colWidths=[2*inch, 1*inch])
        attack_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
        story.append(attack_table)
    else:
        story.append(Paragraph("No attacks logged.", normal_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Locked Accounts (Require Admin Unlock)", heading_style))
    if locked_accounts:
        locked_data = [["Username", "Locked At"]] + [[row['username'], row['locked_at']] for row in locked_accounts]
        locked_table = Table(locked_data, colWidths=[2*inch, 2*inch])
        locked_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
        story.append(locked_table)
    else:
        story.append(Paragraph("No locked accounts.", normal_style))
    doc.build(story)
    buffer.seek(0)
    return Response(buffer, mimetype='application/pdf', headers={'Content-Disposition': 'attachment;filename=security_report.pdf'})
@app.route('/admin/audit_logs')
@login_required
@admin_required
def admin_audit_logs():
    db = get_db()
    logs = db.execute('''
        SELECT a.*, u.username as admin_name 
        FROM admin_audit_logs a
        LEFT JOIN users u ON a.admin_id = u.id
        ORDER BY a.timestamp DESC
        LIMIT 200
    ''').fetchall()
    return render_template('admin_audit_logs.html', logs=logs)

@app.route('/admin/bias_analysis')
@login_required
@admin_required
def bias_analysis():
    db = get_db()
    
    # --- Helper: classify ISP from IP ---
    def classify_isp(ip):
        if not ip:
            return 'Other'
        if ip.startswith('192.168'):
            return 'College_WiFi'
        if ip.startswith('202.70'):
            return 'WorldLink'
        if ip.startswith('103.1'):
            return 'Ncell'
        if ip.startswith('27.34'):
            return 'NTC'
        return 'Other'
    
    # Fetch all login logs for students (teachers/admins might have different patterns)
    logs = db.execute('''
        SELECT l.ip_address, l.device, l.status, l.action, l.risk_score,
               strftime('%H', l.login_time) as hour
        FROM login_logs l
        JOIN users u ON l.user_id = u.id
        WHERE u.role = 'student'
    ''').fetchall()
    
    # Convert to list of dicts for processing
    data = [dict(row) for row in logs]
    
    # Group by ISP
    isp_groups = {}
    for row in data:
        isp = classify_isp(row['ip_address'])
        isp_groups.setdefault(isp, {'total': 0, 'alert': 0})
        isp_groups[isp]['total'] += 1
        if row['status'] == 'blocked' or row['action'] == 'otp_required':
            isp_groups[isp]['alert'] += 1
    
    # Group by device type (simplify: extract from device string)
    device_groups = {}
    for row in data:
        device = row['device'].lower() if row['device'] else 'unknown'
        if 'mobile' in device or 'iphone' in device or 'android' in device:
            dev = 'mobile'
        elif 'tablet' in device or 'ipad' in device:
            dev = 'tablet'
        elif 'desktop' in device or 'laptop' in device:
            dev = 'desktop'
        else:
            dev = 'other'
        device_groups.setdefault(dev, {'total': 0, 'alert': 0})
        device_groups[dev]['total'] += 1
        if row['status'] == 'blocked' or row['action'] == 'otp_required':
            device_groups[dev]['alert'] += 1
    
    # Group by hour (normal 8-20 vs unusual)
    hour_groups = {'normal': {'total': 0, 'alert': 0}, 'unusual': {'total': 0, 'alert': 0}}
    for row in data:
        hour = int(row['hour']) if row['hour'] else 12
        if 8 <= hour <= 20:
            hour_groups['normal']['total'] += 1
            if row['status'] == 'blocked' or row['action'] == 'otp_required':
                hour_groups['normal']['alert'] += 1
        else:
            hour_groups['unusual']['total'] += 1
            if row['status'] == 'blocked' or row['action'] == 'otp_required':
                hour_groups['unusual']['alert'] += 1
    
    # Prepare data for charts (alert rate %)
    isp_labels = []
    isp_rates = []
    for isp, vals in isp_groups.items():
        if vals['total'] > 0:
            rate = round(100 * vals['alert'] / vals['total'], 1)
            isp_labels.append(isp)
            isp_rates.append(rate)
    
    device_labels = []
    device_rates = []
    for dev, vals in device_groups.items():
        if vals['total'] > 0:
            rate = round(100 * vals['alert'] / vals['total'], 1)
            device_labels.append(dev)
            device_rates.append(rate)
    
    hour_labels = ['Normal (8-20)', 'Unusual (21-7)']
    hour_rates = [
        round(100 * hour_groups['normal']['alert'] / max(1, hour_groups['normal']['total']), 1),
        round(100 * hour_groups['unusual']['alert'] / max(1, hour_groups['unusual']['total']), 1)
    ]
    
    # Chi-square test for ISP groups (to show p-value)
    from scipy.stats import chi2_contingency
    import numpy as np
    
    def chi_square_pvalue(groups_dict):
        # groups_dict: {group: {'total': t, 'alert': a}}
        observed = []
        for g in groups_dict:
            a = groups_dict[g]['alert']
            na = groups_dict[g]['total'] - a
            observed.append([a, na])
        if len(observed) < 2:
            return None
        chi2, p, dof, expected = chi2_contingency(observed)
        return round(p, 4)
    
    isp_pvalue = chi_square_pvalue(isp_groups) if len(isp_groups) >= 2 else None
    device_pvalue = chi_square_pvalue(device_groups) if len(device_groups) >= 2 else None
    # For hour groups (2x2 contingency)
    hour_obs = [[hour_groups['normal']['alert'], hour_groups['normal']['total'] - hour_groups['normal']['alert']],
                [hour_groups['unusual']['alert'], hour_groups['unusual']['total'] - hour_groups['unusual']['alert']]]
    if hour_groups['normal']['total'] > 0 and hour_groups['unusual']['total'] > 0:
        _, hour_p, _, _ = chi2_contingency(hour_obs)
        hour_pvalue = round(hour_p, 4)
    else:
        hour_pvalue = None
    
    return render_template('bias_analysis.html',
                           isp_labels=isp_labels,
                           isp_rates=isp_rates,
                           device_labels=device_labels,
                           device_rates=device_rates,
                           hour_labels=hour_labels,
                           hour_rates=hour_rates,
                           isp_pvalue=isp_pvalue,
                           device_pvalue=device_pvalue,
                           hour_pvalue=hour_pvalue,
                           isp_totals=isp_groups,
                           device_totals=device_groups,
                           hour_totals=hour_groups)
                           

@app.route('/security')
@login_required
def security_center():
    db = get_db()
    
    # 1. Login history (existing)
    logs = db.execute('''
        SELECT id, ip_address, device, login_time, status, risk_score, action, risk_factors, geo_country
        FROM login_logs
        WHERE user_id = ?
        ORDER BY login_time DESC
        LIMIT 50
    ''', (session['user_id'],)).fetchall()
    login_history = []
    for row in logs:
        entry = dict(row)
        entry['login_time_local'] = utc_to_nepal_time(entry.get('login_time'))
        login_history.append(entry)

    # 2. Active Sessions (NEW)
    # Get the current session token to identify the current device
    user = db.execute("SELECT session_token FROM users WHERE id = ?", (session['user_id'],)).fetchone()
    current_token = user['session_token'] if user else None

    # Fetch recent successful logins (last 7 days, or last 10 sessions)
    sessions = db.execute('''
        SELECT id, ip_address, device, login_time, geo_country
        FROM login_logs
        WHERE user_id = ? AND status = 'success'
        ORDER BY login_time DESC
        LIMIT 10
    ''', (session['user_id'],)).fetchall()
    
    session_list = []
    for idx, row in enumerate(sessions):
        entry = dict(row)
        entry['login_time_local'] = utc_to_nepal_time(entry.get('login_time'))
        # Mark the first (latest) one as "Current" because it matches the current session
        entry['is_current'] = (idx == 0)
        session_list.append(entry)

    # 3. Trusted devices (existing)
    devices = db.execute('''
        SELECT id, device_label, user_agent, ip_address, is_trusted, last_used, created_at
        FROM trusted_devices
        WHERE user_id = ?
        ORDER BY last_used DESC
    ''', (session['user_id'],)).fetchall()
    device_list = []
    for row in devices:
        entry = dict(row)
        entry['last_used_local'] = utc_to_nepal_time(entry.get('last_used'))
        device_list.append(entry)

    return render_template('security_center.html', 
                           login_history=login_history,
                           sessions=session_list,
                           devices=device_list)

@app.route('/security/logout_others', methods=['POST'])
@login_required
def logout_other_sessions():
    _validate_csrf()
    db = get_db()
    new_token = secrets.token_urlsafe(32)
    db.execute("UPDATE users SET session_token = ? WHERE id = ?", (new_token, session['user_id']))
    db.commit()
    session['session_token'] = new_token
    log_activity(session['user_id'], session['username'], 'logout_other_devices', 
                 f"Logged out all other devices from IP {request.remote_addr}")
    flash('✅ All other devices have been logged out successfully.', 'success')
    return redirect(url_for('security_center'))

@app.route('/security/trusted_devices/<int:device_id>/remove', methods=['POST'])
@login_required
def remove_trusted_device(device_id):
    _validate_csrf()
    db = get_db()
    device = db.execute("SELECT id FROM trusted_devices WHERE id = ? AND user_id = ?", (device_id, session['user_id'])).fetchone()
    if not device:
        flash('Device not found.', 'error')
        return redirect(url_for('security_center'))
    db.execute("DELETE FROM trusted_devices WHERE id = ?", (device_id,))
    db.commit()
    log_activity(session['user_id'], session['username'], 'device_removed', f"Removed trusted device {device_id}")
    flash('Trusted device removed.', 'success')
    return redirect(url_for('security_center'))

@app.route('/security/report_login/<int:log_id>', methods=['POST'])
@login_required
def report_suspicious_login(log_id):
    _validate_csrf()
    db = get_db()
    log_row = db.execute(
        "SELECT * FROM login_logs WHERE id = ? AND user_id = ?",
        (log_id, session['user_id']),
    ).fetchone()
    if not log_row:
        flash('Login record not found.', 'error')
        return redirect(url_for('security_center'))
    user = db.execute("SELECT email, username FROM users WHERE id = ?", (session['user_id'],)).fetchone()
    send_admin_alert(
        "User reported suspicious login",
        f"User: {user['username']}\nLog ID: {log_id}\nIP: {log_row['ip_address']}\n"
        f"Time: {log_row['login_time']}\nStatus: {log_row['status']}",
    )
    log_activity(
        session['user_id'], session['username'], 'report_suspicious_login',
        f"Reported login log {log_id} from IP {log_row['ip_address']}",
    )
    flash('Thank you. An administrator has been notified about this login.', 'success')
    return redirect(url_for('security_center'))

@app.route('/exams/<int:exam_id>/questions', methods=['GET', 'POST'])
@login_required
@behavior_check_required
def manage_exam_questions(exam_id):
    if session['role'] != 'teacher':
        abort(403)
    db = get_db()
    exam = db.execute('''
        SELECT e.*, c.course_name, c.teacher_id
        FROM exams e JOIN courses c ON e.course_id = c.id
        WHERE e.id = ?
    ''', (exam_id,)).fetchone()
    if not exam or exam['teacher_id'] != session['user_id']:
        abort(403)
    if not exam['is_online']:
        flash('Enable "Online timed exam" when scheduling to add MCQ questions.', 'warning')
    if request.method == 'POST':
        _validate_csrf()
        action = request.form.get('action', 'add')
        if action == 'delete':
            qid = request.form.get('question_id')
            if qid:
                db.execute(
                    "DELETE FROM exam_questions WHERE id = ? AND exam_id = ?",
                    (qid, exam_id),
                )
                db.commit()
                flash('Question removed.', 'success')
            return redirect(url_for('manage_exam_questions', exam_id=exam_id))
        question_text = request.form.get('question_text', '').strip()
        option_a = request.form.get('option_a', '').strip()
        option_b = request.form.get('option_b', '').strip()
        option_c = request.form.get('option_c', '').strip()
        option_d = request.form.get('option_d', '').strip()
        correct_option = request.form.get('correct_option', 'A').upper()
        marks = int(request.form.get('marks', 1) or 1)
        if not all([question_text, option_a, option_b, option_c, option_d]):
            flash('All question fields are required.', 'error')
        elif correct_option not in ('A', 'B', 'C', 'D'):
            flash('Correct option must be A, B, C, or D.', 'error')
        else:
            sort_order = db.execute(
                "SELECT COUNT(*) as c FROM exam_questions WHERE exam_id = ?",
                (exam_id,),
            ).fetchone()['c']
            db.execute('''
                INSERT INTO exam_questions
                (exam_id, question_text, option_a, option_b, option_c, option_d, correct_option, marks, sort_order)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (exam_id, question_text, option_a, option_b, option_c, option_d, correct_option, marks, sort_order))
            db.commit()
            flash('Question added.', 'success')
        return redirect(url_for('manage_exam_questions', exam_id=exam_id))
    questions = db.execute(
        "SELECT * FROM exam_questions WHERE exam_id = ? ORDER BY sort_order, id",
        (exam_id,),
    ).fetchall()
    total_q_marks = sum(q['marks'] for q in questions)
    return render_template('exam_questions.html', exam=exam, questions=questions, total_q_marks=total_q_marks)

def _grade_exam_attempt(attempt_id, exam_id):
    db = get_db()
    questions = db.execute(
        "SELECT id, correct_option, marks FROM exam_questions WHERE exam_id = ?",
        (exam_id,),
    ).fetchall()
    score = 0
    for q in questions:
        ans = db.execute(
            "SELECT selected_option FROM exam_answers WHERE attempt_id = ? AND question_id = ?",
            (attempt_id, q['id']),
        ).fetchone()
        selected = (ans['selected_option'] or '').upper() if ans else ''
        is_correct = 1 if selected == q['correct_option'] else 0
        if is_correct:
            score += q['marks']
        db.execute('''
            INSERT OR REPLACE INTO exam_answers (attempt_id, question_id, selected_option, is_correct)
            VALUES (?, ?, ?, ?)
        ''', (attempt_id, q['id'], selected or None, is_correct))
    attempt = db.execute("SELECT student_id FROM exam_attempts WHERE id = ?", (attempt_id,)).fetchone()
    db.execute(
        "UPDATE exam_attempts SET score = ?, submitted_at = CURRENT_TIMESTAMP WHERE id = ?",
        (score, attempt_id),
    )
    db.execute('''
        INSERT OR REPLACE INTO exam_grades (exam_id, student_id, marks_obtained)
        VALUES (?, ?, ?)
    ''', (exam_id, attempt['student_id'], score))
    db.commit()
    return score

@app.route('/exams/<int:exam_id>/take', methods=['GET', 'POST'])
@login_required
def take_online_exam(exam_id):
    if session['role'] != 'student':
        abort(403)
    db = get_db()
    exam = db.execute('''
        SELECT e.*, c.course_name
        FROM exams e JOIN courses c ON e.course_id = c.id
        WHERE e.id = ?
    ''', (exam_id,)).fetchone()
    if not exam or not exam['is_online']:
        flash('This exam is not available as an online test.', 'error')
        return redirect(url_for('list_exams'))
    enrolled = db.execute(
        "SELECT 1 FROM enrollments WHERE student_id = ? AND course_id = ?",
        (session['user_id'], exam['course_id']),
    ).fetchone()
    if not enrolled:
        flash('You are not enrolled in this course.', 'error')
        return redirect(url_for('list_exams'))
    _, _, window_open = get_exam_window(exam)
    if not window_open:
        flash('The exam window is not open right now. Check the scheduled date and time.', 'warning')
        return redirect(url_for('list_exams'))
    question_count = db.execute(
        "SELECT COUNT(*) as c FROM exam_questions WHERE exam_id = ?",
        (exam_id,),
    ).fetchone()['c']
    if question_count == 0:
        flash('No questions have been added for this exam yet.', 'warning')
        return redirect(url_for('list_exams'))
    attempt = db.execute(
        "SELECT * FROM exam_attempts WHERE exam_id = ? AND student_id = ?",
        (exam_id, session['user_id']),
    ).fetchone()
    if attempt and attempt['status'] in ('submitted', 'auto_submitted'):
        flash(f"Exam already submitted. Score: {attempt['score']}", 'info')
        return redirect(url_for('list_exams'))
    if request.method == 'POST':
        _validate_csrf()
        if not attempt:
            flash('Invalid exam session. Please start again.', 'error')
            return redirect(url_for('take_online_exam', exam_id=exam_id))
        auto_submit = request.form.get('auto_submit') == '1'
        status = 'auto_submitted' if auto_submit else 'submitted'
        for key, value in request.form.items():
            if key.startswith('q_'):
                qid = key[2:]
                db.execute('''
                    INSERT OR REPLACE INTO exam_answers (attempt_id, question_id, selected_option)
                    VALUES (?, ?, ?)
                ''', (attempt['id'], qid, value.upper()))
        db.execute("UPDATE exam_attempts SET status = ? WHERE id = ?", (status, attempt['id']))
        db.commit()
        score = _grade_exam_attempt(attempt['id'], exam_id)
        log_activity(session['user_id'], session['username'], 'exam_submitted',
                     f"Online exam {exam_id} score {score}")
        flash(f"Exam submitted successfully. Your score: {score}/{exam['total_marks']}", 'success')
        return redirect(url_for('list_exams'))
    if not attempt:
        cur = db.execute('''
            INSERT INTO exam_attempts (exam_id, student_id, status)
            VALUES (?, ?, 'in_progress')
        ''', (exam_id, session['user_id']))
        db.commit()
        attempt = db.execute(
            "SELECT * FROM exam_attempts WHERE id = ?",
            (cur.lastrowid,),
        ).fetchone()
    questions = db.execute(
        "SELECT * FROM exam_questions WHERE exam_id = ? ORDER BY sort_order, id",
        (exam_id,),
    ).fetchall()
    existing_answers = {}
    for row in db.execute(
        "SELECT question_id, selected_option FROM exam_answers WHERE attempt_id = ?",
        (attempt['id'],),
    ).fetchall():
        existing_answers[row['question_id']] = row['selected_option']
    start_dt, end_dt, _ = get_exam_window(exam)
    duration_seconds = int((end_dt - datetime.now()).total_seconds())
    duration_seconds = max(0, min(duration_seconds, (exam['duration_minutes'] or 60) * 60))
    return render_template(
        'take_exam.html',
        exam=exam,
        questions=questions,
        attempt=attempt,
        existing_answers=existing_answers,
        duration_seconds=duration_seconds,
    )
# ---------- Notification API Routes ----------
@app.route('/api/notifications/count')
@login_required
def notification_count():
    db = get_db()
    count = db.execute('''
        SELECT COUNT(*) as count FROM notifications
        WHERE user_id = ? AND is_read = 0
    ''', (session['user_id'],)).fetchone()['count']
    return jsonify({'count': count})

@app.route('/api/notifications/list')
@login_required
def notification_list():
    db = get_db()
    notifs = db.execute('''
        SELECT id, message, link, is_read, created_at
        FROM notifications
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 10
    ''', (session['user_id'],)).fetchall()
    data = []
    for n in notifs:
        data.append({
            'id': n['id'],
            'message': n['message'],
            'link': n['link'],
            'is_read': bool(n['is_read']),
            'time': n['created_at'].strftime('%Y-%m-%d %H:%M')
        })
    return jsonify(data)

@app.route('/api/notifications/read/<int:notification_id>', methods=['POST'])
@login_required
def mark_notification_read(notification_id):
    _validate_csrf()
    db = get_db()
    db.execute('''
        UPDATE notifications SET is_read = 1
        WHERE id = ? AND user_id = ?
    ''', (notification_id, session['user_id']))
    db.commit()
    return jsonify({'success': True})

@app.route('/api/notifications/read_all', methods=['POST'])
@login_required
def mark_all_read():
    _validate_csrf()
    db = get_db()
    db.execute('''
        UPDATE notifications SET is_read = 1
        WHERE user_id = ? AND is_read = 0
    ''', (session['user_id'],))
    db.commit()
    return jsonify({'success': True})


@app.route('/admin/ml_training', methods=['GET', 'POST'])
@login_required
@admin_required
def ml_training_dashboard():
    from train_model import run_training, load_training_metadata
    db = get_db()
    login_count = db.execute("SELECT COUNT(*) as c FROM login_logs").fetchone()['c']
    metadata = load_training_metadata()
    model_loaded = ensemble_risk is not None
    model_path = 'models/ensemble_risk.pkl'
    model_file_mtime = None
    if os.path.isfile(model_path):
        model_file_mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
    if request.method == 'POST':
        _validate_csrf() 
        outcome = run_training(DATABASE)
        if outcome['success']:
            reload_ensemble_model()
            flash(outcome['message'], 'success')
        else:
            flash(outcome['message'], 'warning')
        metadata = load_training_metadata()
        model_loaded = ensemble_risk is not None
        if os.path.isfile(model_path):
            model_file_mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
    feature_importance = []
    # ... rest unchanged
    return render_template('ml_training.html',
                           metadata=metadata,
                           login_count=login_count,
                           model_loaded=model_loaded,
                           model_file_mtime=model_file_mtime,
                           feature_importance=feature_importance)

# ---------- Admin Active Sessions ----------
@app.route('/admin/sessions', methods=['GET', 'POST'])
@login_required
@admin_required
def admin_sessions():
    db = get_db()
    
    # Search functionality
    search = request.args.get('search', '').strip()
    if search:
        # Search by username or email
        users = db.execute('''
            SELECT id, username, email, session_token, is_locked, is_banned
            FROM users
            WHERE username LIKE ? OR email LIKE ?
            ORDER BY username
        ''', (f'%{search}%', f'%{search}%')).fetchall()
    else:
        # Show all users with a non-null session_token (i.e., active sessions)
        users = db.execute('''
            SELECT id, username, email, session_token, is_locked, is_banned
            FROM users
            WHERE session_token IS NOT NULL AND session_token != ''
            ORDER BY username
        ''').fetchall()
    
    # Get the latest login info for each user
    session_data = []
    for user in users:
        # Get last successful login
        last_login = db.execute('''
            SELECT ip_address, device, login_time, geo_country
            FROM login_logs
            WHERE user_id = ? AND status = 'success'
            ORDER BY login_time DESC
            LIMIT 1
        ''', (user['id'],)).fetchone()
        
        if last_login:
            session_data.append({
                'user': user,
                'ip': last_login['ip_address'],
                'device': last_login['device'],
                'login_time': last_login['login_time'],
                'geo_country': last_login['geo_country'],
                'is_current': (user['id'] == session['user_id'])  # current admin's own session
            })
        else:
            # User has session_token but no login logs (edge case)
            session_data.append({
                'user': user,
                'ip': 'Unknown',
                'device': 'Unknown',
                'login_time': None,
                'geo_country': None,
                'is_current': (user['id'] == session['user_id'])
            })
    
    # Sort: current session first, then by login_time desc
    session_data.sort(key=lambda x: (not x['is_current'], x['login_time'] or datetime.min), reverse=True)
    
    return render_template('admin_sessions.html', sessions=session_data, search=search)

@app.route('/admin/sessions/logout/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def admin_logout_user(user_id):
    _validate_csrf()
    
    # Prevent admin from logging out themselves
    if user_id == session['user_id']:
        flash('You cannot log out your own session from here.', 'error')
        return redirect(url_for('admin_sessions'))
    
    db = get_db()
    user = db.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('admin_sessions'))
    
    # Generate a new session token to invalidate all sessions
    new_token = secrets.token_urlsafe(32)
    db.execute("UPDATE users SET session_token = ? WHERE id = ?", (new_token, user_id))
    db.commit()
    
    log_admin_action(session['user_id'], session['username'], 'force_logout', user_id, user['username'], 
                     f"Forced logout of user {user['username']} from all devices")
    log_activity(session['user_id'], session['username'], 'admin_force_logout', 
                 f"Admin forced logout of user {user['username']}")
    
    flash(f"✅ User '{user['username']}' has been logged out from all devices.", 'success')
    return redirect(url_for('admin_sessions'))

@app.route('/test-sendgrid')
def test_sendgrid():
    import os
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
    
    email = request.args.get('email', 'admin.academics@gmail.com')
    
    try:
        sg = SendGridAPIClient(api_key=os.getenv('SENDGRID_API_KEY'))
        message = Mail(
            from_email='admin.academics@gmail.com',
            to_emails=email,
            subject='Test Email from EduShield',
            html_content='<p>SendGrid is working!</p>'
        )
        response = sg.send(message)
        return f"""
        <h2>✅ Email Test Result</h2>
        <p><strong>Status Code:</strong> {response.status_code}</p>
        <p><strong>To:</strong> {email}</p>
        <p>Check your inbox/spam folder!</p>
        <a href='/'>Back to Home</a>
        """
    except Exception as e:
        return f"❌ Error: {e}"
    
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

with app.app_context():
    init_db()

if __name__ == '__main__':
    with app.app_context():
        init_db()
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)