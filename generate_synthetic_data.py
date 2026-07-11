# generate_synthetic_data.py
# Generates realistic synthetic login data for the Academic Portal IDS
# Includes Nepal-specific ISP simulation (WorldLink, Ncell, college Wi-Fi)
# matching the research hypothesis about ISP-based behavioral profiling

import sqlite3
import random
from datetime import datetime, timedelta

DB = "academic_portal.db"

# Nepal-specific ISP IP ranges (simulated)
ISP_PROFILES = {
    "WorldLink":    {"prefix": "202.70",  "weight": 0.35},
    "Ncell":        {"prefix": "103.1",   "weight": 0.25},
    "College_WiFi": {"prefix": "192.168", "weight": 0.30},
    "NTC":          {"prefix": "27.34",   "weight": 0.10},
}

def generate_nepal_ip(isp_name):
    """Generate a realistic IP address for a given Nepal ISP."""
    prefix = ISP_PROFILES[isp_name]["prefix"]
    return f"{prefix}.{random.randint(1,254)}.{random.randint(1,254)}"

def pick_isp():
    """Pick an ISP weighted by realistic usage distribution."""
    names = list(ISP_PROFILES.keys())
    weights = [ISP_PROFILES[n]["weight"] for n in names]
    return random.choices(names, weights=weights, k=1)[0]

conn = sqlite3.connect(DB)
cursor = conn.cursor()

# Get existing student user IDs
cursor.execute("SELECT id FROM users WHERE is_admin=0 AND role='student'")
user_ids = [row[0] for row in cursor.fetchall()]
if not user_ids:
    print("No student users found. Register a student first, then run this.")
    exit(1)

print(f"Generating synthetic login data for users: {user_ids}")
print("Simulating Nepal ISP profiles: WorldLink, Ncell, College Wi-Fi, NTC")

blocks_by_ip = {}
now = datetime.now()

# Build a consistent ISP profile per user (each student has a primary ISP)
user_primary_isp = {uid: pick_isp() for uid in user_ids}
user_primary_device = {uid: random.choice(["laptop", "mobile", "desktop"]) for uid in user_ids}

for _ in range(2200):
    user_id = random.choice(user_ids)
    primary_isp = user_primary_isp[user_id]
    primary_device = user_primary_device[user_id]

    # 80% of time use primary ISP, 20% roam (simulates moving between home/college/mobile)
    if random.random() < 0.80:
        isp = primary_isp
    else:
        isp = pick_isp()  # Different ISP = behavioural anomaly signal

    ip = generate_nepal_ip(isp)

    # 85% primary device, 15% different device
    if random.random() < 0.85:
        device = primary_device
    else:
        device = random.choice(["mobile", "laptop", "desktop", "tablet"])

    # Determine login outcome
    r = random.random()
    if r < 0.05:       # 5% blocked
        status = "blocked"
        action = "block"
        risk = random.randint(7, 10)
        blocks_by_ip[ip] = blocks_by_ip.get(ip, 0) + 1
    elif r < 0.15:     # 10% OTP required
        status = "otp_sent"
        action = "otp_required"
        risk = random.randint(3, 6)
    else:              # 85% normal
        status = "success"
        action = "allow"
        risk = random.randint(0, 2)

    # Normal hours: 8am–8pm; suspicious: late night / early morning
    hour = random.randint(8, 20)
    if status in ("blocked", "otp_sent") and random.random() < 0.3:
        hour = random.choice([1, 2, 3, 4, 23, 0])

    login_time = now - timedelta(
        days=random.randint(0, 30),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )

    alert_sent = 1 if risk >= 7 else 0

    cursor.execute('''
        INSERT INTO login_logs
            (user_id, ip_address, device, login_time, status, risk_score, action, alert_sent)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, ip, device, login_time, status, risk, action, alert_sent))

# Log failed IPs for brute-force table
for ip, count in blocks_by_ip.items():
    for _ in range(min(count, 5)):
        cursor.execute(
            "INSERT INTO ip_failed_attempts (ip_address, attempt_time) VALUES (?, ?)",
            (ip, now - timedelta(minutes=random.randint(1, 120)))
        )

conn.commit()
conn.close()
print(f"Done. 2200 synthetic login events inserted.")
print("ISP distribution reflects WorldLink, Ncell, College Wi-Fi, NTC patterns.")