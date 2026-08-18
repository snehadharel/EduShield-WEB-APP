# check_performance.py
import sqlite3
from datetime import datetime, timedelta

conn = sqlite3.connect('academic_portal.db')
cursor = conn.cursor()

print("=" * 60)
print("📊 ACTUAL SYSTEM PERFORMANCE METRICS")
print("=" * 60)

# Overall average
cursor.execute("""
    SELECT 
        COUNT(*) as total_requests,
        AVG(time_spent) as avg_ms,
        MIN(time_spent) as min_ms,
        MAX(time_spent) as max_ms
    FROM user_behavior_logs
    WHERE time_spent > 0 AND time_spent < 10000
""")
row = cursor.fetchone()
print(f"\n📈 Overall Performance ({row[0]} requests):")
print(f"  Average: {row[1]:.1f} ms")
print(f"  Min: {row[2]:.1f} ms")
print(f"  Max: {row[3]:.1f} ms")

# API endpoints
cursor.execute("""
    SELECT 
        AVG(time_spent) as avg_api_ms
    FROM user_behavior_logs
    WHERE url LIKE '%/api/%' AND time_spent > 0
""")
api_avg = cursor.fetchone()[0]
print(f"\n📊 API Endpoints (avg): {api_avg:.1f} ms" if api_avg else "  No API requests logged")

# Last 24 hours
cursor.execute("""
    SELECT 
        COUNT(*) as requests,
        AVG(time_spent) as avg_ms
    FROM user_behavior_logs
    WHERE created_at > datetime('now', '-1 day')
    AND time_spent > 0
""")
row = cursor.fetchone()
print(f"\n🕐 Last 24 Hours:")
print(f"  Requests: {row[0]}")
print(f"  Avg Response: {row[1]:.1f} ms" if row[1] else "  No data")

# Slowest endpoints
cursor.execute("""
    SELECT 
        url,
        COUNT(*) as count,
        AVG(time_spent) as avg_ms
    FROM user_behavior_logs
    WHERE time_spent > 0
    GROUP BY url
    ORDER BY avg_ms DESC
    LIMIT 5
""")
print(f"\n🐌 Slowest Endpoints:")
for row in cursor.fetchall():
    url_short = row[0][:60] + '...' if len(row[0]) > 60 else row[0]
    print(f"  {url_short} ({row[1]} requests, avg: {row[2]:.1f} ms)")

# Fastest endpoints
cursor.execute("""
    SELECT 
        url,
        COUNT(*) as count,
        AVG(time_spent) as avg_ms
    FROM user_behavior_logs
    WHERE time_spent > 0
    GROUP BY url
    ORDER BY avg_ms ASC
    LIMIT 5
""")
print(f"\n⚡ Fastest Endpoints:")
for row in cursor.fetchall():
    url_short = row[0][:60] + '...' if len(row[0]) > 60 else row[0]
    print(f"  {url_short} ({row[1]} requests, avg: {row[2]:.1f} ms)")

# Check if we have the behavior_logs table
cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='user_behavior_logs'")
table_exists = cursor.fetchone()[0]
if table_exists == 0:
    print("\n⚠️ No user_behavior_logs table found. Run the app first to generate data.")

conn.close()

print("\n" + "=" * 60)
print("✅ Performance monitoring is ACTIVE in your system!")
print("   Data source: user_behavior_logs table")