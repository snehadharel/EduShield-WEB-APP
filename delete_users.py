import sqlite3

def delete_all_users_except_admin():
    conn = sqlite3.connect('academic_portal.db')
    cursor = conn.cursor()
    
    try:
        # Get admin ID first
        admin = cursor.execute("SELECT id FROM users WHERE username='admin'").fetchone()
        if not admin:
            print("❌ Admin user not found!")
            return
        
        admin_id = admin[0]
        print(f"✅ Found admin with ID: {admin_id}")
        
        # Delete from related tables first (foreign key constraints)
        cursor.execute("DELETE FROM login_logs WHERE user_id != ?", (admin_id,))
        cursor.execute("DELETE FROM failed_attempts WHERE user_id != ?", (admin_id,))
        cursor.execute("DELETE FROM user_activity_logs WHERE user_id != ?", (admin_id,))
        cursor.execute("DELETE FROM notifications WHERE user_id != ?", (admin_id,))
        cursor.execute("DELETE FROM profiles WHERE user_id != ?", (admin_id,))
        
        # Finally delete users
        cursor.execute("DELETE FROM users WHERE id != ?", (admin_id,))
        
        conn.commit()
        print(f"✅ All users except admin (ID: {admin_id}) deleted successfully!")
        print(f"   Rows affected: {cursor.rowcount}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    delete_all_users_except_admin()