from werkzeug.security import generate_password_hash, check_password_hash
from db_conn import get_db

# --- User operations ---

def create_user(username: str, password: str, role: str = 'user'):
    conn = get_db()
    c = conn.cursor()
    try:
        password_hash = generate_password_hash(password)
        c.execute('INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)', (username, password_hash, role))
        conn.commit()
        user_id = c.lastrowid
        return {'id': user_id, 'username': username, 'role': role}
    except Exception:
        return None
    finally:
        conn.close()


def verify_user(username: str, password: str):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT id, username, password_hash, role FROM users WHERE username = ?', (username,))
    user = c.fetchone()
    conn.close()
    if user and check_password_hash(user[2], password):
        return {'id': user[0], 'username': user[1], 'role': user[3]}
    return None


def list_users(limit: int = 500):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT id, username, role, created_at FROM users ORDER BY created_at DESC LIMIT ?', (limit,))
    users = [dict(row) for row in c.fetchall()]
    conn.close()
    return users


# --- Dataset operations ---

def create_dataset_record(name: str, storage_path: str, uploaded_by: int = None, session_id: str = None):
    conn = get_db()
    c = conn.cursor()
    c.execute('INSERT INTO datasets (name, storage_path, uploaded_by, session_id) VALUES (?, ?, ?, ?)',
              (name, storage_path, uploaded_by, session_id))
    conn.commit()
    ds_id = c.lastrowid
    conn.close()
    return ds_id


def get_dataset_by_session_id(session_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT id, name, storage_path, uploaded_by, session_id, uploaded_at FROM datasets WHERE session_id = ?', (session_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def list_datasets(limit: int = 500):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT id, name, storage_path, uploaded_by, session_id, uploaded_at FROM datasets ORDER BY uploaded_at DESC LIMIT ?', (limit,))
    datasets = [dict(row) for row in c.fetchall()]
    conn.close()
    return datasets


# --- Audit operations ---

def log_action(user_id: int, action: str, dataset_id: int = None, details: str = None):
    conn = get_db()
    c = conn.cursor()
    c.execute('INSERT INTO audit_logs (user_id, dataset_id, action, details) VALUES (?, ?, ?, ?)', (user_id, dataset_id, action, details))
    conn.commit()
    conn.close()


def list_audit_logs(limit: int = 500):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT id, user_id, dataset_id, action, details, timestamp FROM audit_logs ORDER BY timestamp DESC LIMIT ?', (limit,))
    logs = [dict(row) for row in c.fetchall()]
    conn.close()
    return logs
