import sqlite3
from config import DATABASE

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_training_jobs_product_column():
    conn = get_db()
    try:
        table_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'training_jobs'"
        ).fetchone()
        if not table_exists:
            return

        columns = {
            row['name']
            for row in conn.execute("PRAGMA table_info(training_jobs)").fetchall()
        }
        if 'product_id' not in columns:
            conn.execute("ALTER TABLE training_jobs ADD COLUMN product_id INTEGER REFERENCES products(product_id)")
            conn.commit()
    finally:
        conn.close()

def query_db(query, args=(), one=False):
    conn = get_db()
    cur = conn.execute(query, args)
    rv = cur.fetchall()
    conn.commit()
    conn.close()
    return (rv[0] if rv else None) if one else rv

def insert_db(query, args=()):
    conn = get_db()
    cur = conn.execute(query, args)
    conn.commit()
    last_id = cur.lastrowid
    conn.close()
    return last_id
