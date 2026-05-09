CREATE TABLE IF NOT EXISTS admins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT NOT NULL,
    sku TEXT UNIQUE,
    category TEXT,
    price REAL NOT NULL,
    stock INTEGER DEFAULT 0,
    image_reference TEXT,
    model_class_name TEXT UNIQUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS product_images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER,
    file_path TEXT NOT NULL,
    FOREIGN KEY(product_id) REFERENCES products(product_id)
);

CREATE TABLE IF NOT EXISTS product_classes (
    class_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER,
    model_class_name TEXT,
    FOREIGN KEY(product_id) REFERENCES products(product_id)
);

CREATE TABLE IF NOT EXISTS training_jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT, -- pending, running, success, failed
    epochs INTEGER,
    product_id INTEGER,
    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_time DATETIME,
    FOREIGN KEY(product_id) REFERENCES products(product_id)
);

CREATE TABLE IF NOT EXISTS model_versions (
    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER,
    weights_path TEXT,
    is_active BOOLEAN DEFAULT 0,
    metrics_mAP REAL,
    FOREIGN KEY(job_id) REFERENCES training_jobs(job_id)
);

CREATE TABLE IF NOT EXISTS billing_sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_time DATETIME,
    total_amount REAL DEFAULT 0.0,
    status TEXT -- in_progress, completed, cancelled
);

CREATE TABLE IF NOT EXISTS billing_items (
    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    product_id INTEGER,
    quantity INTEGER DEFAULT 1,
    price_at_time REAL,
    FOREIGN KEY(session_id) REFERENCES billing_sessions(session_id),
    FOREIGN KEY(product_id) REFERENCES products(product_id)
);

CREATE TABLE IF NOT EXISTS detection_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    model_version_id INTEGER,
    detected_class TEXT,
    confidence REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES billing_sessions(session_id),
    FOREIGN KEY(model_version_id) REFERENCES model_versions(version_id)
);

CREATE TABLE IF NOT EXISTS failed_detections (
    fail_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    image_path TEXT,
    reason TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(session_id) REFERENCES billing_sessions(session_id)
);
