import os
import re
import socket
import threading
import time
import uuid
from collections import Counter
from datetime import datetime

from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename

from config import SECRET_KEY, UPLOAD_FOLDER, DATASET_FOLDER
from core.cv_engine import cv_engine
from core.db import ensure_training_jobs_product_column, get_db, insert_db, query_db
from core.trainer import trainer
from core.utils import check_cooldown

app = Flask(__name__)
app.secret_key = SECRET_KEY

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(os.path.join('static', 'product_images'), exist_ok=True)
ensure_training_jobs_product_column()

HTTP_PORT = 5000
HTTPS_PORT = 5443
CERT_FILE = 'cert.pem'
KEY_FILE = 'key.pem'
MOBILE_FRAME_TIMEOUT_SECONDS = 4.0
PRODUCT_IMAGE_FOLDER = os.path.join('static', 'product_images')
PRODUCT_DETECTION_MIN_CONFIDENCE = 0.80
HAS_CUSTOM_CERT_FILES = os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE)
HTTPS_ENABLED = False
HTTPS_MODE = 'disabled'

mobile_stream_lock = threading.Lock()
mobile_stream_state = {
    'frame_bytes': None,
    'frame_id': 0,
    'items': [],
    'detections': [],
    'updated_at': 0.0,
}


def get_local_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(('8.8.8.8', 80))
        return sock.getsockname()[0]
    except OSError:
        return '127.0.0.1'
    finally:
        sock.close()


def is_loopback_host(hostname):
    return hostname in {'127.0.0.1', 'localhost', '::1'}


def is_secure_request():
    forwarded_proto = request.headers.get('X-Forwarded-Proto', '')
    return request.is_secure or forwarded_proto == 'https'


def build_https_url():
    host = request.host.split(':', 1)[0]
    path = request.full_path if request.query_string else request.path
    return f'https://{host}:{HTTPS_PORT}{path.rstrip("?")}'


def save_product_image(file_storage):
    if not file_storage or not file_storage.filename:
        return None

    filename = secure_filename(file_storage.filename)
    if not filename:
        return None

    ext = os.path.splitext(filename)[1].lower()
    stored_name = f"{uuid.uuid4().hex}{ext}"
    target_path = os.path.join(PRODUCT_IMAGE_FOLDER, stored_name)
    file_storage.save(target_path)
    return f"/static/product_images/{stored_name}"


def delete_product_image(image_reference):
    if not image_reference:
        return

    prefix = '/static/product_images/'
    if not image_reference.startswith(prefix):
        return

    filename = os.path.basename(image_reference)
    file_path = os.path.abspath(os.path.join(PRODUCT_IMAGE_FOLDER, filename))
    folder_path = os.path.abspath(PRODUCT_IMAGE_FOLDER)

    if not file_path.startswith(folder_path):
        return

    if os.path.exists(file_path):
        os.remove(file_path)


def normalize_class_name(value):
    return re.sub(r'[^a-z0-9]+', '', (value or '').strip().lower())


def format_product_name_from_class(class_name):
    cleaned = re.sub(r'[_\-]+', ' ', (class_name or '').strip())
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned.title() or 'Imported Product'


def generate_product_sku(class_name):
    base = re.sub(r'[^A-Z0-9]+', '_', (class_name or '').upper()).strip('_') or 'PRODUCT'
    candidate = base
    suffix = 1
    while query_db("SELECT product_id FROM products WHERE sku = ?", [candidate], one=True):
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


def get_product_class_names(product_id):
    aliases = query_db(
        """
        SELECT model_class_name
        FROM product_classes
        WHERE product_id = ?
        """,
        [product_id]
    )

    product = query_db(
        """
        SELECT model_class_name
        FROM products
        WHERE product_id = ?
        """,
        [product_id],
        one=True
    )

    class_names = []
    seen = set()

    for value in [product['model_class_name']] if product else []:
        normalized = normalize_class_name(value)
        if normalized and normalized not in seen:
            class_names.append(value)
            seen.add(normalized)

    for alias in aliases:
        value = alias['model_class_name']
        normalized = normalize_class_name(value)
        if normalized and normalized not in seen:
            class_names.append(value)
            seen.add(normalized)

    return class_names


def find_product_by_class_name(class_name):
    product = query_db(
        "SELECT * FROM products WHERE lower(model_class_name) = lower(?)",
        [class_name],
        one=True
    )
    if product:
        return product

    normalized_class = normalize_class_name(class_name)
    if not normalized_class:
        return None

    alias_match = query_db(
        """
        SELECT p.*
        FROM product_classes pc
        JOIN products p ON p.product_id = pc.product_id
        WHERE lower(pc.model_class_name) = lower(?)
        LIMIT 1
        """,
        [class_name],
        one=True
    )
    if alias_match:
        return alias_match

    products = query_db("SELECT * FROM products")
    alias_rows = query_db(
        """
        SELECT p.*, pc.model_class_name AS alias_class_name
        FROM product_classes pc
        JOIN products p ON p.product_id = pc.product_id
        """
    )

    for candidate in products:
        if normalize_class_name(candidate['model_class_name']) == normalized_class:
            return candidate

    for candidate in alias_rows:
        if normalize_class_name(candidate['alias_class_name']) == normalized_class:
            return candidate

    return None


def find_product_for_detection(item):
    product_id = item.get('product_id')
    if product_id:
        product = query_db(
            "SELECT * FROM products WHERE product_id = ?",
            [int(product_id)],
            one=True
        )
        if product:
            return product

    return find_product_by_class_name(item.get('class_name'))


def map_detected_items(detected, min_confidence, include_metadata=False):
    valid_items = []
    out_of_stock_items = []
    seen_out_of_stock = set()

    for item in detected:
        if item['confidence'] < min_confidence:
            continue

        product = find_product_for_detection(item)
        if not product:
            continue

        current_stock = max(int(product['stock'] or 0), 0)
        if current_stock <= 0:
            if product['product_id'] not in seen_out_of_stock:
                out_of_stock_items.append({
                    'product_id': product['product_id'],
                    'product_name': product['product_name'],
                    'class_name': item['class_name'],
                    'stock_available': current_stock
                })
                seen_out_of_stock.add(product['product_id'])
            continue

        if not check_cooldown(product['product_id']):
            continue

        mapped_item = {
            'product_id': product['product_id'],
            'product_name': product['product_name'],
            'price': float(product['price']),
            'stock_available': current_stock,
            'image_reference': product['image_reference']
        }

        if include_metadata:
            mapped_item['confidence'] = float(item['confidence'])
            mapped_item['class_name'] = item['class_name']

        valid_items.append(mapped_item)

    return {
        'items': valid_items,
        'out_of_stock': out_of_stock_items
    }


def update_mobile_stream(frame_bytes, detected_items, detections=None):
    with mobile_stream_lock:
        mobile_stream_state['frame_bytes'] = frame_bytes
        mobile_stream_state['frame_id'] += 1
        mobile_stream_state['items'] = detected_items
        mobile_stream_state['detections'] = list(detections or [])
        mobile_stream_state['updated_at'] = time.time()


def get_mobile_stream_snapshot():
    with mobile_stream_lock:
        return {
            'frame_bytes': mobile_stream_state['frame_bytes'],
            'frame_id': mobile_stream_state['frame_id'],
            'items': list(mobile_stream_state['items']),
            'detections': list(mobile_stream_state['detections']),
            'updated_at': mobile_stream_state['updated_at'],
        }


def get_mobile_connection_payload():
    snapshot = get_mobile_stream_snapshot()
    connected = (time.time() - snapshot['updated_at']) <= MOBILE_FRAME_TIMEOUT_SECONDS
    snapshot['connected'] = connected
    return snapshot


@app.before_request
def redirect_secure_mobile_routes():
    if not HTTPS_ENABLED or is_secure_request():
        return None

    host = request.host.split(':', 1)[0]
    if is_loopback_host(host):
        return None

    secure_prefixes = ('/mobile/camera', '/camera/test', '/api/mobile/')
    if request.path.startswith(secure_prefixes):
        return redirect(build_https_url(), code=302)

    return None


# ===== Admin Routes =====
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        admin = query_db("SELECT * FROM admins WHERE username = ?", [username], one=True)
        if admin and check_password_hash(admin['password_hash'], password):
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        return render_template('admin/login.html', error="Invalid Credentials")
    return render_template('admin/login.html')


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))


@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    stats = {
        'total_sales': query_db(
            "SELECT SUM(total_amount) as total FROM billing_sessions WHERE status='completed'",
            one=True
        )['total'] or 0,
        'products_count': query_db("SELECT COUNT(*) as count FROM products", one=True)['count'] or 0
    }
    recent_sessions = query_db("SELECT * FROM billing_sessions ORDER BY start_time DESC LIMIT 5")
    return render_template('admin/dashboard.html', stats=stats, recent_sessions=recent_sessions)


@app.route('/admin/products', methods=['GET'])
def admin_products():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    products = query_db("SELECT * FROM products")
    return render_template('admin/products.html', products=products)


@app.route('/admin/billing')
def admin_billing():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    sessions = query_db(
        """
        SELECT
            bs.session_id,
            bs.start_time,
            bs.end_time,
            bs.total_amount,
            bs.status,
            COUNT(bi.item_id) AS item_count
        FROM billing_sessions bs
        LEFT JOIN billing_items bi ON bi.session_id = bs.session_id
        GROUP BY bs.session_id, bs.start_time, bs.end_time, bs.total_amount, bs.status
        ORDER BY bs.start_time DESC
        """
    )
    return render_template('admin/billing.html', sessions=sessions)


@app.route('/admin/billing/<int:session_id>/invoice')
def admin_billing_invoice(session_id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    session_row = query_db(
        """
        SELECT session_id, start_time, end_time, total_amount, status
        FROM billing_sessions
        WHERE session_id = ?
        """,
        [session_id],
        one=True
    )

    if not session_row:
        return redirect(url_for('admin_billing'))

    items = query_db(
        """
        SELECT
            p.product_name,
            p.sku,
            SUM(bi.quantity) AS quantity,
            bi.price_at_time,
            SUM(bi.quantity * bi.price_at_time) AS line_total
        FROM billing_items bi
        LEFT JOIN products p ON p.product_id = bi.product_id
        WHERE bi.session_id = ?
        GROUP BY p.product_name, p.sku, bi.price_at_time
        ORDER BY p.product_name
        """,
        [session_id]
    )

    total_items = sum(int(item['quantity'] or 0) for item in items)
    return render_template(
        'admin/invoice.html',
        bill_session=session_row,
        items=items,
        total_items=total_items
    )


@app.route('/api/products', methods=['POST'])
def add_product():
    if not session.get('admin_logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.form
    image_reference = None
    try:
        image_reference = save_product_image(request.files.get('image'))
        product_id = insert_db(
            """
            INSERT INTO products (
                product_name, sku, category, price, stock, image_reference, model_class_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data['name'],
                data['sku'],
                data['category'],
                float(data['price']),
                int(data['stock']),
                image_reference,
                data['model_class_name']
            )
        )
        return jsonify({'success': True, 'product_id': product_id})
    except Exception as exc:
        delete_product_image(image_reference)
        return jsonify({'error': str(exc)}), 400


@app.route('/api/products/<int:product_id>', methods=['POST'])
def update_product(product_id):
    if not session.get('admin_logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.form
    new_image_reference = None
    try:
        existing_product = query_db(
            """
            SELECT product_id, image_reference
            FROM products
            WHERE product_id = ?
            """,
            [product_id],
            one=True
        )
        if not existing_product:
            return jsonify({'error': 'Product not found.'}), 404

        old_image_reference = existing_product['image_reference']
        image_reference = old_image_reference
        new_image_reference = save_product_image(request.files.get('image'))
        if new_image_reference:
            image_reference = new_image_reference

        query_db(
            """
            UPDATE products
            SET product_name = ?, sku = ?, category = ?, price = ?, stock = ?, image_reference = ?, model_class_name = ?
            WHERE product_id = ?
            """,
            (
                data['name'],
                data['sku'],
                data.get('category', ''),
                float(data['price']),
                int(data['stock']),
                image_reference,
                data['model_class_name'],
                product_id
            )
        )
        if new_image_reference:
            delete_product_image(old_image_reference)
        return jsonify({'success': True, 'product_id': product_id})
    except Exception as exc:
        if new_image_reference:
            delete_product_image(new_image_reference)
        return jsonify({'error': str(exc)}), 400


@app.route('/api/products/<int:product_id>/delete', methods=['POST'])
def delete_product(product_id):
    if not session.get('admin_logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401

    product = query_db(
        """
        SELECT product_id, product_name, image_reference
        FROM products
        WHERE product_id = ?
        """,
        [product_id],
        one=True
    )
    if not product:
        return jsonify({'error': 'Product not found.'}), 404

    usage = query_db(
        """
        SELECT COUNT(*) AS count
        FROM billing_items
        WHERE product_id = ?
        """,
        [product_id],
        one=True
    )
    if usage and int(usage['count'] or 0) > 0:
        return jsonify({
            'error': 'This product already exists in billing history and cannot be deleted.'
        }), 400

    try:
        delete_product_image(product['image_reference'])
        query_db("DELETE FROM product_classes WHERE product_id = ?", [product_id])
        query_db("DELETE FROM product_images WHERE product_id = ?", [product_id])
        query_db("DELETE FROM products WHERE product_id = ?", [product_id])
        return jsonify({'success': True, 'product_id': product_id})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/admin/training')
def admin_training():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    products = query_db(
        "SELECT product_id, product_name, sku, category, model_class_name FROM products ORDER BY product_name"
    )
    selected_product_id = request.args.get('product_id', type=int)
    selected_product = None
    if selected_product_id:
        selected_product = query_db(
            """
            SELECT product_id, product_name, sku, category, model_class_name
            FROM products
            WHERE product_id = ?
            """,
            [selected_product_id],
            one=True
        )

    jobs = query_db(
        """
        SELECT
            tj.*,
            p.product_name
        FROM training_jobs tj
        LEFT JOIN products p ON p.product_id = tj.product_id
        ORDER BY tj.start_time DESC
        LIMIT 10
        """
    )
    classes = query_db("SELECT model_class_name FROM products")
    models = query_db("SELECT * FROM model_versions")
    return render_template(
        'admin/training.html',
        jobs=jobs,
        classes=classes,
        models=models,
        products=products,
        selected_product=selected_product
    )


@app.route('/api/training/upload', methods=['POST'])
def upload_dataset():
    if not session.get('admin_logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    product_id = request.form.get('product_id', type=int)
    if not product_id:
        return jsonify({'error': 'Select a product before uploading a dataset.'}), 400
    product = query_db(
        """
        SELECT product_id, product_name, sku, model_class_name
        FROM products
        WHERE product_id = ?
        """,
        [product_id],
        one=True
    )
    if not product:
        return jsonify({'error': 'Selected product was not found.'}), 404
    if 'dataset' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['dataset']
    if not file.filename.endswith('.zip'):
        return jsonify({'error': 'Please upload a .zip file (Roboflow format)'}), 400

    filename = f"dataset_{uuid.uuid4().hex}.zip"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    success, msg = trainer.extract_dataset(filepath)
    if success:
        classes = trainer.parse_classes()
        expected_class = product['model_class_name']
        expected_classes = get_product_class_names(product['product_id'])
        expected_class_keys = {normalize_class_name(class_name) for class_name in expected_classes}
        matched_classes = [
            class_name for class_name in classes
            if normalize_class_name(class_name) in expected_class_keys
        ]
        return jsonify({
            'success': True,
            'msg': msg,
            'classes': classes,
            'expected_class': expected_class,
            'expected_classes': expected_classes,
            'selected_product': {
                'product_id': product['product_id'],
                'product_name': product['product_name'],
                'sku': product['sku']
            },
            'class_match': bool(matched_classes),
            'matched_classes': matched_classes
        })
    return jsonify({'error': msg}), 500


@app.route('/api/training/start', methods=['POST'])
def start_training():
    if not session.get('admin_logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.json or {}
    product_id = data.get('product_id')
    if not product_id:
        return jsonify({'error': 'Select a product before starting training.'}), 400
    product = query_db(
        """
        SELECT product_id, product_name, sku, model_class_name
        FROM products
        WHERE product_id = ?
        """,
        [int(product_id)],
        one=True
    )
    if not product:
        return jsonify({'error': 'Selected product was not found.'}), 404
    epochs = int(data.get('epochs', 20))
    try:
        job_id = insert_db(
            "INSERT INTO training_jobs (status, epochs, product_id) VALUES (?, ?, ?)",
            ('pending', epochs, int(product_id))
        )
        trainer.start_training(job_id=job_id, epochs=epochs)
        return jsonify({
            'success': True,
            'job_id': job_id,
            'product': {
                'product_id': product['product_id'],
                'product_name': product['product_name'],
                'sku': product['sku']
            }
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/training/bulk-local', methods=['POST'])
def bulk_local_training():
    if not session.get('admin_logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.json or {}
    epochs = int(data.get('epochs', 20))
    force_retrain = bool(data.get('force_retrain', False))

    zip_candidates = {}
    search_roots = [DATASET_FOLDER, os.path.dirname(DATASET_FOLDER)]
    for root in search_roots:
        for name in os.listdir(root):
            if not name.lower().endswith('.zip'):
                continue
            zip_candidates[os.path.abspath(os.path.join(root, name))] = True
    dataset_zips = sorted(zip_candidates.keys())

    if not dataset_zips:
        return jsonify({'error': 'No dataset zip files were found in the datasets folder.'}), 404

    summary = {
        'queued': [],
        'created_products': [],
        'skipped_existing_models': [],
        'skipped_invalid': [],
    }

    for zip_path in dataset_zips:
        zip_name = os.path.basename(zip_path)
        try:
            class_names = trainer.inspect_dataset_zip(zip_path)
        except Exception as exc:
            summary['skipped_invalid'].append({
                'zip': zip_name,
                'reason': str(exc)
            })
            continue

        if not class_names:
            summary['skipped_invalid'].append({
                'zip': zip_name,
                'reason': 'No classes found in data.yaml'
            })
            continue

        grouped_classes = {}
        for class_name in class_names:
            product = find_product_by_class_name(class_name)
            if not product:
                product_id = insert_db(
                    """
                    INSERT INTO products (
                        product_name, sku, category, price, stock, image_reference, model_class_name
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        format_product_name_from_class(class_name),
                        generate_product_sku(class_name),
                        'Auto Imported',
                        0,
                        0,
                        None,
                        class_name
                    )
                )
                product = query_db(
                    "SELECT * FROM products WHERE product_id = ?",
                    [product_id],
                    one=True
                )
                summary['created_products'].append({
                    'product_id': product_id,
                    'product_name': product['product_name'],
                    'class_name': class_name
                })

            grouped_classes.setdefault(product['product_id'], {
                'product': product,
                'class_names': []
            })
            grouped_classes[product['product_id']]['class_names'].append(class_name)

        for item in grouped_classes.values():
            product = item['product']
            matched_classes = item['class_names']
            existing_model = query_db(
                """
                SELECT mv.version_id
                FROM model_versions mv
                JOIN training_jobs tj ON tj.job_id = mv.job_id
                WHERE tj.product_id = ?
                  AND tj.status = 'success'
                ORDER BY mv.version_id DESC
                LIMIT 1
                """,
                [product['product_id']],
                one=True
            )
            if existing_model and not force_retrain:
                summary['skipped_existing_models'].append({
                    'zip': zip_name,
                    'product_name': product['product_name'],
                    'class_name': ', '.join(matched_classes)
                })
                continue

            job_id = insert_db(
                "INSERT INTO training_jobs (status, epochs, product_id) VALUES (?, ?, ?)",
                ('pending', epochs, int(product['product_id']))
            )
            trainer.start_training(
                job_id=job_id,
                epochs=epochs,
                dataset_zip_path=zip_path,
                target_class_name=matched_classes
            )
            summary['queued'].append({
                'job_id': job_id,
                'zip': zip_name,
                'product_id': product['product_id'],
                'product_name': product['product_name'],
                'class_name': ', '.join(matched_classes)
            })

    return jsonify({
        'success': True,
        'summary': summary
    })


# ===== Dropbox Routes =====
@app.route('/')
def landing_page():
    return render_template('landing.html')


@app.route('/dropbox')
def dropbox_index():
    return render_template(
        'dropbox/index.html',
        network_host=get_local_ip(),
        http_port=HTTP_PORT,
        https_port=HTTPS_PORT
    )


@app.route('/api/dropbox/infer', methods=['POST'])
def dropbox_infer():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    img_bytes = request.files['image'].read()
    detected = cv_engine.infer(img_bytes)
    if not detected:
        return jsonify({'success': True, 'items': [], 'detections': [], 'out_of_stock': []})

    mapped = map_detected_items(detected, min_confidence=PRODUCT_DETECTION_MIN_CONFIDENCE)

    return jsonify({
        'success': True,
        'items': mapped['items'],
        'detections': detected,
        'out_of_stock': mapped['out_of_stock']
    })


@app.route('/api/dropbox/cart/clear', methods=['POST'])
def clear_cart():
    return jsonify({'success': True})


@app.route('/api/dropbox/checkout', methods=['POST'])
def process_checkout():
    data = request.json or {}
    cart = data.get('cart', [])
    if not cart:
        return jsonify({'error': 'Cart is empty'}), 400

    conn = None
    try:
        conn = get_db()
        requested_counts = Counter(int(item['id']) for item in cart)
        product_rows = {}

        for product_id, requested_qty in requested_counts.items():
            product = conn.execute(
                "SELECT product_id, product_name, price, stock FROM products WHERE product_id = ?",
                (product_id,)
            ).fetchone()

            if not product:
                return jsonify({'error': f'Product {product_id} was not found.'}), 400

            available_stock = max(int(product['stock'] or 0), 0)
            product_rows[product_id] = product

            if available_stock < requested_qty:
                return jsonify({
                    'error': f"{product['product_name']} is out of stock.",
                    'product_id': product_id,
                    'product_name': product['product_name'],
                    'available_stock': available_stock,
                    'requested_quantity': requested_qty
                }), 409

        cursor = conn.execute("INSERT INTO billing_sessions (status) VALUES ('completed')")
        session_id = cursor.lastrowid
        total = 0.0

        for item in cart:
            product_id = int(item['id'])
            product = product_rows[product_id]
            item_price = float(product['price'])
            total += item_price

            conn.execute(
                "INSERT INTO billing_items (session_id, product_id, price_at_time) VALUES (?, ?, ?)",
                (session_id, product_id, item_price)
            )
        
        for product_id, requested_qty in requested_counts.items():
            conn.execute(
                "UPDATE products SET stock = stock - ? WHERE product_id = ?",
                (requested_qty, product_id)
            )

        conn.execute(
            "UPDATE billing_sessions SET total_amount = ? WHERE session_id = ?",
            (total, session_id)
        )
        conn.commit()
        return jsonify({'success': True, 'total': total})
    except Exception as exc:
        if conn:
            conn.rollback()
        return jsonify({'error': str(exc)}), 500
    finally:
        if conn:
            conn.close()


# ===== Mobile Camera Routes =====
@app.route('/api/mobile/detect', methods=['POST'])
def mobile_detect():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    try:
        img_bytes = request.files['image'].read()
        detected = cv_engine.infer(img_bytes)
        mapped = map_detected_items(detected, min_confidence=PRODUCT_DETECTION_MIN_CONFIDENCE, include_metadata=True) if detected else {
            'items': [],
            'out_of_stock': []
        }
        valid_items = mapped['items']

        update_mobile_stream(img_bytes, valid_items, detected)

        for item in valid_items:
            insert_db(
                "INSERT INTO detection_logs (detected_class, confidence, timestamp) VALUES (?, ?, ?)",
                (item['class_name'], item['confidence'], datetime.now())
            )

        return jsonify({
            'success': True,
            'items': valid_items,
            'detections': detected,
            'out_of_stock': mapped['out_of_stock']
        })
    except Exception as exc:
        print(f"Error in mobile_detect: {exc}")
        return jsonify({'success': False, 'error': str(exc)}), 500


@app.route('/api/mobile/status')
def mobile_status():
    snapshot = get_mobile_connection_payload()
    return jsonify({
        'success': True,
        'connected': snapshot['connected'],
        'frame_id': snapshot['frame_id'],
        'updated_at': snapshot['updated_at'],
        'items': snapshot['items'],
        'detections': snapshot['detections']
    })


@app.route('/api/mobile/frame')
def mobile_frame():
    snapshot = get_mobile_stream_snapshot()
    if not snapshot['frame_bytes']:
        return '', 404

    response = Response(snapshot['frame_bytes'], mimetype='image/jpeg')
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['X-Frame-Id'] = str(snapshot['frame_id'])
    return response


@app.route('/camera/test')
def camera_test():
    return render_template('camera_test.html')


@app.route('/mobile/camera')
def mobile_camera():
    return render_template(
        'dropbox/mobile_camera_v2.html',
        network_host=get_local_ip(),
        http_port=HTTP_PORT,
        https_port=HTTPS_PORT
    )


def run_server(server, label):
    print(f"{label} server listening on {server.server_address[0]}:{server.server_address[1]}")
    server.serve_forever()


def create_https_server(make_server_func):
    global HTTPS_ENABLED, HTTPS_MODE

    if HAS_CUSTOM_CERT_FILES:
        try:
            server = make_server_func(
                '0.0.0.0',
                HTTPS_PORT,
                app,
                threaded=True,
                ssl_context=(CERT_FILE, KEY_FILE)
            )
            HTTPS_ENABLED = True
            HTTPS_MODE = 'custom'
            return server
        except Exception as exc:
            print(f"Custom HTTPS certificate load failed: {exc}")
            print("Falling back to an ad-hoc development certificate.")

    try:
        server = make_server_func(
            '0.0.0.0',
            HTTPS_PORT,
            app,
            threaded=True,
            ssl_context='adhoc'
        )
        HTTPS_ENABLED = True
        HTTPS_MODE = 'adhoc'
        return server
    except Exception as exc:
        print(f"Unable to start HTTPS server: {exc}")
        HTTPS_ENABLED = False
        HTTPS_MODE = 'disabled'
        return None


if __name__ == '__main__':
    from werkzeug.serving import make_server

    local_ip = get_local_ip()
    servers = []

    http_server = make_server('0.0.0.0', HTTP_PORT, app, threaded=True)
    servers.append(('HTTP', http_server))

    https_server = create_https_server(make_server)
    if https_server:
        servers.append(('HTTPS', https_server))

    print("Starting Smart Dropbox Server...")
    print(f"Local landing page: http://127.0.0.1:{HTTP_PORT}")
    print(f"Laptop monitor: http://127.0.0.1:{HTTP_PORT}/dropbox")
    print(f"Network monitor: http://{local_ip}:{HTTP_PORT}/dropbox")
    if HTTPS_ENABLED:
        print(f"Mobile camera URL: https://{local_ip}:{HTTPS_PORT}/mobile/camera")
        print(f"Mobile test URL: https://{local_ip}:{HTTPS_PORT}/camera/test")
        if HTTPS_MODE == 'adhoc':
            print("HTTPS is using an ad-hoc development certificate.")
            print("Your phone may show a certificate warning. Continue past it or trust the certificate on the device.")
    else:
        print("HTTPS could not be started.")
        print("Mobile camera access will fail on most phones until HTTPS is available.")

    threads = []
    for label, server in servers:
        thread = threading.Thread(target=run_server, args=(server, label), daemon=True)
        thread.start()
        threads.append(thread)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        for _, server in servers:
            server.shutdown()
