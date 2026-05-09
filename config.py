import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE = os.path.join(BASE_DIR, 'database', 'store.sqlite3')
SECRET_KEY = 'super_secret_prototype_key'
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
DATASET_FOLDER = os.path.join(BASE_DIR, 'datasets')
