# Smart Dropbox

Smart Dropbox is an AI-powered product detection and checkout system built with Python, Flask, and YOLOv8. It supports both laptop webcam detection and mobile camera detection while storing product, billing, and training data in SQLite.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology](#technology)
- [Requirements](#requirements)
- [Setup](#setup)
- [Run the Project](#run-the-project)
- [Main Pages](#main-pages)
- [Admin Access](#admin-access)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

This project creates a smart retail kiosk style application that:

- Detects products using an AI model (YOLOv8)
- Maps detections to product records in a database
- Builds a cart and completes checkout
- Stores billing and detection history for analytics
- Offers a web-based admin panel to manage products and training jobs

The application supports two main detection modes:

- Laptop/desktop product detection
- Mobile camera detection using the phone browser

## Features

- Real-time product detection with YOLOv8
- Webcam and mobile-camera support
- Product catalog and stock management
- Checkout and billing history tracking
- Admin dashboard with product and billing views
- Dataset upload and custom training support
- SQLite database for easy local use

## Technology

- Python
- Flask web framework
- Ultralytics YOLOv8
- OpenCV for image processing
- SQLite for local database storage
- HTML, CSS, JavaScript front end

## Requirements

- Python 3.8+
- pip
- Local network access for mobile camera mode

## Setup

1. Open a terminal in the project folder:

```powershell
cd "c:\Users\harsh\Downloads\Smart Dropbox\Smart Dropbox"
```

2. Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Initialize the database:

```powershell
python init_db.py
```

This creates `database/store.sqlite3` and seeds a default admin account.

## Run the Project

Start the Flask application:

```powershell
python app.py
```

Then open your browser to:

- `http://127.0.0.1:5000`

## Main Pages

- `/` — Landing page
- `/dropbox` — Desktop product detection interface
- `/mobile/camera` — Mobile camera detection interface
- `/camera/test` — Camera access test page
- `/admin/login` — Admin login page
- `/admin/dashboard` — Admin dashboard

## Admin Access

The default admin credentials are:

- Username: `admin`
- Password: `admin`

> For production use, change the admin password and update security settings in `config.py`.

## How It Works

### Detection Flow

1. The browser sends an image to the Flask backend.
2. `core/cv_engine.py` runs YOLOv8 inference on the image.
3. The backend maps detected classes to products in SQLite.
4. Products are added to the cart and can be checked out.
5. Billing sessions and detection logs are saved in the database.

### Mobile Camera Flow

- The mobile page sends camera frames to `/api/mobile/detect`.
- The backend returns detection results and stream status.
- A live image preview is available at `/api/mobile/frame`.

### Checkout Flow

- Checkout requests are sent to `/api/dropbox/checkout`.
- The backend validates stock and reduces inventory.
- Billing records are saved in `billing_sessions` and `billing_items`.

## Project Structure

```
app.py                       # Main Flask application
config.py                    # Configuration settings
init_db.py                   # SQLite initialization and default admin
requirements.txt             # Python dependencies

core/                        # Core application logic
  cv_engine.py               # YOLO inference and detection helpers
  db.py                      # Database access helpers
  trainer.py                 # Dataset upload and model training logic
  utils.py                   # Utility functions

database/                    # Database schema and runtime storage
  schema.sql                 # SQLite schema definition
  store.sqlite3              # Runtime database file (created after setup)

datasets/                    # Training dataset folders
static/                      # Static assets and upload storage
templates/                   # HTML templates for pages
```

## Troubleshooting

- If app startup fails, confirm the virtual environment is active.
- If `yolov8n.pt` or `yolo26n.pt` are missing, add the model weights to the repository root.
- If the camera page cannot access hardware, check browser permissions.
- For mobile mode, ensure your phone is on the same local network as the PC.
- If admin login fails, re-run `python init_db.py` to recreate the default admin user.

## License

This repository includes a license file. Review `LICENSE` for terms.


<img width="1915" height="882" alt="image" src="https://github.com/user-attachments/assets/a6ad7855-56b6-43b0-97ab-16e3c7f87130" />
<img width="1917" height="888" alt="image" src="https://github.com/user-attachments/assets/d00e195e-e439-48bd-a62e-037b8df295d7" />
<img width="1919" height="893" alt="image" src="https://github.com/user-attachments/assets/6dd67fdd-b1fb-4d58-aa14-bd07897e29bd" />
<img width="1919" height="871" alt="image" src="https://github.com/user-attachments/assets/8aec5da4-497b-4a63-a605-9f14529d5785" />


