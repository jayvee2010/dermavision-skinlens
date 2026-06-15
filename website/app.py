"""
Skin Lens — Flask Backend
=========================
Serves the frontend and exposes two API endpoints:
  POST /api/scan/quick    — instant CNN inference
  POST /api/scan/detailed — queues a job (returns job_id, result polled later)
  GET  /api/scan/status/<job_id> — check detailed scan status

Run:
    python app.py
    or for production:
    gunicorn -w 2 -b 0.0.0.0:5000 app:app
"""

import os
import uuid
import time
import threading
import logging
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from predictor import SkinLensPredictor
from image_quality import check_image_quality

# ── Setup ──────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Ensure this is __name__ (double underscore), not _name_
app = Flask(__name__, static_folder='static')
CORS(app)  # Allow requests from the frontend

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
MAX_FILE_SIZE_MB = 10

# In-memory store for detailed scan jobs (use Redis/DB in production)
detailed_jobs = {}

# Load model once at startup
# 

# ── Helpers ────────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload(file):
    """Save uploaded file with a unique name. Returns file path."""
    ext = file.filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(path)
    return path


def cleanup_file(path, delay=60):
    """Delete uploaded file after delay seconds (privacy by design)."""
    def _delete():
        time.sleep(delay)
        try:
            os.remove(path)
            logger.info(f"Deleted temp file: {path}")
        except FileNotFoundError:
            pass
    threading.Thread(target=_delete, daemon=True).start()


def adjust_risk_with_metadata(result, metadata):
    """
    Adjusts the risk level based on patient metadata.
    Heuristic: High sun exposure or advanced age increases risk profile.
    """
    if not metadata:
        return result

    age = metadata.get('age')
    sun = metadata.get('sun_exposure')
    history = metadata.get('family_history')
    change = metadata.get('has_changed')
    symptoms = metadata.get('symptoms')
    original_risk = result.get('risk_level')

    # Heuristic: If 65+ or High Sun Exposure, bump risk level one step
    risk_factors = []
    if age == '65+': risk_factors.append('Age 65+')
    if sun == 'high': risk_factors.append('High Sun Exposure')
    if history == 'yes': risk_factors.append('Family History')
    if change == 'yes': risk_factors.append('Lesion Changing')
    if symptoms == 'yes': risk_factors.append('Symptoms (Itch/Bleed)')

    if risk_factors and original_risk in ['low', 'moderate']:
        new_risk = 'moderate' if original_risk == 'low' else 'high'
        
        result['risk_level'] = new_risk
        result['risk_label'] = f"{new_risk.title()} Risk (Adjusted)"
        
        # Append note to guidance
        note = f" (Risk elevated due to: {', '.join(risk_factors)})"
        if isinstance(result.get('guidance'), dict):
            for lang in result['guidance']:
                result['guidance'][lang] += note
    
    return result


def run_detailed_analysis(job_id, image_path):
    """
    Simulates a multi-pass detailed scan running in a background thread.
    In production, replace with a proper task queue (Celery + Redis).
    """
    detailed_jobs[job_id]['status'] = 'processing'

    # Simulate a deeper analysis (multiple passes, larger input resolution, etc.)
    # In reality: run model with TTA (test-time augmentation), ensemble, etc.
    time.sleep(10)  # Simulate heavier computation

    try:
        result = predictor.predict(image_path, detailed=True)
        
        # Adjust based on metadata stored in the job
        metadata = detailed_jobs[job_id].get('metadata', {})
        result = adjust_risk_with_metadata(result, metadata)

        if 'error' in result:
            detailed_jobs[job_id].update({'status': 'error', 'error': result['error']})
        else:
            detailed_jobs[job_id].update({
                'status': 'complete',
                'result': result,
                'completed_at': datetime.utcnow().isoformat()
            })
    except Exception as e:
        detailed_jobs[job_id].update({'status': 'error', 'error': str(e)})
    finally:
        cleanup_file(image_path, delay=0)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': predictor.is_loaded()})


@app.route('/api/scan/quick', methods=['POST'])
def quick_scan():
    """
    Quick scan endpoint.
    Accepts: multipart/form-data with 'image' field.
    Returns: JSON with risk level, confidence, class probabilities.
    """
    # ── Validate request ──
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG or PNG.'}), 400

    # Check file size
    file.seek(0, 2)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    if size_mb > MAX_FILE_SIZE_MB:
        return jsonify({'error': f'File too large. Maximum size is {MAX_FILE_SIZE_MB}MB.'}), 400

    # ── Collect Metadata ──
    metadata = {
        'gender': request.form.get('gender'),
        'age': request.form.get('age'),
        'sun_exposure': request.form.get('sun'),
        'family_history': request.form.get('history'),
        'has_changed': request.form.get('change'),
        'symptoms': request.form.get('symptoms')
    }

    # ── Save & check quality ──
    image_path = save_upload(file)

    quality = check_image_quality(image_path)
    if quality['score'] < 40:
        cleanup_file(image_path, delay=0)
        return jsonify({
            'error': 'Image quality too low',
            'quality': quality,
            'message': 'Please upload a clearer, well-lit image of the lesion.'
        }), 422

    # ── Run prediction ──
    try:
        result = predictor.predict(image_path, detailed=False)
        if 'error' in result:
            return jsonify(result), 500

        result = adjust_risk_with_metadata(result, metadata)
        result['quality'] = quality
        result['metadata'] = metadata
        return jsonify(result), 200
    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500
    finally:
        cleanup_file(image_path, delay=30)


@app.route('/api/scan/detailed', methods=['POST'])
def detailed_scan():
    """
    Detailed scan endpoint. Queues a background job.
    Returns: job_id to poll /api/scan/status/<job_id>
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use JPG or PNG.'}), 400

    metadata = {
        'gender': request.form.get('gender'),
        'age': request.form.get('age'),
        'sun_exposure': request.form.get('sun'),
        'family_history': request.form.get('history'),
        'has_changed': request.form.get('change'),
        'symptoms': request.form.get('symptoms')
    }

    image_path = save_upload(file)
    quality = check_image_quality(image_path)

    if quality['score'] < 40:
        cleanup_file(image_path, delay=0)
        return jsonify({'error': 'Image quality too low', 'quality': quality}), 422

    job_id = uuid.uuid4().hex
    detailed_jobs[job_id] = {
        'status': 'queued',
        'created_at': datetime.utcnow().isoformat(),
        'quality': quality,
        'metadata': metadata
    }

    thread = threading.Thread(target=run_detailed_analysis, args=(job_id, image_path), daemon=True)
    thread.start()

    return jsonify({'job_id': job_id, 'status': 'queued', 'quality': quality}), 202


@app.route('/api/scan/status/<job_id>', methods=['GET'])
def scan_status(job_id):
    """Poll the status of a detailed scan job."""
    job = detailed_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job), 200


# ── Run ────────────────────────────────────────────────────────────────────────
# Ensure this is __name__ and __main__ (double underscores)
if __name__ == '__main__':
    # Change 5000 to 5001
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)