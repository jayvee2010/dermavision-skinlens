import os
import uuid
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Direct structural imports matching teammate's file
import predict
from image_quality import check_image_quality

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_upload(file):
    ext = file.filename.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(path)
    return path

def cleanup_file(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': predict.is_model_loaded()})

@app.route('/api/scan/quick', methods=['POST'])
def quick_scan():
    if not predict.is_model_loaded():
        return jsonify({'error': 'AI Model engine is offline.', 'message': 'The model file could not be initialized.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type.'}), 400

    image_path = save_upload(file)

    try:
        # 1. Quality Validation Gate
        quality = check_image_quality(image_path)
        if quality['score'] < 40:
            cleanup_file(image_path)
            return jsonify({
                'error': 'Image quality too low',
                'quality': quality,
                'message': 'Please upload a clearer, well-lit image.'
            }), 422

        # 2. Extract Answers from incoming FormData mapping matching index.html
        answers = {
            "age": request.form.get("age"),
            "sun": request.form.get("sun"),
            "history": request.form.get("history"),
            "duration": request.form.get("duration"),
            "change": request.form.get("change"),
            "existing": request.form.get("existing"),
            "itching": request.form.get("symptoms"), # maps frontend symptoms -> teammate's itching
            "bleeding": request.form.get("bleeding"),
            "oozing": request.form.get("oozing")
        }

        # 3. Process Hybrid Inference Analysis
        analysis = predict.analyze_patient(image_path, answers)

        # 4. Format Output Structure to satisfy what frontend JavaScript components expect
        response_payload = {
            "top_class": analysis["image_assessment"],
            "confidence": f"{analysis['final_risk']}%",
            "risk_level": analysis["risk_level"].lower(),
            "risk_label": f"{analysis['risk_level']} Risk Structure",
            "quality": quality,
            "classes": [
                {"name": "Hybrid Risk Calculation Score", "probability": analysis["final_risk"], "color": "#f5a623"},
                {"name": "Pure Machine Learning Output Index", "probability": analysis["cnn_risk"], "color": "#ff6b6b"},
                {"name": "Epidemiological Metadata Weight", "probability": analysis["questionnaire_risk"], "color": "#c8f560"}
            ],
            "guidance": {
                "en": f"Analysis Summary: {', '.join(analysis['explanations']) if analysis['explanations'] else 'No high risk clinical flags found.'} Note: {analysis['disclaimer']}",
                "hi": f"जोखिम मूल्यांकन विश्लेषण। नोट: {analysis['disclaimer']}",
                "mr": f"जोखीम मूल्यमापन विश्लेषण. टीप: {analysis['disclaimer']}"
            }
        }

        return jsonify(response_payload), 200

    except Exception as e:
        logger.exception("Engine process exception path")
        return jsonify({'error': f'Analysis engine failure processing request: {str(e)}'}), 500
    finally:
        cleanup_file(image_path)

if __name__ == '__main__':
    import os
    # Render assigns a dynamic port via the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    # Must bind to 0.0.0.0 in production so external traffic can reach it
    app.run(host="0.0.0.0", port=port)