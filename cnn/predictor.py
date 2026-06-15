"""
predictor.py — Skin Lens CNN Inference Engine
=============================================
Loads the trained Keras/TensorFlow model and runs predictions.

Model output classes (adjust to match your training labels):
  0 → Benign Nevus       (Low Risk)
  1 → Seborrheic Keratosis (Low Risk)
  2 → Dysplastic Nevus   (Moderate Risk)
  3 → BCC                (Moderate Risk)
  4 → Melanoma           (High Risk)
  5 → SCC                (High Risk)

To swap in a PyTorch model, replace the TensorFlow sections
with torch.load() and model.eval() equivalents — the rest stays the same.
"""

import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Class definitions ──────────────────────────────────────────────────────────
CLASSES = [
    {'id': 0, 'name': 'benign_nevus',          'risk': 'low',      'display': 'Benign Nevus'},
    {'id': 1, 'name': 'seborrheic_keratosis',  'risk': 'low',      'display': 'Seb. Keratosis'},
    {'id': 2, 'name': 'dysplastic_nevus',      'risk': 'moderate', 'display': 'Dysplastic Nevus'},
    {'id': 3, 'name': 'bcc',                   'risk': 'moderate', 'display': 'BCC'},
    {'id': 4, 'name': 'melanoma',              'risk': 'high',     'display': 'Melanoma'},
    {'id': 5, 'name': 'scc',                   'risk': 'high',     'display': 'SCC'},
]

RISK_META = {
    'low':      {'label': 'Low Risk',      'color': '#c8f560'},
    'moderate': {'label': 'Moderate Risk', 'color': '#f5a623'},
    'high':     {'label': 'High Risk',     'color': '#ff6b6b'},
}

INPUT_SIZE = (224, 224)   # Must match what your model was trained on


# ── Helper: Safe Resampling ────────────────────────────────────────────────────
try:
    # Pillow 10+
    RESAMPLE_MODE = Image.Resampling.LANCZOS
except AttributeError:
    # Older Pillow versions
    RESAMPLE_MODE = getattr(Image, 'LANCZOS', getattr(Image, 'ANTIALIAS', Image.BICUBIC))


# ── Predictor class ────────────────────────────────────────────────────────────
class SkinLensPredictor:
    def __init__(self, model_path: str, force_demo: bool = False):
        self.model_path = model_path
        self.model = None
        self.input_size = INPUT_SIZE  # Default fallback
        self.force_demo = force_demo
        if not self.force_demo:
            self._load_model()

    def _load_model(self):
        """Load the saved Keras model. Falls back gracefully if not found."""
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"✅ Model loaded from {self.model_path}")
            
            # Auto-detect input shape from the loaded model
            if hasattr(self.model, 'input_shape'):
                shape = self.model.input_shape
                # Handle list of shapes (for multi-input models) or single shape tuple
                if isinstance(shape, list): shape = shape[0]
                
                if shape and len(shape) == 4 and shape[1] is not None and shape[2] is not None:
                    self.input_size = (shape[1], shape[2])
                    logger.info(f"   Auto-detected input size: {self.input_size}")
        except Exception as e:
            logger.error(f"❌ Failed to load model from '{self.model_path}': {e}")
            raise e

    def is_loaded(self) -> bool:
        return self.model is not None

    def preprocess(self, image_path: str, augment: bool = False) -> np.ndarray:
        """
        Load and preprocess image for CNN input.
        - Resize to model's expected input size
        - Normalize pixel values to [0, 1]
        - Optionally apply test-time augmentation (TTA) for detailed scan
        """
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.input_size, RESAMPLE_MODE)
        # Normalize to [-1, 1] for MobileNetV2
        arr = np.array(img, dtype=np.float32) / 127.5 - 1.0

        if augment:
            # TTA: average over original + flipped variants
            flipped_h = arr[:, ::-1, :]
            flipped_v = arr[::-1, :, :]
            flipped_hv = arr[::-1, ::-1, :]
            return np.stack([arr, flipped_h, flipped_v, flipped_hv])  # (4, H, W, 3)
        else:
            return np.expand_dims(arr, axis=0)  # (1, H, W, 3)

    def predict(self, image_path: str, detailed: bool = False) -> dict:
        """
        Run inference and return structured result dict.
        Falls back to demo predictions if model is not loaded.
        """
        if self.force_demo:
            return self._demo_prediction()

        if self.model is None:
            raise RuntimeError("Model is not loaded. Please ensure the model file exists and is valid.")

        try:
            input_arr = self.preprocess(image_path, augment=detailed)

            # Run inference
            raw = self.model.predict(input_arr, verbose=0)  # shape: (N, num_classes)

            # Average over TTA passes for detailed scan
            probs = raw.mean(axis=0)  # shape: (num_classes,)

            return self._build_result(probs)

        except Exception as e:
            logger.exception("Inference error")  # Prints full traceback to console
            raise

    def _build_result(self, probs: np.ndarray) -> dict:
        """Convert raw softmax probabilities to structured API response."""
        top_idx = int(np.argmax(probs))
        top_class = CLASSES[top_idx]
        risk_level = top_class['risk']
        risk_info = RISK_META[risk_level]

        # Build per-class confidence list (sorted by probability desc)
        class_scores = sorted(
            [{'name': c['display'], 'probability': round(float(p) * 100, 1), 'color': RISK_META[c['risk']]['color']}
             for c, p in zip(CLASSES, probs)],
            key=lambda x: x['probability'],
            reverse=True
        )

        return {
            'risk_level':   risk_level,
            'risk_label':   risk_info['label'],
            'top_class':    top_class['display'],
            'confidence':   f"{round(float(probs[top_idx]) * 100, 1)}%",
            'classes':      class_scores,
            'guidance':     self._guidance(risk_level),
        }

    def _guidance(self, risk_level: str) -> dict:
        """Return risk guidance text in all three supported languages."""
        return {
            'en': {
                'low':      'This lesion appears likely benign. Continue regular self-monitoring and schedule a routine dermatology check at your next visit. No immediate action required.',
                'moderate': 'This lesion shows features that warrant medical attention. Please consult a dermatologist within 2–4 weeks. Do not self-medicate or delay.',
                'high':     'This lesion shows characteristics associated with higher-risk conditions. Please consult a dermatologist urgently — within the next few days. Prompt professional evaluation is strongly advised.',
            }[risk_level],
            'hi': {
                'low':      'यह घाव दृश्य विशेषताओं के आधार पर संभवतः सौम्य प्रतीत होता है। नियमित स्व-निगरानी जारी रखें और अगली बार किसी चर्मरोग विशेषज्ञ से जाँच करवाएं।',
                'moderate': 'इस घाव में ऐसी विशेषताएं हैं जो चिकित्सीय ध्यान की आवश्यकता दर्शाती हैं। कृपया 2–4 सप्ताह के भीतर किसी चर्मरोग विशेषज्ञ से मिलें।',
                'high':     'इस घाव में उच्च जोखिम वाली स्थितियों से जुड़ी विशेषताएं हैं। कृपया तुरंत किसी चर्मरोग विशेषज्ञ से मिलें।',
            }[risk_level],
            'mr': {
                'low':      'ही जखम सौम्य असण्याची शक्यता आहे. नियमित स्व-निरीक्षण सुरू ठेवा आणि पुढील भेटीत त्वचारोगतज्ज्ञाकडे तपासणी करा.',
                'moderate': 'या जखमेत वैद्यकीय लक्ष आवश्यक असलेली वैशिष्ट्ये आहेत. कृपया 2–4 आठवड्यांच्या आत त्वचारोगतज्ज्ञाचा सल्ला घ्या.',
                'high':     'या जखमेत उच्च धोकादायक स्थितींशी संबंधित वैशिष्ट्ये आहेत. कृपया तातडीने त्वचारोगतज्ज्ञाला भेट द्या.',
            }[risk_level],
        }

    def _demo_prediction(self) -> dict:
        """Return a realistic-looking demo result when model is not loaded."""
        import random
        risk_level = random.choice(['low', 'moderate', 'high'])
        # Simulate realistic probability distribution
        probs = np.random.dirichlet(np.ones(len(CLASSES)) * 0.5)
        # Boost the top class probability to be realistic
        top_class_idx = {'low': 0, 'moderate': 2, 'high': 4}[risk_level]
        probs[top_class_idx] += 0.5
        probs /= probs.sum()
        result = self._build_result(probs)
        result['demo_mode'] = True
        return result