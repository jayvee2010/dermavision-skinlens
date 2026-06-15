import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Throttle full TensorFlow memory to prevent Render RAM crashes
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ==========================
# CONFIG
# ==========================
IMG_SIZE = (224, 224)

# ==========================
# LOAD MODEL 
# ==========================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'skinlens_model.tflite')

try:
    print("Initializing modern TFLite Interpreter...")
    # Use modern TF to securely parse the tiny TFLite file
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"❌ Critical Error loading lightweight model: {str(e)}")
    interpreter = None

# Helper method for app.py
def is_model_loaded():
    return interpreter is not None

# ==========================
# IMAGE PREDICTION
# ==========================
def predict_image(image_path):
    if interpreter is None:
        return {"image_assessment": "Model Engine Offline", "cnn_risk": 0.0}

    # Preprocess image precisely as trained
    image = Image.open(image_path)
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image = np.array(image, dtype=np.float32)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Invoke the TFLite engine inference pipeline
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    
    probability = float(interpreter.get_tensor(output_details[0]['index'])[0][0])

    cnn_risk = round(probability * 100, 2)

    if probability >= 0.5:
        image_assessment = "Melanoma-like pattern detected"
    else:
        image_assessment = "No melanoma-like pattern detected"

    return {
        "image_assessment": image_assessment,
        "cnn_risk": cnn_risk
    }

# ==========================
# QUESTIONNAIRE SCORING
# ==========================
def calculate_questionnaire_risk(answers):
    score = 0
    explanations = []

    age = answers.get("age")
    sun = answers.get("sun")
    history = answers.get("history")
    duration = answers.get("duration")
    change = answers.get("change")
    existing = answers.get("existing")
    itching = answers.get("itching")
    bleeding = answers.get("bleeding")
    oozing = answers.get("oozing")

    # AGE
    if age == "36-50":
        score += 5
    elif age == "51-65":
        score += 10
    elif age == "65+":
        score += 15

    # SUN EXPOSURE
    if sun == "moderate":
        score += 5
    elif sun == "high":
        score += 10

    # FAMILY HISTORY
    if history == "yes":
        score += 15
        explanations.append("Family history of skin cancer")

    # DURATION OF CHANGE
    if duration == "slow":
        score += 5
    elif duration == "rapid":
        score += 15
        explanations.append("Rapid lesion change")

    # LESION CHANGE
    if change == "yesd":
        score += 20
        explanations.append("Lesion is continuing to develop")
    elif change == "unknown":
        score += 5

    # EXISTING MOLE
    if existing == "yes":
        score += 10
        explanations.append("Developed from an existing mole")
    elif existing == "unknown":
        score += 5

    # ITCHING
    if itching == "yes":
        score += 10
        explanations.append("Itching reported")

    # BLEEDING
    if bleeding == "yes":
        score += 20
        explanations.append("Bleeding reported")
    elif bleeding == "unknown":
        score += 5

    # OOZING
    if oozing == "yes":
        score += 15
        explanations.append("Oozing reported")
    elif oozing == "unknown":
        score += 5

    questionnaire_risk = min(round((score / 120) * 100), 100)

    return {
        "questionnaire_risk": questionnaire_risk,
        "explanations": explanations
    }

# ==========================
# FINAL ANALYSIS
# ==========================
def analyze_patient(image_path, answers):
    cnn_result = predict_image(image_path)
    questionnaire_result = calculate_questionnaire_risk(answers)

    cnn_risk = cnn_result["cnn_risk"]
    questionnaire_risk = questionnaire_result["questionnaire_risk"]

    final_risk = round((0.7 * cnn_risk) + (0.3 * questionnaire_risk), 2)

    if final_risk < 30:
        risk_level = "Low"
    elif final_risk < 60:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    explanations = list(questionnaire_result["explanations"])

    if cnn_risk > 70:
        explanations.append("CNN detected suspicious visual patterns")

    return {
        "image_assessment": cnn_result["image_assessment"],
        "cnn_risk": cnn_risk,
        "questionnaire_risk": questionnaire_risk,
        "final_risk": final_risk,
        "risk_level": risk_level,
        "explanations": explanations,
        "disclaimer": "This tool provides a risk assessment and is not a medical diagnosis. Please consult a qualified dermatologist for professional evaluation."
    }