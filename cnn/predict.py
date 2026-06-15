import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)

model = tf.keras.models.load_model("skinlens_model.keras")

def predict_image(image_path):

    image = Image.open(image_path)

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    image = np.array(image, dtype=np.float32)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    probability = model.predict(image, verbose=0)[0][0]

    result = {
        "prediction": "Melanoma" if probability >= 0.5 else "Benign",
        "confidence": float(probability)
    }

    return result


if __name__ == "__main__":
    result = predict_image("sample.jpg")
    print(result)
    