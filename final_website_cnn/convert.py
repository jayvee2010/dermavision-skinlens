import tensorflow as tf

# 1. Load your existing heavy model
keras_model_path = 'final_website_cnn/model/skinlens_model.keras'
print("Loading Keras model...")
model = tf.keras.models.load_model(keras_model_path)

# 2. Convert it to the lightweight TFLite format
print("Converting to TFLite format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3. Save the new compact file
tflite_model_path = 'final_website_cnn/model/skinlens_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Success! Lightweight model saved to {tflite_model_path}")