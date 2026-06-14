import tensorflow as tf
import os

model_path = 'model/skin_lens_cnn.h5'

if os.path.exists(model_path):
    print(f"File Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        print(f"Model Input Shape: {model.input_shape}")
        # Check if weights are zero or random
        weights_sum = sum([tf.reduce_sum(w).numpy() for w in model.get_weights()])
        print(f"Weights Checksum: {weights_sum:.4f}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print("❌ File not found! Check the path.")