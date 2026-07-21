import tensorflow as tf

print("Loading model...")

model = tf.keras.models.load_model("models/densenet121_leukemia.keras")

print("Model Loaded Successfully!")