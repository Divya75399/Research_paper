# import tensorflow as tf
# from tensorflow.keras.preprocessing import image_dataset_from_directory
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt

# # Dataset path
# train_path = "dataset_classification/train"
# val_path = "dataset_classification/val"

# # Load Training Dataset
# train_dataset = image_dataset_from_directory(
#     train_path,
#     image_size=(224, 224),
#     batch_size=32
# )

# # Load Validation Dataset
# val_dataset = image_dataset_from_directory(
#     val_path,
#     image_size=(224, 224),
#     batch_size=32
# )

# # Load EfficientNetB0
# base_model = EfficientNetB0(
#     weights="imagenet",
#     include_top=False,
#     input_shape=(224,224,3)
# )

# base_model.trainable = False

# # Build Model
# model = models.Sequential([
#     base_model,
#     layers.GlobalAveragePooling2D(),
#     layers.Dense(128, activation="relu"),
#     layers.Dropout(0.3),
#     layers.Dense(2, activation="softmax")
# ])

# # Compile Model
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# # Train Model
# history = model.fit(
#     train_dataset,
#     validation_data=val_dataset,
#     epochs=20
# )

# # Save Model
# model.save("leukemia_model.keras")

# print("Model Saved Successfully!")

# # Accuracy Graph
# plt.plot(history.history["accuracy"], label="Training Accuracy")
# plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
# plt.legend()
# plt.title("Accuracy")
# plt.show()

# # Loss Graph
# plt.plot(history.history["loss"], label="Training Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.legend()
# plt.title("Loss")
# plt.show()