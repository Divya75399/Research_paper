import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# ==========================================
# Load Trained Model
# ==========================================

print("=" * 50)
print("Loading Trained DenseNet121 Model...")
print("=" * 50)

model = tf.keras.models.load_model(
    "models/densenet121_leukemia.keras"
)

print("Model Loaded Successfully!\n")

# ==========================================
# Load Validation Dataset
# ==========================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("Loading Validation Dataset...\n")

test_dataset = image_dataset_from_directory(
    "dataset_classification/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names

print("\nClasses :", class_names)
print()

# ==========================================
# Prediction
# ==========================================

y_true = []
y_pred = []
y_prob = []

batch = 1

print("=" * 50)
print("Predicting Images...")
print("=" * 50)

for images, labels in test_dataset:

    print(f"Processing Batch {batch}")

    predictions = model.predict(images, verbose=0)

    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)
    y_prob.extend(predictions[:, 1])

    batch += 1

print("\nPrediction Completed!\n")

# ==========================================
# Metrics
# ==========================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="binary"
)

recall = recall_score(
    y_true,
    y_pred,
    average="binary"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="binary"
)

print("=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")

# ==========================================
# Classification Report
# ==========================================

print("\n")
print("=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))

plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")

plt.colorbar()

plt.xticks([0,1], class_names)
plt.yticks([0,1], class_names)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i,j]),
            ha="center",
            va="center",
            color="black",
            fontsize=12
        )

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()

print("Confusion Matrix Saved as confusion_matrix.png")

# ==========================================
# ROC Curve
# ==========================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_prob
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    'r--'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig("roc_curve.png")

plt.show()

print("ROC Curve Saved as roc_curve.png")

print("\n")
print("=" * 50)
print("Evaluation Completed Successfully!")
print("=" * 50)