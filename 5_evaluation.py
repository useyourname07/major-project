import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
MODEL_SAVE_PATH = "C:/Users/User/Desktop/minor/Detection leaf disease/Datasets/Plant Village dataset/saved_model/leaf_classifier_v1.h5"
TEST_DIR = "C:/Users/User/Desktop/minor/Detection leaf disease/Datasets/Plant Village dataset/split_dataset/test"

# Config
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_test_data(test_dir, img_size, batch_size):
    """Loads and preprocesses test data."""
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    return datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

def load_model(model_path):
    """Loads a trained model from disk."""
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded.")
    return model

def evaluate_model(model, test_generator):
    """Generates evaluation metrics and plots."""
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"\n📊 Evaluation Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}\n")

    print("🧾 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_generator = load_test_data(TEST_DIR, IMG_SIZE, BATCH_SIZE)
    model = load_model(MODEL_SAVE_PATH)
    print("\n🚀 Evaluating on Test Dataset...")
    evaluate_model(model, test_generator)