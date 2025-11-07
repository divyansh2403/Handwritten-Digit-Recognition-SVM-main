"""
MNIST SVM Classifier with PCA

USAGE:

1. Train and save the model (first time only):
   python main.py train

2. Evaluate the saved model:
   python main.py eval

3. Predict a digit from an image:
   python main.py predict path/to/image.png

"""

import numpy as np
import joblib
import json
import sys
import cv2
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
from tensorflow.keras.datasets import mnist
from PIL import Image


# ==========================
# 1. Train + Save Model
# ==========================
def train_and_save():
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GridSearchCV

    # Load MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    # PCA + SVM
    pca = PCA(n_components=50)
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True)
    pipe = make_pipeline(pca, svm)

    # Hyperparameter grid
    param_grid = {
        "svc__C": [10, 50],
        "svc__gamma": [0.01, 0.05]
    }
    grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, verbose=2)

    print("🚀 Training model...")
    grid.fit(x_train[:20000], y_train[:20000])  # subset for speed

    best_model = grid.best_estimator_

    # Save model
    joblib.dump(best_model, "mnist_svm_pca.pkl")
    print("✅ Model saved as mnist_svm_pca.pkl")

    # Evaluate immediately
    evaluate_model()


# ==========================
# 2. Evaluate Saved Model
# ==========================
def evaluate_model():
    # Load test data
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    # Load model
    model = joblib.load("mnist_svm_pca.pkl")

    # Predict
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
    class_metrics = {
        str(i): {
            "precision": round(precision[i], 3),
            "recall": round(recall[i], 3),
            "f1": round(f1[i], 3),
            "support": int(support[i])
        }
        for i in range(10)
    }

    print("\n📊 Evaluation Results")
    print("Accuracy:", round(acc, 4))

    # Save stats
    with open("mnist_svm_stats.json", "w") as f:
        json.dump({
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm,
            "class_metrics": class_metrics
        }, f, indent=4)

    print("✅ Evaluation metrics saved to mnist_svm_stats.json")


# ==========================
# 3. Preprocessing for Uploaded Images
# ==========================
def preprocess_image(img_path):
    img = Image.open(img_path).convert("L")
    img = np.array(img)

    # Invert if needed (MNIST is white digit on black background)
    if img.mean() > 127:
        img = 255 - img

    # Threshold (binary image)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find bounding box of digit
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y:y+h, x:x+w]

    # Resize digit to 20x20
    digit = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)

    # Pad to 28x28
    padded = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    padded[y_offset:y_offset+20, x_offset:x_offset+20] = digit

    # Normalize
    padded = padded.astype("float32") / 255.0

    return padded.reshape(1, 784)


# ==========================
# 4. Predict Function
# ==========================
def predict_image(img_path):
    model = joblib.load("mnist_svm_pca.pkl")
    processed = preprocess_image(img_path)
    pred = model.predict(processed)[0]
    probas = model.predict_proba(processed)[0]

    return {
        "prediction": int(pred),
        "confidence": round(float(np.max(probas)), 4),
        "all_probs": {str(i): round(float(probas[i]), 4) for i in range(10)}
    }


# ==========================
# USAGE
# ==========================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py train                # Train and save model")
        print("  python main.py eval                 # Evaluate saved model")
        print("  python main.py predict image.png    # Predict digit from image")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "train":
        train_and_save()
    elif cmd == "eval":
        evaluate_model()
    elif cmd == "predict":
        if len(sys.argv) < 3:
            print("Please provide the path to an image.")
            sys.exit(1)
        result = predict_image(sys.argv[2])
        print("\n🔮 Prediction Result")
        print("Digit:", result["prediction"])
        print("Confidence:", result["confidence"])
        print("All probabilities:", result["all_probs"])
    else:
        print("Unknown command:", cmd)
