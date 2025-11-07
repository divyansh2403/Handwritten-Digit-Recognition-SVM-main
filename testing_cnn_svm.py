import cv2
import numpy as np
import joblib
import sys
from scipy.ndimage import measurements, interpolation

# === LOAD CNN+PCA+SVM MODEL ===
def load_cnn_svm_model(path="cnn_svm_model.pkl"):
    cnn, pca, svm = joblib.load(path)
    print("✅ Loaded CNN+SVM model successfully.")
    return cnn, pca, svm


# === PREPROCESSING FUNCTION (MNIST style) ===
def preprocess_digit(digit_img):
    # Resize while preserving aspect ratio
    h, w = digit_img.shape
    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w, new_h = 20, int(20 / aspect_ratio)
    else:
        new_h, new_w = 20, int(20 * aspect_ratio)
    digit = cv2.resize(digit_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center in 28x28 black frame
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit

    # Center-of-mass shift
    cy, cx = measurements.center_of_mass(canvas)
    shift_x = np.round(14 - cx).astype(int)
    shift_y = np.round(14 - cy).astype(int)
    canvas = interpolation.shift(canvas, shift=[shift_y, shift_x], mode='constant')

    # Normalize for CNN input
    canvas = canvas.astype("float32") / 255.0
    canvas = canvas.reshape(1, 28, 28, 1)
    return canvas


# === MULTI-DIGIT DETECTION + PREDICTION ===
def predict_multi_digit_image(img_path, show_output=True, save_output=True):
    """
    Detect multiple digits, preprocess each to MNIST format, and predict using CNN+SVM.
    """
    cnn, pca, svm = load_cnn_svm_model()

    # Read grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"❌ Could not read image: {img_path}")

    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Preprocessing: Blur + Threshold
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours (each digit)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("❌ No digits detected!")

    # Sort contours left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    results = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 5 or h < 5:
            continue

        roi = thresh[y:y+h, x:x+w]
        processed = preprocess_digit(roi)

        # Get CNN features
        features = cnn.predict(processed, verbose=0)
        features_pca = pca.transform(features)

        # Predict with SVM
        pred = svm.predict(features_pca)[0]
        probas = svm.decision_function(features_pca)
        confidence = float(np.max(probas)) if probas.ndim > 1 else float(probas)

        # Store results
        results.append({
            "bounding_box": [int(x), int(y), int(w), int(h)],
            "prediction": int(pred),
            "confidence": round(confidence, 4)
        })

        # Draw box + label
        cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            color_img,
            f"{pred} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    # Save/show visualization
    if save_output:
        out_path = "multi_digit_detected.png"
        cv2.imwrite(out_path, color_img)
        print(f"🖼️ Saved visualization as {out_path}")

    if show_output:
        cv2.imshow("Detected Digits", color_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results


# === SINGLE IMAGE PREDICTION ===
def predict_single_image(img_path):
    cnn, pca, svm = load_cnn_svm_model()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    processed = preprocess_digit(thresh)

    features = cnn.predict(processed, verbose=0)
    features_pca = pca.transform(features)

    pred = svm.predict(features_pca)[0]
    probas = svm.decision_function(features_pca)
    confidence = float(np.max(probas)) if probas.ndim > 1 else float(probas)

    return {
        "prediction": int(pred),
        "confidence": round(confidence, 4)
    }


# === MAIN SCRIPT HANDLER ===
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python cnnsvm_predict.py predict image.png")
        print("  python cnnsvm_predict.py predict-multi digits.png")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    img_path = sys.argv[2]

    if cmd == "predict":
        result = predict_single_image(img_path)
        print("\n🔮 Prediction Result")
        print("Digit:", result["prediction"])
        print("Confidence:", result["confidence"])

    elif cmd == "predict-multi":
        results = predict_multi_digit_image(img_path)
        print("\n🔢 Multiple Digit Predictions:")
        for i, r in enumerate(results):
            print(f"Digit {i+1}: {r['prediction']} (conf: {r['confidence']}) Box: {r['bounding_box']}")
    else:
        print("❌ Unknown command:", cmd)
