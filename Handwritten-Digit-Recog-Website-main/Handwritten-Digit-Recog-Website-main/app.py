from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import joblib
import os
from scipy.ndimage import measurements, interpolation
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model variables
cnn_model = None
pca_model = None
svm_model = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_models():
    global cnn_model, pca_model, svm_model
    try:
        cnn_model, pca_model, svm_model = joblib.load("cnn_svm_model.pkl")
        print("✅ Models loaded successfully")
        print(f"CNN Model: {type(cnn_model)}")
        print(f"PCA Model: {type(pca_model)}")
        print(f"SVM Model: {type(svm_model)}")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False


def preprocess_digit(digit_img):
    h, w = digit_img.shape
    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w, new_h = 20, int(20 / aspect_ratio)
    else:
        new_h, new_w = 20, int(20 * aspect_ratio)
    digit = cv2.resize(digit_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit

    cy, cx = measurements.center_of_mass(canvas)
    shift_x = np.round(14 - cx).astype(int)
    shift_y = np.round(14 - cy).astype(int)
    canvas = interpolation.shift(canvas, shift=[shift_y, shift_x], mode='constant')

    canvas = canvas.astype("float32") / 255.0
    canvas = canvas.reshape(1, 28, 28, 1)
    return canvas


def predict_single_digit(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    processed = preprocess_digit(thresh)

    features = cnn_model.predict(processed, verbose=0)
    features_pca = pca_model.transform(features)

    pred = svm_model.predict(features_pca)[0]
    probas = svm_model.decision_function(features_pca)
    confidence = float(np.max(probas)) if probas.ndim > 1 else float(probas)

    return {
        "prediction": int(pred),
        "confidence": round(confidence, 4)
    }


def predict_multiple_digits(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image")

    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No digits detected")

    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    results = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 5 or h < 5:
            continue

        roi = thresh[y:y+h, x:x+w]
        processed = preprocess_digit(roi)

        features = cnn_model.predict(processed, verbose=0)
        features_pca = pca_model.transform(features)

        pred = svm_model.predict(features_pca)[0]
        probas = svm_model.decision_function(features_pca)
        confidence = float(np.max(probas)) if probas.ndim > 1 else float(probas)

        results.append({
            "bounding_box": [int(x), int(y), int(w), int(h)],
            "prediction": int(pred),
            "confidence": round(confidence, 4)
        })

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

    # Convert annotated image to base64
    _, buffer = cv2.imencode('.png', color_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return results, img_base64


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG'}), 400

    prediction_type = request.form.get('type', 'single')

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        
        results, annotated_img = predict_multiple_digits(filepath)
        os.remove(filepath)
        return jsonify({
            'digits': results,
            'annotated_image': annotated_img
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)