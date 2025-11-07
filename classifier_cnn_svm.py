from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.datasets import mnist
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

def train_cnn_svm():
    print("🚀 Starting CNN+SVM training...")

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # ✅ CNN feature extractor
    feature_extractor = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu')  # feature vector layer
    ])

    # CNN for training (with softmax head)
    cnn_full = Sequential([
        feature_extractor,
        Dense(10, activation='softmax')
    ])

    # ✅ Train CNN normally
    cnn_full.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_full.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1)

    # ✅ Extract features using trained feature_extractor (shared weights!)
    print("🔍 Extracting CNN features...")
    X_train_feats = feature_extractor.predict(x_train)
    X_test_feats = feature_extractor.predict(x_test)

    # ✅ Reduce feature dimensions
    print("⚙️ Reducing feature dimensions with PCA...")
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train_feats)
    X_test_pca = pca.transform(X_test_feats)

    # ✅ Train SVM
    print("🧠 Training SVM on CNN features...")
    svm = SVC(kernel='rbf', C=10, gamma=0.01)
    svm.fit(X_train_pca, y_train)

    # ✅ Evaluate
    y_pred = svm.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"✅ CNN+SVM Test Accuracy: {acc:.4f}")

    # ✅ Save models
    joblib.dump((feature_extractor, pca, svm), "cnn_svm_model.pkl")
    print("💾 Saved CNN+SVM model successfully as cnn_svm_model.pkl")

# Run training
if __name__ == "__main__":
    train_cnn_svm()
