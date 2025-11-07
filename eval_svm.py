# svm_pca_advanced_evaluation.py
# Advanced standalone evaluation of the PCA+SVM MNIST model

import joblib
from tensorflow.keras.datasets import mnist
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import pandas as pd
from itertools import cycle

# ---------------------- Configuration ----------------------
# t-SNE is slow. We'll use a subset of the test data.
TSNE_SAMPLES = 2500
MODEL_FILE = "mnist_svm_pca.pkl"

# ---------------------- Load Model ----------------------
print(f"🔄 Loading PCA+SVM model from '{MODEL_FILE}'...")
try:
    # The model is an scikit-learn Pipeline
    model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    print(f"❌ ERROR: Model file not found: '{MODEL_FILE}'")
    print("Please run 'python main.py train' first to create the model.")
    exit()

# Extract the PCA step for analysis
try:
    pca = model.named_steps['pca']
    svm = model.named_steps['svc']
except KeyError:
    print("❌ ERROR: Model pipeline does not contain 'pca' or 'svc'.")
    print("This script is designed for the pipeline from 'main.py'.")
    exit()

# ---------------------- Load MNIST Test Data ----------------------
print("📥 Loading MNIST test data...")
# Load raw data for plotting misclassified images
(_, _), (x_test_raw, y_test) = mnist.load_data()
# Load and process data for prediction
x_test_normalized = x_test_raw.reshape(-1, 784).astype("float32") / 255.0

# ---------------------- Transformation & Prediction ----------------------
print("⚙️ Applying PCA transformation...")
# We need the PCA-transformed data for t-SNE
X_test_pca = pca.transform(x_test_normalized)

print("🧠 Predicting with SVM...")
y_pred = model.predict(x_test_normalized)
# Get probabilities for ROC curve (since probability=True was set)
y_score = model.predict_proba(x_test_normalized)

# ---------------------- Basic Metrics ----------------------
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ PCA+SVM Test Accuracy: {acc:.4f}")

print("\n📈 Classification Report (Text):")
class_report_text = classification_report(y_test, y_pred)
print(class_report_text)
class_report_dict = classification_report(y_test, y_pred, output_dict=True)

# ---------------------- Plot 1: Confusion Matrix ----------------------
print("\n...Generating Plot 1: Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Confusion Matrix - PCA+SVM", fontsize=14)
plt.savefig("svm_pca_confusion_matrix.png")
plt.show()

# ---------------------- Plot 2: Per-Class Metrics Bar Chart ----------------------
print("...Generating Plot 2: Per-Class Metrics Bar Chart")
try:
    df_report = pd.DataFrame(class_report_dict).iloc[:-1, :].T
    # Exclude avg/support rows
    df_report = df_report.drop(['accuracy', 'macro avg', 'weighted avg'])
    
    df_report.plot(kind='bar', figsize=(14, 7), rot=0)
    plt.title("Per-Class Performance (Precision, Recall, F1-Score)", fontsize=16)
    plt.xlabel("Digit Class", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0.9, 1.0) # Zoom in on the high scores
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.savefig("svm_pca_class_metrics.png")
    plt.show()
except Exception as e:
    print(f"Could not plot per-class metrics: {e}")

# ---------------------- Plot 3: PCA Explained Variance ----------------------
print("...Generating Plot 3: PCA Explained Variance")
plt.figure(figsize=(14, 6))

# Plot 1: Explained Variance by Component
plt.subplot(1, 2, 1)
plt.plot(pca.explained_variance_ratio_, 'bo-', markersize=5)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance by Component")
plt.grid(True)

# Plot 2: Cumulative Explained Variance
plt.subplot(1, 2, 2)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumulative_variance, 'ro-', markersize=5)
total_var = cumulative_variance[-1]
plt.axhline(y=total_var * 0.95, color='g', linestyle='--', label=f'95% of total ({total_var*0.95:.2f})')
plt.axhline(y=total_var, color='k', linestyle='--', label=f'Total Variance Captured ({total_var:.2f})')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.grid(True)
plt.legend()

plt.suptitle(f"PCA Analysis ({pca.n_components_} Components)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("svm_pca_variance.png")
plt.show()

# ---------------------- Plot 4: t-SNE Feature Visualization ----------------------
print(f"...Generating Plot 4: t-SNE Feature Visualization (on {TSNE_SAMPLES} samples)")
try:
    # Subsample the PCA-transformed data
    np.random.seed(42) # for reproducibility
    indices = np.random.choice(range(len(X_test_pca)), TSNE_SAMPLES, replace=False)
    X_test_pca_sub = X_test_pca[indices]
    y_test_sub = y_test[indices]

    # Run t-SNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(X_test_pca_sub)
    
    # Plot
    plt.figure(figsize=(14, 10))
    cmap = plt.get_cmap('tab10', 10)
    for i in range(10):
        class_indices = np.where(y_test_sub == i)
        plt.scatter(
            tsne_results[class_indices, 0], 
            tsne_results[class_indices, 1], 
            c=[cmap(i)], 
            label=str(i), 
            s=10 # smaller dots
        )
        
    plt.title("t-SNE Visualization of PCA-Transformed Features (Colored by True Class)", fontsize=16)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Digit", markerscale=3)
    plt.savefig("svm_pca_tsne_features.png")
    plt.show()

except Exception as e:
    print(f"Could not generate t-SNE plot: {e}")

# ---------------------- Plot 5: Multi-Class ROC Curve ----------------------
print("...Generating Plot 5: Multi-Class ROC Curve (One-vs-Rest)")
try:
    # Binarize the true labels
    y_test_bin = label_binarize(y_test, classes=range(10))
    n_classes = y_test_bin.shape[1]
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure(figsize=(12, 9))
    
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average (AUC = {roc_auc["micro"]:0.3f})',
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average (AUC = {roc_auc["macro"]:0.3f})',
             color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 
                    'purple', 'brown', 'pink', 'gray', 'olive'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Class {i} (AUC = {roc_auc[i]:0.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Multi-Class ROC Curve (One-vs-Rest) - PCA+SVM', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("svm_pca_roc_curve.png")
    plt.show()

except Exception as e:
    print(f"Could not generate ROC plot: {e}")

# ---------------------- Plot 6: Misclassified Samples ----------------------
misclassified_indices = np.where(y_pred != y_test)[0]
print(f"\nFound {len(misclassified_indices)} misclassified samples.")

if len(misclassified_indices) > 0:
    print(f"...Generating Plot 6: {min(15, len(misclassified_indices))} Misclassified Samples")
    
    # Select 15 random misclassified samples
    if len(misclassified_indices) > 15:
        plot_indices = np.random.choice(misclassified_indices, 15, replace=False)
    else:
        plot_indices = misclassified_indices
        
    plt.figure(figsize=(15, 9))
    plt.suptitle("Misclassified Samples (True vs. Predicted)", fontsize=16, y=1.02)
    
    for i, idx in enumerate(plot_indices):
        plt.subplot(3, 5, i + 1)
        # Use the raw 28x28 test image
        plt.imshow(x_test_raw[idx], cmap='gray')
        plt.title(f"True: {y_test[idx]}\nPred: {y_pred[idx]}", color='red')
        plt.axis("off")
        
    plt.tight_layout()
    plt.savefig("svm_pca_misclassified_samples.png")
    plt.show()

# ---------------------- Save Final JSON Report ----------------------
# ---------------------- Save Final JSON Report ----------------------
print("\n💾 Saving detailed evaluation report to JSON...")

# Helper function to convert numpy types to native python types
def clean_json_types(data):
    """Recursively converts numpy types to native Python types for JSON."""
    if isinstance(data, dict):
        return {k: clean_json_types(v) for k, v in data.items()}
    if isinstance(data, list):
        return [clean_json_types(item) for item in data]
    if isinstance(data, (np.float32, np.float64)):
        return float(data)
    if isinstance(data, (np.int32, np.int64)):
        return int(data)
    return data

report_data = {
    "model_type": "PCA+SVM",
    "model_file": MODEL_FILE,
    "dataset": "MNIST",
    "pca_components": pca.n_components_,
    "total_variance_captured": cumulative_variance[-1],
    "test_accuracy": float(acc),
    "total_test_samples": len(y_test),
    "misclassified_samples": len(misclassified_indices),
    "classification_report": class_report_dict,
    "confusion_matrix": cm.tolist(),
    "roc_auc_scores": {k: v for k, v in roc_auc.items()} if 'roc_auc' in locals() else "Not computed",
    "misclassified_indices_sample": misclassified_indices[:20].tolist() # Save first 20
}

# Clean the entire dictionary of any numpy types before saving
report_data_cleaned = clean_json_types(report_data)

# Save the cleaned data to JSON
with open("svm_pca_advanced_evaluation_report.json", "w") as f:
    json.dump(report_data_cleaned, f, indent=2)

print("✅ Advanced report saved to 'svm_pca_advanced_evaluation_report.json'")
print("\n🎉 Evaluation complete.")