# cnn_svm_advanced_evaluation.py
# Advanced standalone evaluation of CNN+SVM MNIST model

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
# For t-SNE: 10,000 samples is slow. Let's use a subset.
TSNE_SAMPLES = 2500

# ---------------------- Load Model ----------------------
print("🔄 Loading CNN+SVM model...")
# We assume the model was saved as a tuple: (feature_extractor, pca, svm)
feature_extractor, pca, svm = joblib.load("cnn_svm_model.pkl")

# ---------------------- Load MNIST Test Data ----------------------
print("📥 Loading MNIST test data...")
(_, _), (x_test, y_test) = mnist.load_data()
x_test_normalized = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# ---------------------- Feature Extraction & PCA ----------------------
print("⚙️ Extracting CNN features and applying PCA...")
X_test_feats = feature_extractor.predict(x_test_normalized)
X_test_pca = pca.transform(X_test_feats)

# ---------------------- Prediction & Scores ----------------------
print("🧠 Predicting with SVM...")
y_pred = svm.predict(X_test_pca)
# Get decision function scores for ROC curve (works even if probability=False)
try:
    y_score = svm.decision_function(X_test_pca)
except Exception as e:
    print(f"Could not get decision_function: {e}")
    y_score = None # ROC plot will be skipped

# ---------------------- Basic Metrics ----------------------
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ CNN+SVM Test Accuracy: {acc:.4f}")

print("\n📈 Classification Report (Text):")
class_report_text = classification_report(y_test, y_pred)
print(class_report_text)
class_report_dict = classification_report(y_test, y_pred, output_dict=True)

# ---------------------- Confusion Matrix Plot ----------------------
print("\n...Generating Plot 1: Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Confusion Matrix - CNN+SVM", fontsize=14)
plt.savefig("cnn_svm_confusion_matrix.png")
plt.show()

# ---------------------- NEW PLOT 1: Per-Class Metrics ----------------------
print("...Generating Plot 2: Per-Class Metrics Bar Chart")
try:
    # Convert report to DataFrame, excluding avg/support rows
    df_report = pd.DataFrame(class_report_dict).iloc[:-1, :].T
    df_report = df_report.drop(['accuracy', 'macro avg', 'weighted avg'])
    
    df_report.plot(kind='bar', figsize=(14, 7), rot=0)
    plt.title("Per-Class Performance (Precision, Recall, F1-Score)", fontsize=16)
    plt.xlabel("Digit Class", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0.9, 1.0) # Zoom in on the high scores
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.savefig("cnn_svm_class_metrics.png")
    plt.show()
except Exception as e:
    print(f"Could not plot per-class metrics: {e}")

# ---------------------- NEW PLOT 2: PCA Explained Variance ----------------------
print("...Generating Plot 3: PCA Explained Variance")
try:
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
    plt.axhline(y=0.95, color='k', linestyle='--', label='95% threshold') # Example threshold
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance")
    plt.grid(True)
    plt.legend()
    
    plt.suptitle("PCA Analysis (50 Components)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("cnn_svm_pca_variance.png")
    plt.show()
except Exception as e:
    print(f"Could not plot PCA variance: {e}")


# ---------------------- NEW PLOT 3: t-SNE Feature Visualization ----------------------
print(f"...Generating Plot 4: t-SNE Feature Visualization (on {TSNE_SAMPLES} samples)")
try:
    # Subsample the data
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
        
    plt.title("t-SNE Visualization of PCA Features (Colored by True Class)", fontsize=16)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Digit", markerscale=3)
    plt.savefig("cnn_svm_tsne_features.png")
    plt.show()

except Exception as e:
    print(f"Could not generate t-SNE plot: {e}")


# ---------------------- NEW PLOT 4: Multi-Class ROC Curve ----------------------
if y_score is not None:
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
        
        # Plot micro and macro
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'Micro-average (AUC = {roc_auc["micro"]:0.3f})',
                 color='deeppink', linestyle=':', linewidth=4)
        
        plt.plot(fpr["macro"], tpr["macro"],
                 label=f'Macro-average (AUC = {roc_auc["macro"]:0.3f})',
                 color='navy', linestyle=':', linewidth=4)
        
        # Plot individual class curves
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
        plt.title('Multi-Class ROC Curve (One-vs-Rest)', fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig("cnn_svm_roc_curve.png")
        plt.show()

    except Exception as e:
        print(f"Could not generate ROC plot: {e}")
else:
    print("\nSkipping ROC Curve plot (could not get decision_function scores).")


# ---------------------- Plot: Misclassified Samples (Expanded) ----------------------
misclassified_indices = np.where(y_pred != y_test)[0]
if len(misclassified_indices) > 0:
    print(f"\n...Generating Plot 6: {min(15, len(misclassified_indices))} Misclassified Samples")
    
    # Select 15 random misclassified samples
    if len(misclassified_indices) > 15:
        plot_indices = np.random.choice(misclassified_indices, 15, replace=False)
    else:
        plot_indices = misclassified_indices
        
    plt.figure(figsize=(15, 9))
    plt.suptitle("Misclassified Samples (True vs. Predicted)", fontsize=16, y=1.02)
    
    for i, idx in enumerate(plot_indices):
        plt.subplot(3, 5, i + 1)
        plt.imshow(x_test[idx], cmap='gray') # Use original x_test for plotting
        plt.title(f"True: {y_test[idx]}\nPred: {y_pred[idx]}", color='red')
        plt.axis("off")
        
    plt.tight_layout()
    plt.savefig("cnn_svm_misclassified_samples.png")
    plt.show()

# ---------------------- Save Report to JSON ----------------------
print("\n💾 Saving detailed evaluation report to JSON...")

# Prepare report data
report_data = {
    "model_type": "CNN+SVM",
    "dataset": "MNIST",
    "test_accuracy": float(acc),
    "total_test_samples": len(y_test),
    "misclassified_samples": len(misclassified_indices),
    "classification_report": class_report_dict,
    "confusion_matrix": cm.tolist(),
    # Add ROC AUC scores if they were calculated
    "roc_auc_scores": {k: v for k, v in roc_auc.items()} if 'roc_auc' in locals() else "Not computed",
    "misclassified_indices": misclassified_indices[:20].tolist() # Save first 20
}

# Save to JSON file
with open("cnn_svm_advanced_evaluation_report.json", "w") as f:
    json.dump(report_data, f, indent=2)

print("✅ Advanced report saved to 'cnn_svm_advanced_evaluation_report.json'")
print("\n🎉 Evaluation complete.")