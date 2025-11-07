# Model Evaluation Comparison: CNN+SVM vs. SVM on MNIST

## 1. Overview

This document provides a comprehensive comparison of two machine learning models evaluated on the MNIST dataset:

1.  **CNN+SVM**: A hybrid model that uses a Convolutional Neural Network (CNN) for automated, deep feature extraction, followed by a Support Vector Machine (SVM) for classification.
2.  **PCA+SVM**: A classical model that uses Principal Component Analysis (PCA) for linear dimensionality reduction on the raw pixel data, followed by an SVM for classification.

Based on the evaluation data, the **CNN+SVM model demonstrates significantly superior performance** in nearly every metric, highlighting the power of deep learning for feature extraction in image classification tasks.

## 2. Executive Summary: Key Differences

| Metric | CNN+SVM (Deep Features) | PCA+SVM (Classical Features) | Key Takeaway |
| :--- | :---: | :---: | :--- |
| **Test Accuracy** | **99.25%** | 97.86% | CNN+SVM is ~1.4% more accurate. |
| **Misclassified Samples** | **75 / 10,000** | 214 / 10,000 | CNN+SVM makes **2.85x fewer errors**. |
| **Macro Avg F1-Score**| **0.9924** | 0.9785 | CNN+SVM has a better balance of precision and recall. |
| **Macro Avg AUC** | 0.99972 | 0.99970 | Both models are near-perfect at ranking predictions. |
| **Feature Quality** | **Extremely high** | Moderate-High | This is the single biggest differentiator. |

---

## 3. Analysis 1: Feature-Space Visualization (t-SNE)

The most striking difference between the two models is the **quality of the features** they use for classification. The t-SNE plots visualize the high-dimensional feature space in 2D.

### CNN+SVM Features

The features extracted by the CNN produce **extremely tight, dense, and well-separated clusters** for each digit. There is almost no overlap between classes. This makes the SVM's job trivial, as a simple linear boundary can easily separate the digits.

### PCA+SVM Features

The features from PCA are **significantly more diffuse and overlapping**. Clusters for different digits (e.g., the grey, light blue, purple, and pink dots) are spread out and mixed, making it much harder for the SVM to find a clean hyperplane to separate them. This visual overlap directly explains the higher error rate (214 misclassifications) of the PCA+SVM model.

---

## 4. Analysis 2: Overall Performance Metrics

The CNN+SVM model is the clear winner in all key classification metrics.

### CNN+SVM
* **Accuracy:** **99.25%**
* **Total Errors:** 75
* **Classification Report:** All classes show precision, recall, and F1-scores **above 0.988**. The performance is exceptionally high and consistent across all 10 digits.

### PCA+SVM
* **Accuracy:** 97.86%
* **Total Errors:** 214
* **Classification Report:** While still good, the scores are consistently lower than the CNN+SVM model. Several classes have F1-scores around 0.97, indicating more confusion.

The **Multi-Class ROC Curves** for both models show near-perfect AUC (Area Under the Curve) scores (all 0.999 or 1.000). This indicates that both models are excellent at ranking the correct class as the most probable. The difference in accuracy comes from the *decision boundary*—the CNN's superior features allow for a more accurate final classification.

---

## 5. Analysis 3: Error Analysis (Confusion Matrix)

By comparing the confusion matrices, we can see *where* each model fails.

### CNN+SVM Confusion Matrix

* The diagonal is extremely strong; most off-diagonal errors are just 1 or 2.
* The largest single error source is 6 instances of a true "5" being predicted as a "3".
* Other notable errors are 4 instances of "2" -> "7" and 3 instances of "7" -> "9". The errors are minimal.

### PCA+SVM Confusion Matrix (from `svm_pca_advanced_evaluation_report.json`)
* The model makes larger, more frequent errors.
* The largest error source is **16 instances of a true "7" being predicted as a "2"**. This is 4x worse than the CNN+SVM's worst error.
* Other major errors include 11 instances of "4" -> "9", 10 instances of "9" -> "4", and 9 instances of "5" -> "3".
* The errors are more numerous and spread across more class pairs, confirming the cluster overlap seen in the t-SNE plot.

---

## 6. Analysis 4: Feature Extraction Efficiency (PCA Variance)

The PCA variance plots reveal a fundamental difference in the *nature* of the features.

### CNN+SVM Feature Analysis

This plot shows a PCA analysis performed on the *CNN's output features*.
* **High Concentration:** The variance is highly concentrated in the first few components. The very first component alone accounts for ~22% of the variance.
* **Efficiency:** 95% of the variance from the CNN's features is captured in only **~30 components**. This shows the CNN has learned a very efficient and compact representation of the data.

### PCA+SVM Feature Analysis

This plot shows the PCA performed on the *raw image pixels*.
* **Low Concentration:** The variance is spread out. The first component only accounts for ~10%.
* **Information Loss:** The model used **50 components**, which only captured **82.7%** of the total variance. This means the SVM is forced to make classifications based on an incomplete representation of the data, which contributes to its lower accuracy.

## 7. Conclusion

The evaluation data provides a clear and decisive result:

The **CNN+SVM** model is vastly superior. Its strength comes from the **CNN's ability to learn complex, non-linear, and highly discriminative features** from the raw image data. These features are so well-structured (as seen in the t-SNE plot) that the subsequent SVM classifier can perform its job with near-perfect accuracy (99.25%).

The **PCA+SVM** model is limited by its linear feature extractor (PCA). PCA can only capture linear variance and, in this case, failed to capture over 17% of the data's variance with 50 components. This resulted in less discriminative features (seen as overlapping clusters) and ultimately led to **2.85 times more classification errors** than the CNN-based approach.
