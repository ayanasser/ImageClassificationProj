import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE
from collections import Counter


# Plot Confusion Matrix
def plot_confusion_matrix(y_test, predictions, labels, model_name, output_dir):
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f"{output_dir}/{model_name}_confusion_matrix.png")
    plt.close()


# Plot ROC Curve
def plot_roc_curve(y_test, y_prob, n_classes, model_name, output_dir):
    # Binarize the labels for ROC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    plt.figure(figsize=(8, 6))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Random guessing line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f"{output_dir}/{model_name}_roc_curve.png")
    plt.close()


# Train SVM Model
def train_svm(X_train, y_train, X_test, y_test, output_dir):
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Evaluation Metrics
    print("SVM Classifier:")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    # Generate Plots
    plot_confusion_matrix(y_test, predictions, labels=['Red', 'Green', 'Yellow'], model_name="SVM", output_dir=output_dir)
    plot_roc_curve(y_test, y_prob, n_classes=3, model_name="SVM", output_dir=output_dir)

    return model


# Train Logistic Regression Model
def train_logistic_regression(X_train, y_train, X_test, y_test, output_dir):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Evaluation Metrics
    print("Logistic Regression Classifier:")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    # Generate Plots
    plot_confusion_matrix(y_test, predictions, labels=['Red', 'Green', 'Yellow'], model_name="LogReg", output_dir=output_dir)
    plot_roc_curve(y_test, y_prob, n_classes=3, model_name="LogReg", output_dir=output_dir)

    return model


# Main Function
def main():
    # Load Preprocessed Data
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')

    # Output directory for visualizations
    output_dir = 'results_with_smote'
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Before SMOTE - Print class distribution
    print("Original Class Distribution:", Counter(y_train))


    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    # Apply SMOTE for oversampling minority classes
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # After SMOTE - Print class distribution
    print("Resampled Class Distribution:", Counter(y_train_resampled))

    # Train Models
    train_svm(X_train_resampled, y_train_resampled, X_test, y_test, output_dir)
    train_logistic_regression(X_train_resampled, y_train_resampled, X_test, y_test, output_dir)


if __name__ == "__main__":
    main()
