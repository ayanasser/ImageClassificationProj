import numpy as np
import os
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize

"""
input dataset X passed to the SVM classifier has 4 dimensions,
 but SVM in Scikit-learn expects the input to be 2-dimensional ([n_samples, n_features]).

 Image data in the shape (n_samples, height, width, channels) instead of flattened features with shape (n_samples, n_features).

    The solution is to flatten the data before passing it to the SVM classifier.


"""


# Plot Confusion Matrix
def plot_confusion_matrix(y_test, predictions, labels, model_name, output_dir):
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f"{output_dir}/{model_name}_confusion_matrix.png")
    plt.close()


# Plot ROC Curve
def plot_roc_curve(y_test, y_prob, model_name, output_dir):
    # Determine the number of classes
    n_classes = len(np.unique(y_test))
    
    # Binarize the labels for ROC
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
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


## SVM in sklearn needs the data to be 
def train_svm(X_train, y_train, X_test, y_test, output_dir):
    # Flatten data if necessary
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    # Train SVM
    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    y_prob = model.predict_proba(X_test)


    print("SVM Classifier:")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))
    unique_labels = np.unique(y_test)
    plot_confusion_matrix(y_test, predictions, labels=unique_labels, model_name="SVM", output_dir=output_dir)

    # Generate Plots
    plot_roc_curve(y_test, y_prob, model_name="SVM", output_dir=output_dir)
    # plot_roc_curve(y_test, y_prob, n_classes=3, model_name="SVM", output_dir=output_dir)

    return model

def train_logistic_regression(X_train, y_train, X_test, y_test, output_dir):
    # Flatten data if necessary
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    # Train Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    y_prob = model.predict_proba(X_test)


    print("Logistic Regression Classifier:")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    # Generate Plots
    unique_labels = np.unique(y_test)  # Get unique labels from test data
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f"unique_labels are {unique_labels}")

    plot_roc_curve(y_test, y_prob, model_name="LogReg", output_dir=output_dir)
    # plot_roc_curve(y_test, y_prob, n_classes=3, model_name="LogReg", output_dir=output_dir)

def main():

    """
    Main function to load preprocessed data, train SVM and Logistic Regression models,
    and generate evaluation plots.

    Expected input files:
    - X_train_augmentation.npy: Training data features
    - X_test_augmentation.npy: Test data features
    - y_train_augmentation.npy: Training data labels
    - y_test_augmentation.npy: Test data labels

    Output:
    - Saves confusion matrix and ROC curve plots in the 'results_augmentation' directory.
    """
    # Load preprocessed data
    X_train = np.load('X_train_augmentation.npy')
    X_test = np.load('X_test_augmentation.npy')
    y_train = np.load('y_train_augmentation.npy')
    y_test = np.load('y_test_augmentation.npy')


    # Output directory for visualizations
    output_dir = 'results_augmentation'
    os.makedirs(output_dir, exist_ok=True)
    # Train and evaluate models
    train_svm(X_train, y_train, X_test, y_test, output_dir)
    train_logistic_regression(X_train, y_train, X_test, y_test,  output_dir)

if __name__ == "__main__":
    main()
