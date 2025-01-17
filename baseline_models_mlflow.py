import numpy as np
import os
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn

def plot_confusion_matrix(y_test, predictions, labels, model_name, output_dir):
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f'Confusion Matrix - {model_name}')
    file_path = f"{output_dir}/{model_name}_confusion_matrix.png"
    plt.savefig(file_path)
    plt.close()
    mlflow.log_artifact(file_path)

def plot_roc_curve(y_test, y_prob, model_name, output_dir):
    n_classes = len(np.unique(y_test))
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    file_path = f"{output_dir}/{model_name}_roc_curve.png"
    plt.savefig(file_path)
    plt.close()
    mlflow.log_artifact(file_path)

def log_classification_report(y_test, predictions, labels):
    # Generate classification report
    class_report = classification_report(y_test, predictions, target_names=labels, output_dict=True)
    # Log metrics for each label in MLflow
    for label, metrics in class_report.items():
        if isinstance(metrics, dict):
            mlflow.log_metric(f"precision_{label}", metrics.get("precision", 0))
            mlflow.log_metric(f"recall_{label}", metrics.get("recall", 0))
            mlflow.log_metric(f"f1_score_{label}", metrics.get("f1-score", 0))
    # Save classification report as a text artifact
    report_path = "artifacts/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(y_test, predictions, target_names=labels))
    mlflow.log_artifact(report_path)

def train_model_with_mlflow(model, model_name, X_train, y_train, X_test, y_test, output_dir, labels, training_technique):

    # Ensure any previous run is ended
    mlflow.end_run()  
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    with mlflow.start_run(run_name=model_name):

        mlflow.sklearn.autolog()

        # Log the training technique as a parameter
        mlflow.log_param("training_technique", training_technique)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # # Log custom metrics
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Log detailed classification report
        log_classification_report(y_test, predictions, labels)

        # Log parameters
        if hasattr(model, 'get_params'):
            mlflow.log_params(model.get_params())

        # Log artifacts
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            plot_roc_curve(y_test, y_prob, model_name, output_dir)

        plot_confusion_matrix(y_test, predictions, labels=labels, model_name=model_name, output_dir=output_dir)

        #  Define input example and signature
        input_example = X_test[:1]  # One sample for input example
        signature = infer_signature(X_test, predictions)

        # Register model version with input example and signature
        mlflow.sklearn.log_model(
            model,
            artifact_path=model_name,
            registered_model_name="Traffic_Light_Classifier",
            input_example=input_example,
            signature=signature,
        )

        print(f"{model_name} Classifier:")
        print("Accuracy:", accuracy)
        print(classification_report(y_test, predictions, target_names=labels))

def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Traffic Light Classifier")

    # Add tags
    mlflow.set_tag("project", "traffic_light_classifier")
    mlflow.set_tag("author", "Aya")
    mlflow.set_tag("environment", "development")

    feature_extractor = "combined"  # Specify feature extractor here
    x_train_numpy = f"X_train_{feature_extractor}.npy"
    x_test_numpy = f"X_test_{feature_extractor}.npy"
    y_train_numpy = f"y_train_{feature_extractor}.npy"
    y_test_numpy = f"y_test_{feature_extractor}.npy"



    # Load dataset
    X_train = np.load(x_train_numpy)
    X_test = np.load( x_test_numpy)
    y_train = np.load(y_train_numpy)
    y_test = np.load(y_test_numpy)

    labels = ['Green', 'Red','Yellow','off']

    # Log dataset artifacts
    os.makedirs('artifacts', exist_ok=True)
    np.save(f"artifacts/{x_train_numpy}", X_train)
    np.save(f"artifacts/{x_test_numpy}", X_test)
    np.save(f"artifacts/{y_train_numpy}", y_train)
    np.save(f"artifacts/{y_test_numpy}", y_test)

    mlflow.log_artifact(f"artifacts/{x_train_numpy}")
    mlflow.log_artifact(f"artifacts/{x_test_numpy}")
    mlflow.log_artifact(f"artifacts/{y_train_numpy}")
    mlflow.log_artifact(f"artifacts/{y_test_numpy}")

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    # Train models 
    training_technique = "Combined features"  

    svm_model = SVC(kernel='rbf', probability=True, random_state=42)
    train_model_with_mlflow(svm_model, f"SVM {training_technique}", X_train, y_train, X_test, y_test, output_dir, labels, training_technique)

    logreg_model = LogisticRegression(random_state=42, max_iter=500)
    train_model_with_mlflow(logreg_model, f"Logistic Regression {training_technique}", X_train, y_train, X_test, y_test, output_dir, labels, training_technique)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    train_model_with_mlflow(knn_model, f"KNN {training_technique}", X_train, y_train, X_test, y_test, output_dir, labels, training_technique)


    random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
    train_model_with_mlflow(random_forest_model, f"Random Forest {training_technique}", X_train, y_train, X_test, y_test, output_dir, labels, training_technique)



if __name__ == "__main__":
    main()
