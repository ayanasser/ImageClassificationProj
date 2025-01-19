import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.preprocessing import label_binarize
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
import mlflow
import mlflow.keras

def plot_confusion_matrix(y_test, predictions, labels, model_name, output_dir):
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f'Confusion Matrix - {model_name}')
    file_path = f"{output_dir}/{model_name}_confusion_matrix.png"
    plt.savefig(file_path)
    plt.close()
    mlflow.log_artifact(file_path)

def plot_roc_curve(y_test, y_prob, labels, model_name, output_dir):
    n_classes = len(labels)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{labels[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    file_path = f"{output_dir}/{model_name}_roc_curve.png"
    plt.savefig(file_path)
    plt.close()
    mlflow.log_artifact(file_path)

def plot_training_history(history, output_dir):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    file_path = f"{output_dir}/training_history.png"
    plt.savefig(file_path)
    plt.close()
    mlflow.log_artifact(file_path)

def log_classification_report(y_test, predictions, labels):
    class_report = classification_report(y_test, predictions, target_names=labels, output_dict=True)
    for label, metrics in class_report.items():
        if isinstance(metrics, dict):
            mlflow.log_metric(f"precision_{label}", metrics.get("precision", 0))
            mlflow.log_metric(f"recall_{label}", metrics.get("recall", 0))
            mlflow.log_metric(f"f1_score_{label}", metrics.get("f1-score", 0))
    
    report_path = "artifacts/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(y_test, predictions, target_names=labels))
    mlflow.log_artifact(report_path)

def build_mobilenet_model(img_size, num_classes):
    base_model = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights='imagenet'
    )
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
        
    return model, base_model

def train_mobilenet_with_mlflow(X_train, y_train, X_test, y_test, labels, 
                               img_size=(64, 64), batch_size=16, epochs=50, 
                               fine_tune_epochs=20, output_dir='results'):
    
    mlflow.end_run()
    
    with mlflow.start_run(run_name="MobileNetV2"):
        # Log parameters
        mlflow.log_params({
            "img_size": img_size,
            "batch_size": batch_size,
            "initial_epochs": epochs,
            "fine_tune_epochs": fine_tune_epochs
        })
        
        # Build model
        model, base_model = build_mobilenet_model(img_size, len(labels))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Initial training
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            verbose=1
        )
        
        # Fine-tuning
        for layer in base_model.layers:
            layer.trainable = True
            
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        fine_tune_history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=fine_tune_epochs,
            validation_split=0.2,
            verbose=1
        )
        
        # Combine histories
        total_history = history.history.copy()
        for key in total_history.keys():
            total_history[key].extend(fine_tune_history.history[key])
        
        # Plot and log training history
        plot_training_history(history, output_dir)
        
        # Generate and log predictions
        predictions = model.predict(X_test)
        pred_classes = np.argmax(predictions, axis=1)
        
        # Log metrics
        accuracy = accuracy_score(y_test, pred_classes)
        f1 = f1_score(y_test, pred_classes, average='weighted')
        precision = precision_score(y_test, pred_classes, average='weighted')
        recall = recall_score(y_test, pred_classes, average='weighted')
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Log detailed classification report
        log_classification_report(y_test, pred_classes, labels)
        
        # Generate and log plots
        plot_confusion_matrix(y_test, pred_classes, labels=labels, 
                            model_name="MobileNetV2", output_dir=output_dir)
        plot_roc_curve(y_test, predictions, labels=labels, 
                      model_name="MobileNetV2", output_dir=output_dir)
        
        # Log model
        mlflow.keras.log_model(
            model,
            "model",
            registered_model_name="Traffic_Light_Classifier_MobileNetV2"
        )
        
        return model, total_history

def main():
    # Set up MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Traffic Light Classifier - MobileNetV2")
    
    # Add tags
    mlflow.set_tag("project", "traffic_light_classifier")
    mlflow.set_tag("model_type", "MobileNetV2")
    mlflow.set_tag("environment", "development")
    
    # Create directories
    os.makedirs('artifacts', exist_ok=True)
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    

    # Load data
    feature_extractor = "rgb"
    X_train = np.load(f"X_train_{feature_extractor}.npy")
    X_test = np.load(f"X_test_{feature_extractor}.npy")
    y_train = np.load(f"y_train_{feature_extractor}.npy")
    y_test = np.load(f"y_test_{feature_extractor}.npy")
    
    # Define labels
    labels = ['Green', 'Red', 'Yellow', 'off']
    
    # Train model
    model, history = train_mobilenet_with_mlflow(
        X_train, y_train, X_test, y_test,
        labels=labels,
        img_size=(64, 64),
        batch_size=16,
        epochs=50,
        fine_tune_epochs=20,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()