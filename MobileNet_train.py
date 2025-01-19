from traffic_light_classifier import *
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
import numpy as np

def load_data(x_train_path, x_test_path, y_train_path, y_test_path):
    """Loads data from specified .npy files."""
    X_train = np.load(x_train_path)
    X_test = np.load(x_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)
    return X_train, X_test, y_train, y_test


def train_and_evaluate_model(X_train, X_test, y_train, y_test, img_size=(64, 64), batch_size=16, epochs=50, fine_tune_epochs=20):
    """Trains and evaluates the MobileNetV2-based classifier."""
    classifier = TrafficLightClassifier(img_size=img_size, batch_size=batch_size)
    num_classes = classifier.load_and_preprocess_data(X_train, X_test, y_train, y_test)
    model = classifier.build_model(num_classes)
    history = classifier.train(epochs=epochs, fine_tune_epochs=fine_tune_epochs)
    classifier.plot_training_history()

    predictions = classifier.predict(X_test)
    predictions = np.argmax(predictions, axis=1)

    test_loss, test_accuracy = classifier.evaluate()
    print(f"Test Accuracy: {test_accuracy}")

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    conf_matrix = confusion_matrix(y_test, predictions)
    ConfusionMatrixDisplay(conf_matrix).plot()


def MobileNet_train():
    """Main function to load data and train the model."""
    # Data paths
    x_train_path = 'X_train_rgb.npy'
    x_test_path = 'X_test_rgb.npy'
    y_train_path = 'y_train_rgb.npy'
    y_test_path = 'y_test_rgb.npy'

    # Load data
    X_train, X_test, y_train, y_test = load_data(x_train_path, x_test_path, y_train_path, y_test_path)

    # Train and evaluate the model
    train_and_evaluate_model(X_train, X_test, y_train, y_test)

MobileNet_train()