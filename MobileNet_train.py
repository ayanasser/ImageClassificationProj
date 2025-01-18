from traffic_light_classifier import *

# Load your data
X_train = np.load('X_train_rgb.npy')
X_test = np.load('X_test_rgb.npy')
y_train = np.load('y_train_rgb.npy')
y_test = np.load('y_test_rgb.npy')

# Initialize the classifier
classifier = TrafficLightClassifier(
    img_size=(64, 64),
    batch_size=16
)

# Load and preprocess data
num_classes = classifier.load_and_preprocess_data(X_train, X_test, y_train, y_test)

# Build the model
model = classifier.build_model(num_classes)

# Train the model
history = classifier.train(epochs=50, fine_tune_epochs=20)

# Plot training history
classifier.plot_training_history()

# Evaluate the model
test_loss, test_accuracy = classifier.evaluate()
print(f"Test accuracy: {test_accuracy:.4f}")