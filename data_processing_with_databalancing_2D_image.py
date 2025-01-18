import os
import cv2
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

import logging
logging.basicConfig(level=logging.INFO)

# Load YAML File for Labels
def load_labels(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

# Extract Image Paths and Labels
def parse_annotations(data, base_path):
    selected_labels = {'Red', 'Green', 'Yellow'}
    images = []
    labels = []

    for item in data:
        img_path = os.path.join(base_path, item['path'])
        for box in item['boxes']:
            label = box['label']
            if label in selected_labels:
                x_min = int(box['x_min'])
                y_min = int(box['y_min'])
                x_max = int(box['x_max'])
                y_max = int(box['y_max'])
                images.append((img_path, x_min, y_min, x_max, y_max))
                labels.append(label)
    return images, labels

def preprocess_image(img_path, bbox, size=(64, 64)):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at path: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    x_min, y_min, x_max, y_max = bbox
    cropped_img = img[y_min:y_max, x_min:x_max]
    resized_img = cv2.resize(cropped_img, size)
    normalized_img = resized_img / 255.0
    return normalized_img

def create_dataset(image_paths, labels, size=(64, 64)):
    X = []
    y = []
    for idx, (img_path, x_min, y_min, x_max, y_max) in enumerate(image_paths):
        try:
            img = preprocess_image(img_path, (x_min, y_min, x_max, y_max), size)
            X.append(img)
            y.append(labels[idx])
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")
    return np.array(X), np.array(y)

# Method 1: Oversampling for Yellow Class
def balance_dataset(X, y, yellow_label):
    data = list(zip(X, y))
    data_yellow = [d for d in data if d[1] == yellow_label]
    data_other = [d for d in data if d[1] != yellow_label]


    # Handle case where Yellow class is empty
    if len(data_yellow) == 0:
        logging.warning("No samples found for 'Yellow' class. Skipping oversampling.")
        return X, y
    
    data_yellow_upsampled = resample(data_yellow,
                                     replace=True,
                                     n_samples=len(data_other),
                                     random_state=42)

    balanced_data = data_yellow_upsampled + data_other
    np.random.shuffle(balanced_data)
    X_balanced, y_balanced = zip(*balanced_data)
    return np.array(X_balanced), np.array(y_balanced)


# Method 2: Data Augmentation for Yellow Class
def augment_images(X, y):
    augmented_images = []
    augmented_labels = []
    for img, label in zip(X, y):
        if label == 'Yellow':  # Only augment Yellow class
            for angle in [-15, 0, 15]:
                M = cv2.getRotationMatrix2D((32, 32), angle, 1)
                rotated = cv2.warpAffine(img.reshape(64, 64, 3), M, (64, 64))
                augmented_images.append(rotated)
                augmented_labels.append(label)
        else:
            augmented_images.append(img)
            augmented_labels.append(label)
    return np.array(augmented_images), np.array(augmented_labels)


def log_class_distribution(labels, label_encoder, stage):
    """
    Logs the distribution of classes at different stages.
    Args:
        labels (np.array): Array of labels.
        label_encoder (LabelEncoder): Fitted label encoder.
        stage (str): Stage of processing (e.g., 'Before Augmentation', 'After Augmentation').
    """
    unique, counts = np.unique(labels, return_counts=True)
    label_names = label_encoder.inverse_transform(unique)
    distribution = dict(zip(label_names, counts))
    logging.info(f"Class distribution {stage}: {distribution}")

# Main Function
def main():
    yaml_path = 'train_rgb/train.yaml'
    base_path = 'train_rgb'
    method = 'oversampling'  # Choose between 'oversampling' and 'augmentation'

    logging.info(f'Loading annotations from {yaml_path}')
    data = load_labels(yaml_path)
    image_paths, labels = parse_annotations(data, base_path)

    logging.info('Preprocessing dataset...')
    X, y = create_dataset(image_paths, labels)

    logging.info('Encoding labels...')
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)


    #   Identify the label index for 'Yellow'
    yellow_label = encoder.transform(['Yellow'])[0]

    print(f"Yellow label index: {yellow_label}")

    # Log distribution before augmentation
    log_class_distribution(y_encoded, encoder, "Before Augmentation")

    # Apply imbalance handling based on the method selected
    if method == 'oversampling':
        logging.info('Applying oversampling...')
        X, y_encoded = balance_dataset(X, y_encoded, yellow_label)
    elif method == 'augmentation':
        logging.info('Applying augmentation...')
        X, y_encoded = augment_images(X, y_encoded)


    # Log distribution after augmentation
    log_class_distribution(y_encoded, encoder, "After Augmentation")
    
    logging.info('Splitting dataset...')
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    logging.info(f'Train data shape: {X_train.shape}, Test data shape: {X_test.shape}')
    logging.info(f'Classes: {encoder.classes_}')

    np.save(f'X_train_{method}.npy', X_train)
    np.save(f'X_test_{method}.npy', X_test)
    np.save(f'y_train_{method}.npy', y_train)
    np.save(f'y_test_{method}.npy', y_test)

if __name__ == '__main__':
    main()
