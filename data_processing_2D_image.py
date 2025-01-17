import os
import cv2
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import logging
logging.basicConfig(level=logging.INFO)


# Load YAML File for Labels
def load_labels(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

# Extract Image Paths and Labels
def parse_annotations(data, base_path):
    # Specify the labels to keep
    selected_labels = {'Red', 'Green', 'Yellow', 'off'}
    images = []
    labels = []

    for item in data:
        img_path = os.path.join(base_path, item['path'])
        for box in item['boxes']:
            label = box['label']
            # Filter only selected labels
            if label in selected_labels:
                x_min = int(box['x_min'])
                y_min = int(box['y_min'])
                x_max = int(box['x_max'])
                y_max = int(box['y_max'])
                images.append((img_path, x_min, y_min, x_max, y_max))
                labels.append(label)
    return images, labels

def preprocess_image(img_path, bbox, size=(64, 64)):
    # Check if path exists
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    # Extract bounding box
    x_min, y_min, x_max, y_max = bbox

    # Validate bounding box
    if x_min < 0 or y_min < 0 or x_max > img.shape[1] or y_max > img.shape[0]:
        raise ValueError(f"Bounding box out of bounds for {img_path}")

    # Crop the region
    cropped_img = img[y_min:y_max, x_min:x_max]
    if cropped_img.size == 0:
        raise ValueError(f"Empty cropped image at {img_path}")

    # Resize and normalize
    resized_img = cv2.resize(cropped_img, size)
    normalized_img = resized_img / 255.0
    return normalized_img



# Generate Dataset
def create_dataset(image_paths, labels, size=(64, 64)):
    X = []
    y = []
    for idx, (img_path, x_min, y_min, x_max, y_max) in enumerate(image_paths):
        try:
            img = preprocess_image(img_path, (x_min, y_min, x_max, y_max), size)
            X.append(img)  # Flatten for ML models //HERE
            y.append(labels[idx])
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")
    return np.array(X), np.array(y)



def save_samples(image_paths, labels, output_dir, num_samples=10, size=(64, 64)):
    """
    Save specified number of cropped samples with original filenames and labels.

    Args:
        image_paths (list): List of tuples containing image paths and bounding boxes.
        labels (list): Corresponding labels for the images.
        output_dir (str): Directory to save the samples.
        num_samples (int): Number of samples to save.
        size (tuple): Target size for resizing images.
    """
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for i, (img_path, x_min, y_min, x_max, y_max) in enumerate(image_paths[:num_samples]):
        logging.info(f"Processing image: {img_path}")

        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Error loading image: {img_path}")
            continue

        # Crop and resize the image
        cropped_img = img[y_min:y_max, x_min:x_max]
        resized_img = cv2.resize(cropped_img, size)

        # Extract the original filename
        original_filename = os.path.basename(img_path)  # e.g., '12345.png'
        name, ext = os.path.splitext(original_filename)

        # Save with label appended
        label = labels[i]
        output_path = os.path.join(output_dir, f"{name}_{label}{ext}")
        cv2.imwrite(output_path, resized_img * 255)  # Convert back to 0–255 scale for saving
        logging.info(f"Saved sample: {output_path}")


# Function to calculate dataset statistics
def get_dataset_statistics(labels):
    label_counts = Counter(labels)
    logging.info("Dataset Statistics:")
    for label, count in label_counts.items():
        logging.info(f"{label}: {count} samples")


# Main Function
def main():
    yaml_path = 'train_rgb/train.yaml'  # Path to YAML file
    base_path = 'train_rgb'  # Path to base image folder

    # Load annotations
    logging.info(f'Loading annotations from {yaml_path}')
    data = load_labels(yaml_path)
    logging.info(f'Found {len(data)} annotations')
    image_paths, labels = parse_annotations(data, base_path)

    # Preprocess Dataset
    logging.info('Preprocessing dataset...')
    X, y = create_dataset(image_paths, labels)

    # Encode Labels
    logging.info('Encoding labels...')
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    logging.info('Splitting dataset...')
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    logging.info(f'Train data shape: {X_train.shape}, Test data shape: {X_test.shape}')
    logging.info(f'Classes: {encoder.classes_}')

    save_samples(image_paths[:10], labels[:10], output_dir='samples/', num_samples=10)


    # get statistics
    get_dataset_statistics(labels)

    # Save Preprocessed Data
    np.save('X_train_rgb.npy', X_train)
    np.save('X_test_rgb.npy', X_test)
    np.save('y_train_rgb.npy', y_train)
    np.save('y_test_rgb.npy', y_test)

if __name__ == '__main__':
    main()