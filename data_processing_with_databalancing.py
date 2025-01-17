import os
import cv2
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import logging
from sklearn.utils import resample
import matplotlib.pyplot as plt
import random
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
    # normalized_img = resized_img / 255.0
    return resized_img



# Generate Dataset
def create_dataset(image_paths, labels, size=(64, 64)):
    X = []
    y = []
    for idx, (img_path, x_min, y_min, x_max, y_max) in enumerate(image_paths):
        try:
            img = preprocess_image(img_path, (x_min, y_min, x_max, y_max), size)
            # X.append(img.flatten())  # Flatten for ML models
            y.append(labels[idx])
            X.append(img)
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")
    return np.array(X), np.array(y)


def advanced_augment_image(image, augmentations):
    augmented_images = []
    
    # Ensure the image has 3 channels
    if len(image.shape) == 2 or image.shape[2] == 1:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for aug in augmentations:
        if aug == "flip":
            augmented_images.append(cv2.flip(image, 1))  # Horizontal flip
        elif aug == "brightness":
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 50)  # Increase brightness
            augmented_images.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        elif aug == "rotate":
            rows, cols = image.shape[:2]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)  # Rotate 15 degrees
            augmented_images.append(cv2.warpAffine(image, M, (cols, rows)))
    return augmented_images




def upsample_data(images, labels, target_class, augmentation_methods, target_count):
    """
    Upsample a given class to reach the target count by combining original and augmented images.
    Args:
        images (list): List of images.
        labels (list): List of corresponding labels.
        target_class (str): The class to upsample.
        augmentation_methods (list): List of augmentation techniques to apply.
        target_count (int): Target number of samples for the class.
    Returns:
        tuple: (list of upsampled images, list of upsampled labels)
    """
    # Select images belonging to the target class
    class_indices = [i for i, label in enumerate(labels) if label == target_class]
    class_images = [images[i] for i in class_indices]

    # Calculate the number of augmented images needed
    num_to_augment = target_count - len(class_images)
    if num_to_augment <= 0:
        return class_images[:target_count], [target_class] * target_count

    # Generate augmented images
    augmented_images = []
    while len(augmented_images) < num_to_augment:
        img = random.choice(class_images)
        aug_images = advanced_augment_image(img, augmentation_methods)
        augmented_images.extend(aug_images)

    # Limit augmented images to the exact number needed
    augmented_images = augmented_images[:num_to_augment]

    # Combine original and augmented images
    upsampled_images = class_images + augmented_images
    upsampled_labels = [target_class] * len(upsampled_images)

    return upsampled_images, upsampled_labels



# Downsampling
def downsample_data(images, labels, target_class, target_count):
    class_indices = [i for i, label in enumerate(labels) if label == target_class]
    downsampled_indices = resample(class_indices, n_samples=target_count, random_state=42, replace=False)
    return [images[i] for i in downsampled_indices], [labels[i] for i in downsampled_indices]

def normalize_images(images):
    return [img / 255.0 for img in images]

# Balancing Dataset
def balance_dataset(images, labels, target_count):
    augmented_images, augmented_labels = [], []

    # Upsample "Yellow" and "Off"
    for target_class in ["Yellow", "off"]:
        up_images, up_labels = upsample_data(images, labels, target_class, ["flip", "brightness", "rotate"], target_count)
        augmented_images.extend(up_images)
        augmented_labels.extend(up_labels)

    # Downsample "Green" and "Red"
    for target_class in ["Green", "Red"]:
        down_images, down_labels = downsample_data(images, labels, target_class, target_count)
        augmented_images.extend(down_images)
        augmented_labels.extend(down_labels)

    return augmented_images, augmented_labels


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
    yaml_path = 'train.yaml'  # Path to YAML file
    base_path = ''  # Path to base image folder

    # Load annotations
    logging.info(f'Loading annotations from {yaml_path}')
    data = load_labels(yaml_path)
    logging.info(f'Found {len(data)} annotations')
    image_paths, labels = parse_annotations(data, base_path)

    # Preprocess Dataset
    logging.info('Preprocessing dataset...')
    X, y = create_dataset(image_paths, labels)

    print(f"shape of X: {X.shape}, shape of y: {y.shape}")


    # get statistics
    get_dataset_statistics(labels)

    # exit()
    # Balance Dataset
    print('Balancing dataset...')
    # target_count = 2500  # Desired count for each class
    target_count = int(np.mean([444, 3057, 5207, 726]))

    balanced_images, balanced_labels = balance_dataset(X, y, target_count)
    balanced_images = normalize_images(balanced_images)


    print(f'Balanced dataset size: {len(balanced_images)}')

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(balanced_labels)
    logging.info(f'Classes: {encoder.classes_}')

    class_counts = Counter(balanced_labels)
    print(f"Balanced class counts: {class_counts}")


    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_images, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f'Train size: {len(X_train)}, Test size: {len(X_test)}')



    # Save Preprocessed Data
    np.save(f'X_train_b{target_count}.npy', X_train)
    np.save(f'X_test_b{target_count}.npy', X_test)
    np.save(f'y_train_b{target_count}.npy', y_train)
    np.save(f'y_test_b{target_count}.npy', y_test)




if __name__ == '__main__':
    main()








