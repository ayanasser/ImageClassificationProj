import os
import cv2
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from skimage.feature import local_binary_pattern, hog

# Configuration
CONFIG = {
    "yaml_path": "train.yaml",
    "base_path": "",
    "output_dir": "samples/",
    "image_size": (64, 64),
    "test_size": 0.2,
    "random_state": 42,
    "selected_labels": {"Red", "Green", "Yellow", "off"},
    "num_samples_to_save": 10,
    "feature_extractor": "combined",  # Options: "hog", "color_histogram", "lbp", "combined"
}

# Utility Functions
def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def validate_bbox(bbox, img_shape):
    x_min, y_min, x_max, y_max = bbox
    if x_min < 0 or y_min < 0 or x_max > img_shape[1] or y_max > img_shape[0]:
        raise ValueError("Bounding box out of image bounds.")

def preprocess_image(img_path, bbox, size):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    validate_bbox(bbox, img.shape)
    x_min, y_min, x_max, y_max = bbox
    cropped_img = img[y_min:y_max, x_min:x_max]

    if cropped_img.size == 0:
        raise ValueError(f"Empty cropped image at {img_path}")

    resized_img = cv2.resize(cropped_img, size)
    return resized_img / 255.0


def extract_hog_features(gray):

    features, _ = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True,
        transform_sqrt=True
    )
    return features

# Feature Extraction
def extract_features(image, method):
     # Ensure the image is a valid format (uint8) for OpenCV
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)

    # Validate the image dimensions
    if len(image.shape) == 2:  # Grayscale image
        gray = image
    elif len(image.shape) == 3 and image.shape[2] == 3:  # Color image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image format: expected 2D grayscale or 3-channel BGR image.")

    if method == "hog":
        features = extract_hog_features(gray)
        return features

    if method == "color_histogram":
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    if method == "lbp":
        lbp = local_binary_pattern(gray, 8, 1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
        return hist.astype("float") / hist.sum()

    if method == "combined":
        hog_features = extract_features(image, "hog")
        color_features = extract_features(image, "color_histogram")
        lbp_features = extract_features(image, "lbp")
        return np.hstack([hog_features, color_features, lbp_features])

    raise ValueError(f"Unsupported feature extraction method: {method}")

# Dataset Creation
def create_dataset(image_paths, labels, size, method):
    X, y = [], []
    for idx, (img_path, x_min, y_min, x_max, y_max) in enumerate(image_paths):
        try:
            img = preprocess_image(img_path, (x_min, y_min, x_max, y_max), size)
            features = extract_features(img, method)
            X.append(features)
            y.append(labels[idx])
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    return np.array(X), np.array(y)

# Save Cropped Samples
def save_samples(image_paths, labels, output_dir, num_samples, size):
    ensure_dir_exists(output_dir)
    for i, (img_path, x_min, y_min, x_max, y_max) in enumerate(image_paths[:num_samples]):
        try:
            img = preprocess_image(img_path, (x_min, y_min, x_max, y_max), size)
            original_filename = os.path.splitext(os.path.basename(img_path))[0]
            label = labels[i]
            output_path = os.path.join(output_dir, f"{original_filename}_{label}.png")
            cv2.imwrite(output_path, img * 255)  # Save as 0–255 image
        except Exception as e:
            print(f"Error saving sample {img_path}: {e}")


def check_feature_range(features, feature_name):
    """
    Checks the range of the feature array and prints its minimum, maximum, mean, and standard deviation.
    Args:
        features: Feature array.
        feature_name: Name of the feature for logging.
    """
    print(f"Feature: {feature_name}")
    print(f"  Min: {np.min(features)}")
    print(f"  Max: {np.max(features)}")
    print(f"  Mean: {np.mean(features)}")
    print(f"  Std Dev: {np.std(features)}")
    print("-" * 40)


def main():
    config = CONFIG

    # Load Annotations
    data = load_yaml(config["yaml_path"])
    image_paths, labels = [], []

    for item in data:
        img_path = os.path.join(config["base_path"], item['path'])
        for box in item['boxes']:
            if box['label'] in config["selected_labels"]:
                bbox = (int(box['x_min']), int(box['y_min']), int(box['x_max']), int(box['y_max']))
                image_paths.append((img_path, *bbox))
                labels.append(box['label'])

    feature_extractor_method = config["feature_extractor"]
    X, y = create_dataset(image_paths, labels, config["image_size"], feature_extractor_method)

    # Check feature range
    check_feature_range(X, feature_extractor_method)

    # Encode Labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=config["test_size"], random_state=config["random_state"], stratify=y_encoded
    )

    # Save Samples and Dataset
    save_samples(image_paths, labels, config["output_dir"], config["num_samples_to_save"], config["image_size"])
    np.save(f"X_train_{feature_extractor_method}.npy", X_train)
    np.save(f"X_test_{feature_extractor_method}.npy", X_test)
    np.save(f"y_train_{feature_extractor_method}.npy", y_train)
    np.save(f"y_test_{feature_extractor_method}.npy", y_test)



if __name__ == "__main__":
    main()
