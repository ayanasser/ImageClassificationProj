import os
import cv2
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from skimage.feature import local_binary_pattern
from skimage.feature import hog


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
                x_min, y_min = int(box['x_min']), int(box['y_min'])
                x_max, y_max = int(box['x_max']), int(box['y_max'])
                # print(f'Image: {img_path}, Label: {label}, BBox: ({x_min}, {y_min}, {x_max}, {y_max})')
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

    # Print image shape
    # print(f"Image Shape: {img.shape}, Path: {img_path}")

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



def extract_hog_features(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract HOG features
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


def extract_color_histogram(image, bins=(8, 8, 8)):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Compute histogram and normalize it
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_lbp_features(image, radius=1, n_points=8):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

    # Normalize the histogram
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist


def extract_combined_features(image):
    hog_features = extract_hog_features(image)
    color_histogram = extract_color_histogram(image)
    lbp_features = extract_lbp_features(image)
    return np.hstack([hog_features, color_histogram, lbp_features])



def create_dataset_with_features(image_paths, labels, size=(64, 64)):
    X = []
    y = []
    for idx, (img_path, x_min, y_min, x_max, y_max) in enumerate(image_paths):
        try:
            # Load image and preprocess
            img = preprocess_image(img_path, (x_min, y_min, x_max, y_max), size)

            # Extract features
            features = extract_combined_features(img)

            # Append features and label
            X.append(features)
            y.append(labels[idx])
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
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
        os.makedirs(output_dir)

    # Process the specified number of samples
    for i, (img_path, x_min, y_min, x_max, y_max) in enumerate(image_paths[:num_samples]):
        # Print the base path for debugging
        print(f"Processing image: {img_path}")

        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {img_path}")
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
        print(f"Saved sample: {output_path}")


# Main Function
def main():
    yaml_path = 'train.yaml'  # Path to YAML file
    base_path = ''  # Path to base image folder

    # Load annotations
    print(f'Loading annotations from {yaml_path}')
    data = load_labels(yaml_path)
    print(f'Found {len(data)} annotations')
    image_paths, labels = parse_annotations(data, base_path)

    # Preprocess Dataset
    print('Preprocessing dataset...')
    X, y = create_dataset_with_features(image_paths, labels)

    # Encode Labels
    print('Encoding labels...')
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Split into Train/Test Sets
    print('Splitting dataset...')
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    print(f'Train data shape: {X_train.shape}, Test data shape: {X_test.shape}')
    print(f'Classes: {encoder.classes_}')

    save_samples(image_paths, labels, output_dir='samples/', num_samples=10)


    # Save Preprocessed Data
    np.save('X_train_f.npy', X_train)
    np.save('X_test_f.npy', X_test)
    np.save('y_train_f.npy', y_train)
    np.save('y_test_f.npy', y_test)

if __name__ == '__main__':
    main()