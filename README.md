# Image Classification Project with MobileNetV2 and MLFlow

This project trains an image classification model using MobileNetV2 and tracks experiments with MLflow.

## Prerequisites

Before starting, ensure you have the following installed:

* Python 3.7+
* TensorFlow
* MLflow
* Other necessary dependencies (e.g., NumPy, Pandas, etc.)  Install these using `pip install -r requirements.txt` if you have a `requirements.txt` file, or individually using `pip install <package_name>`.

## Getting Started

1. **Clone the repository:** (Instructions assuming you are using git, adapt if necessary for your source code control)

   ```bash
   git clone <your_repository_url>
   cd ImageClassificationProj 

2. **Set up a virtual environment (recommended):** 

    python3 -m venv .venv
    source .venv/bin/activate  (.venv\Scripts\activate on Windows)

3. **Install dependencies:**

    pip install -r requirements.txt

## Running the Training and MLflow 

1. **Start MLflow UI:**
    mlflow ui

2. **Run the training script:**
    python MobileNet_train_mlflow.py

3. **View results in MLflow:**
    Open your web browser and navigate to the MLflow UI address (usually http://127.0.0.1:5000). You should see your experiment runs listed, with logged metrics, parameters, and artifacts. You can compare different runs, visualize metrics, and download logged models or other files.

## Project Structure (Example)
The project structure should be organized for clarity:

ImageClassificationProj/
├── data/             (Contains your image datasets)
├── MobileNet_train_mlflow.py   (Your training script)
├── requirements.txt   (Lists project dependencies)
└── README.md         (This file)
Troubleshooting
MLflow UI not accessible: Ensure the MLflow server is running and check your firewall settings if necessary.
Dependency issues: Ensure all dependencies are correctly installed in your virtual environment. Double check your requirements.txt file.
Script errors: Carefully examine the output of your training script for any error messages.