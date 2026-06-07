# CNN for Pneumonia Classification with Grad-CAM

## Overview
This repository contains a Python script (`cnn_for_pneumonia_classification_with_grad_cam.py`) designed to train a Convolutional Neural Network (CNN) to classify chest X-ray images as either **Normal** or **Pneumonia**. The project utilizes transfer learning via a pre-trained ResNet50 model and includes a custom **Grad-CAM** (Gradient-weighted Class Activation Mapping) implementation to visually explain the model's decision-making process.

## Features
* **Automated Data Fetching:** Directly downloads and extracts the Kaggle Chest X-Ray (Pneumonia) dataset using the Kaggle API.
* **Transfer Learning:** Leverages the ResNet50 architecture pre-trained on ImageNet for robust and efficient feature extraction.
* **Two-Phase Training Strategy:**
    * **Phase 1:** Trains only the custom dense classification layers while keeping the base ResNet50 layers frozen.
    * **Phase 2:** Fine-tunes the model by unfreezing the last 20 layers of the ResNet50 base with a heavily reduced learning rate.
* **Data Augmentation:** Applies dynamic data augmentation (rotations, spatial shifts, flips, zoom) to the training set to improve generalization and prevent overfitting.
* **Model Interpretability:** Features a complete `GradCAM` class implementation to generate and overlay heatmaps, highlighting the exact regions of the X-ray that influenced the model's prediction.

## Requirements
To run this script, ensure you have the following Python libraries installed:
* `tensorflow`
* `opencv-python` (`cv2`)
* `numpy`
* `matplotlib`
* `scikit-learn`
* `kaggle`

*Note: You will also need a valid `kaggle.json` API token file uploaded to your environment (like Google Colab) to fetch the dataset automatically.*

## Dataset
The script is configured to use the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset. It automatically reads the standard directory structure (`train`, `val`, `test`) and scales all images to a target size of `256x256` pixels across 3 color channels.

## Model Architecture
1. **Base Model:** ResNet50 (excluding the top ImageNet classification head).
2. **Pooling:** Global Average Pooling 2D.
3. **Dense Head:** * Dense layer with 512 units and a ReLU activation function.
    * Dropout layer (0.5) for regularization.
    * Final Dense output layer with 1 unit and a Sigmoid activation function for binary classification.

## Usage
1. **Environment Setup:** It is highly recommended to run this code in an environment with GPU support (such as Google Colab), as indicated by the `!nvidia-smi` hardware checks in the script.
2. **Kaggle Authentication:** When executing the script, you will be prompted to upload your `kaggle.json` file. This authenticates the Kaggle API to download the dataset.
3. **Automated Training:** The script runs the entire preprocessing and training pipeline automatically. It utilizes `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` callbacks to optimize the learning rate dynamically and save the best model weights as `best_pneumonia_model.h5`.
4. **Visual Analysis:** Once training is complete, the script will automatically select random sample images from the test set (from both the Normal and Pneumonia classes) and display the Grad-CAM overlays alongside the original X-rays.

## Explainability: Grad-CAM
To prevent the neural network from functioning purely as a "black box," this script includes a custom `GradCAM` class. It extracts the gradients of the target class flowing into the final convolutional layer of ResNet50, generating a heatmap that maps exactly where the model is looking to determine if lungs show signs of pneumonia. 

The script provides two primary functions for this visual analysis:
* `analyze_sample_images(generator, gradcam_analyzer, model, num_samples)`: Analyzes and plots a batch of images fetched from a Keras ImageDataGenerator.
* `analyze_single_image(image_path, gradcam_analyzer)`: Performs a detailed Grad-CAM analysis on a specific local image path and saves the output visualization to your directory.
