# Comprehensive Chest X-Ray Analysis: Classification and Segmentation

## Overview
This repository contains two distinct deep learning projects focused on analyzing chest X-rays using the UCL Adult Income dataset (Kaggle Chest X-Ray Pneumonia dataset). 

1. **Pneumonia Classification with Grad-CAM:** A TensorFlow/Keras-based CNN that classifies X-rays as Normal or Pneumonia, featuring visual explainability.
2. **Lung Segmentation with U-Net:** A PyTorch-based U-Net model that generates precise binary masks to segment lung tissue from the surrounding anatomy.

## Dataset
Both projects utilize the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle. The scripts include automated fetching and extraction using the Kaggle API. Images are organized into `train`, `val`, and `test` directories.

***

## Project 1: Pneumonia Classification with Grad-CAM

### Overview
This project trains a Convolutional Neural Network (CNN) to detect pneumonia from chest X-rays. It utilizes transfer learning via a pre-trained ResNet50 model and includes a custom Grad-CAM (Gradient-weighted Class Activation Mapping) implementation to visually explain the model's decision-making process.

### Key Features
* **Transfer Learning:** Leverages the ResNet50 architecture pre-trained on ImageNet.
* **Two-Phase Training Strategy:** * Phase 1 trains only the custom dense classification layers.
    * Phase 2 fine-tunes the model by unfreezing the last 20 layers of the base model with a reduced learning rate.
* **Data Augmentation:** Applies dynamic data augmentation (rotations, spatial shifts, flips, zoom) to improve generalization.
* **Grad-CAM Interpretability:** Extracts gradients from the final convolutional layer to generate a heatmap. This highlights the exact anatomical regions that influenced the model's prediction, preventing the network from acting as a "black box."

***

## Project 2: Lung Segmentation with U-Net

### Overview
This project implements a U-Net architecture in PyTorch to perform semantic segmentation of the lungs. Since the base dataset lacks ground-truth segmentation masks, the script includes an automated pipeline to generate synthetic masks for training.

### Key Features
* **Synthetic Mask Generation:** Uses Gaussian blurring, Otsu's thresholding, and morphological operations (skimage) to create synthetic binary lung masks directly from the raw X-ray images.
* **PyTorch U-Net Architecture:** Implements a U-Net model with 1 input channel (grayscale X-ray) and 1 output channel (binary mask).
* **Custom Data Pipeline:** Features a custom PyTorch `Dataset` and `DataLoader` for handling grayscale image and mask pairings, including dynamic resizing and normalization.
* **Inference and Overlay Visualization:** Includes a robust prediction pipeline that generates a binary mask and overlays it onto the original X-ray using a transparent red highlight for easy visual evaluation.

***

## Requirements

To run the scripts in this repository, ensure you have the following Python libraries installed:

**General & Image Processing:**
* `numpy`
* `opencv-python` (`cv2`)
* `matplotlib`
* `scikit-image` (`skimage`)
* `pillow` (`PIL`)
* `kaggle`

**Deep Learning Frameworks:**
* `tensorflow` (For Classification)
* `torch` and `torchvision` (For Segmentation)

*Note: You will need a valid `kaggle.json` API token file uploaded to your environment to fetch the dataset automatically.*

## Usage

1. **Environment Setup:** It is highly recommended to run these scripts in an environment with GPU support (such as Google Colab).
2. **Kaggle Authentication:** Ensure your `kaggle.json` is placed in the `~/.kaggle/` directory before running the data-fetching cells.
3. **Running Classification:** Execute `cnn_for_pneumonia_classification_with_grad_cam.py`. The script will train the model, save the best weights as `best_pneumonia_model.h5`, and plot Grad-CAM heatmaps for sample test images.
4. **Running Segmentation:** Execute `u_net_segmentation.py`. The script will first generate synthetic masks, train the U-Net model, save the weights as `unet_lung_segmentation.pth`, and display overlay visualizations for sample test images.

## Author
* **Name:** Akshay Deepak
* **Roll No:** BE24B002
