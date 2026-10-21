# Runtime-adaptive-embedded-AI

This repository contains the source code for the MSc thesis: **"Runtime-adaptive embedded AI systems for resource-constrained edge devices"**.

## 📌 Project Overview
Briefly describe what your thesis is about here. (e.g., This project focuses on implementing adaptive AI models that can dynamically adjust their computational complexity based on the real-time resource constraints of edge hardware...)

## 📁 Repository Structure
```text
├── .gitignore              # Git ignore configuration
├── export_noise.py         # Script for noise generation/export
├── mix_dataset.py          # Data augmentation and dataset mixing
├── models.py               # Definition of neural network architectures
└── README.md               # Project documentation (this file)
```
## 🚀 Getting Started

### 1. Prerequisites
Ensure you have the following environment installed:
* Python 3.8+
* PyTorch 1.10+
* (Add other dependencies here, e.g., numpy, torchvision...)

### 2. Dataset Preparation
> ⚠️ **Note:** Due to GitHub's file size limits, the raw datasets and trained model weights are excluded from this repository.
* Please download the required dataset from `[Insert Link Here, e.g., Kaggle/Google Drive]`.
* Extract and place the dataset folder (e.g., `train_snr_15/`) directly in the project root directory.

### 3. Running the Code
To mix the dataset:
```bash
python mix_dataset.py
```
To run noise export or related operations:
```bash
python export_noise.py
```

## 📄 License
This project is licensed under the MIT License.