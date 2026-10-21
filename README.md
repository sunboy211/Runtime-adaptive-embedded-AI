# Runtime-adaptive-embedded-AI

This repository contains the source code for the MSc thesis: **"Runtime-adaptive embedded AI systems for resource-constrained edge devices"**.

## 📌 Project Overview
This project investigates runtime-adaptive embedded AI systems for resource-constrained edge devices. Unlike conventional TinyML deployments with fixed models, the focus is on enabling AI systems to dynamically adapt their inference behavior according to available resources and changing operating conditions.

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
* torchaudio
* Numpy
* torchmetrics

### 2. Dataset Preparation
> ⚠️ **Note:** Due to GitHub's file size limits, the raw datasets and trained model weights are excluded from this repository.
* Please download the required dataset from `demand`.
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
