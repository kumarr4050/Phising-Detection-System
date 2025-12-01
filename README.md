# Phishing Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Scikit--Learn-orange)

## 📌 Project Overview
This project implements a comprehensive **Phishing Detection System** using a multi-paradigm approach. We benchmark **Classical Machine Learning**, **Deep Learning**, and **Transformer-based** architectures on a feature-rich tabular dataset. The goal is to identify malicious URLs with high precision and low latency.

## 📊 Dataset Statistics
The dataset consists of extracted features from URLs.

| Metric | Description |
| :--- | :--- |
| **Input Type** | Tabular (Numerical Features) |
| **Features** | URL Length, Special Char Counts, Domain Age, etc. |
| **Classes** | Binary (0: Legitimate, 1: Phishing) |
| **Split** | 80% Training, 20% Testing |

## 🧠 Models Implemented

| Model Type | Architecture | Description |
| :--- | :--- | :--- |
| **Classical ML** | **Logistic Regression** | Baseline linear classifier. |
| **Classical ML** | **Random Forest** | Ensemble of decision trees for robust feature importance. |
| **Deep Learning** | **MLP (Dense)** | Multi-Layer Perceptron with Dropout. |
| **Deep Learning** | **CNN (1D)** | Convolutional Neural Network for local feature pattern extraction. |
| **Deep Learning** | **LSTM** | Long Short-Term Memory network treating features as a sequence. |
| **Transformer** | **Tabular Transformer** | Self-Attention mechanism applied to feature vectors. |

## 🚀 Setup & Installation

1.  **Clone the Repository**
    ```bash
    git clone <repository_url>
    cd Phising Detection
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare Data**
    - Place your `dataset.csv` in the `data/` directory.

## 🏃 Usage

### Training
Run the main training script to train all models and generate a results summary.

```bash
python src/train.py --epochs 20 --batch_size 32
```

### Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--epochs` | 10 | Number of training epochs for DL models. |
| `--batch_size` | 32 | Batch size for training. |

## 📂 Directory Structure

```plaintext
Phising Detection/
├── data/               # Dataset storage
├── docs/               # Documentation & Reports
├── notebooks/          # Exploratory Data Analysis
├── src/                # Source Code
│   ├── models.py       # Model architectures (ML, DL, Transformer)
│   ├── preprocessing.py# Data cleaning & scaling pipeline
│   └── train.py        # Main training entry point
├── LICENSE             # MIT License
├── README.md           # Project Documentation
└── requirements.txt    # Python Dependencies
```

## 📜 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
