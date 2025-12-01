# Phishing Detection System

## Project Overview
This project implements a comprehensive Phishing Detection system using Classical Machine Learning, Deep Learning (CNN, LSTM), and Transformer-based (DistilBERT) approaches. It is designed to benchmark various models on Phishing URL datasets and generate a Scopus-ready report.

## Directory Structure
- `data/`: Contains raw and processed datasets.
- `src/`: Source code for the project.
    - `preprocessing.py`: Data cleaning and feature extraction.
    - `models.py`: Model definitions (ML, DL, Transformers).
    - `train.py`: Training and evaluation scripts.
- `notebooks/`: Jupyter notebooks for analysis.
- `docs/`: Documentation, architecture diagrams, and report assets.

## Setup & Installation
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `requirements.txt` to be generated)*

## Usage
1.  **Data Preparation**: Place your dataset in `data/` and run preprocessing.
2.  **Training**: Run `python src/train.py` to train and evaluate models.
3.  **Results**: Check the console output and generated logs for performance metrics.

## Models Implemented
- **Classical**: Logistic Regression, Random Forest.
- **Deep Learning**: CNN, LSTM, CNN-LSTM.
- **Transformers**: DistilBERT (Fine-tuned).
