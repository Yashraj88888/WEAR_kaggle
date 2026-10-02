# Human Activity Recognition Using Wearable Sensor Data

## Project Overview

This project explores deep learning models for human activity recognition using wearable inertial sensor data. The overall study compares MLP, 1D-CNN, BiLSTM, and Transformer architectures.

## My Contribution: 1D-CNN

Implemented a one-dimensional Convolutional Neural Network (1D-CNN) using accelerometer data from four body locations.

* **Window size:** 50 samples
* **Input features:** 12 accelerometer features
* **Activity classes:** 18
* **Validation strategy:** Participant-based split, with participants 20 and 21 held out for validation

## Preliminary Results

| Metric              |  Score |
| ------------------- | -----: |
| Validation Accuracy | 85.84% |
| Macro Precision     | 87.15% |
| Macro Recall        | 85.87% |
| Macro-F1            | 85.71% |

These are preliminary validation results, not final test-set results.

## Dataset

[3rd WEAR Dataset Challenge — Kaggle](https://www.kaggle.com/competitions/3rd-wear-dataset-challenge-hasca-2026/data)

The dataset is not included in this repository. Access may require a Kaggle account and acceptance of the competition rules.

## Repository Contents

* `notebooks/`: CNN implementation and evaluation
* `results/`: Validation metrics
* `models/`: Saved trained CNN model, if included

## Technologies

Python, TensorFlow/Keras, NumPy, Pandas, scikit-learn, Matplotlib.
