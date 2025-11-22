# Stroke Detection using Machine Learning

A machine learning project for predicting stroke occurrence based on patient health data. This project addresses the critical challenge of early stroke detection using classification algorithms and deep learning techniques.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Problem Statement](#problem-statement)
- [Installation](#installation)
- [Usage](#usage)
- [Models and Methodology](#models-and-methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Academic Context](#academic-context)
- [License](#license)

## Overview

Stroke (cerebrovascular accident) is a medical emergency that can result in death or permanent disability. Early detection is critical as treatment must be administered within the first few hours of symptom appearance. This project implements and compares multiple machine learning approaches to predict stroke occurrence based on patient demographics, health conditions, and lifestyle factors.

## Dataset

The dataset contains **43,400 patient records** with the following features:

- **Demographics**: id, gender, age
- **Health Conditions**: hypertension, heart_disease
- **Lifestyle Factors**: ever_married, work_type, Residence_type, smoking_status
- **Medical Metrics**: avg_glucose_level, bmi
- **Target Variable**: stroke (binary: 0 = no stroke, 1 = stroke)

### Dataset Characteristics

- **Total Records**: 43,400
- **Stroke Cases**: 783 (1.8%)
- **Non-Stroke Cases**: 42,617 (98.2%)
- **Missing Values**:
  - BMI: 1,462 missing values
  - Smoking Status: 13,292 missing values

## Problem Statement

The primary challenge in this project is the **highly imbalanced dataset** where stroke cases represent only 1.8% of the data. This imbalance can lead to models that achieve high overall accuracy but fail to detect actual stroke cases (low recall/sensitivity).

For medical applications, **recall (sensitivity)** is critical as failing to detect a stroke case has severe consequences.

## Installation

### Prerequisites

- Python 3.7+
- Jupyter Notebook or Google Colab
- GPU recommended for neural network training

### Dependencies

```bash
pip install pandas numpy scikit-learn tensorflow imbalanced-learn matplotlib seaborn
```

Or install all dependencies:

```bash
pip install pandas numpy scikit-learn tensorflow keras imbalanced-learn matplotlib seaborn
```

### Running in Google Colab

This project is designed to run in Google Colab. Simply upload the notebook and ensure your dataset is accessible via Google Drive.

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/stroke_detection.git
   cd stroke_detection
   ```

2. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook classification_stroke.ipynb
   ```

3. **Follow the notebook sections**:
   - Data Loading
   - Exploratory Data Analysis
   - Data Preprocessing
   - Model Training and Evaluation

## Models and Methodology

### Data Preprocessing

1. **Missing Data Handling**:
   - Used **KNN Imputation** (k=5) to preserve stroke cases
   - Alternative to deletion which would lose valuable minority class data

2. **Feature Encoding**:
   - Label encoding for categorical variables
   - Features: gender, ever_married, work_type, Residence_type, smoking_status

3. **Handling Class Imbalance**:
   - **SMOTE** (Synthetic Minority Over-sampling Technique)
   - **Random Undersampling**
   - **Class Weighting** (for neural networks)

### Models Implemented

#### Traditional Machine Learning

1. **Logistic Regression** (Baseline)
   - Simple, interpretable model
   - Used as performance baseline

2. **Support Vector Machine (SVM)**
   - Kernel-based classification
   - Tested with different data balancing strategies

3. **Random Forest**
   - Ensemble learning method
   - Best traditional ML model
   - 78% recall with undersampling

#### Deep Learning

**Artificial Neural Network (ANN)**
- **Architecture**:
  - Input Layer: 11 features
  - Hidden Layer 1: 64 neurons (ReLU activation)
  - Batch Normalization
  - Hidden Layer 2: 32 neurons (ReLU activation)
  - Batch Normalization
  - Output Layer: 1 neuron (Sigmoid activation)

- **Training Configuration**:
  - L2 Regularization (0.01)
  - Learning Rate Scheduling (halves every 50 epochs)
  - Early Stopping (patience=25)
  - Class weights for imbalance handling
  - Optimizers tested: Adam and RMSprop

## Results

### Best Performing Models

| Model | Data Balancing | Recall (Stroke Detection) |
|-------|---------------|---------------------------|
| Random Forest | Undersampling | 78% |
| ANN (Adam) | Class Weighting | **84%** |
| ANN (RMSprop) | Class Weighting | 82% |

### Key Findings

- **Neural network with Adam optimizer achieved the best stroke detection rate** at 84% recall
- Undersampling performed better than SMOTE for traditional ML models
- Class weighting in neural networks effectively addressed imbalance
- Recall/sensitivity is more important than overall accuracy for this medical application

## Project Structure

```
stroke_detection/
├── classification_stroke.ipynb          # Main implementation notebook
├── Introduction_to_AI_Final_Project.pdf # Project assignment
├── report.pdf                           # Detailed project report (Persian)
└── README.md                            # Project documentation
```

## Academic Context

This project was completed as a final project for an **Introduction to Artificial Intelligence** course (Fall 2023). The project demonstrates:

- Data preprocessing and feature engineering
- Handling imbalanced datasets in medical applications
- Comparison of traditional ML and deep learning approaches
- Model evaluation with focus on clinically relevant metrics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is available for educational and research purposes.

## Contact

For questions or collaborations, please open an issue in the repository.

## Acknowledgments

- Dataset source and preprocessing methodology
- Course instructors and teaching assistants
- Machine learning community for best practices in handling imbalanced datasets
