Dataset for Disease Prediction

This folder contains the datasets used for training machine learning models in the project "Prediction of Disease Outbreaks Using Machine Learning." These datasets include medical attributes relevant to predicting Diabetes, Heart Disease, and Parkinson’s Disease.

Contents of This Folder:
diabetes.csv – Contains patient data for diabetes prediction based on features like glucose levels, BMI, and age.
heart.csv – Includes heart health parameters such as cholesterol levels, blood pressure, and ECG results for heart disease prediction.
parkinsons.csv – Consists of voice-related features used to identify signs of Parkinson’s disease.

Dataset Details:
1. Diabetes Dataset (diabetes.csv)
  Features:
    Glucose Level
    Blood Pressure
    BMI (Body Mass Index)
    Age
    Insulin Level
    Other medical parameters
    Target: Diabetes diagnosis (1 = Yes, 0 = No)
2. Heart Disease Dataset (heart.csv)
    Features:
    Age
    Cholesterol Level
    Resting Blood Pressure
    Maximum Heart Rate
    ECG Results
    Chest Pain Type
    Target: Heart disease presence (1 = Yes, 0 = No)
3. Parkinson’s Disease Dataset (parkinsons.csv)
  Features:
    Vocal Tremors
    Jitter & Shimmer (Voice Variations)
    Fundamental Frequency (Fo)
    HNR (Harmonic-to-Noise Ratio)
    Dysphonia Measures
  Target: Parkinson’s diagnosis (1 = Yes, 0 = No)

Usage Instructions:
These datasets are used for training, validating, and testing machine learning models.
Ensure proper data preprocessing, including handling missing values and feature scaling before training models.

Requirements:
Python 3.8+
Pandas for data handling
NumPy for numerical operations
