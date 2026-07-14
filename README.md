# Diabetes Prediction Machine Learning Project

## Overview
This repository contains a machine learning pipeline designed to predict the likelihood of a patient having diabetes based on their medical history and demographic data. It trains and compares multiple classification models and includes an interactive widget for real-time predictions.

## Dataset
The project uses the **Diabetes Prediction Dataset**, which contains the following features:
* **Demographics:** `gender`, `age`
* **Medical History:** `hypertension`, `heart_disease`, `smoking_history`
* **Health Metrics:** `bmi`, `HbA1c_level`, `blood_glucose_level`
* **Target Variable:** `diabetes` (0 = Not Diabetic, 1 = Diabetic)

## Data Preprocessing
* **Handling Duplicates:** Removed duplicate entries from the dataset to ensure reliable evaluation.
* **Encoding:** Categorical variables (`gender` and `smoking_history`) were converted into numerical formats using `LabelEncoder`.
* **Class Imbalance:** Addressed the class imbalance in the target variable (`diabetes`) using `RandomOverSampler` from the `imblearn` library.

## Machine Learning Models
The following models were trained and evaluated on the resampled data:
1. **Decision Tree Classifier:** Achieved an accuracy of ~98.4%.
2. **Random Forest Classifier:** Achieved an excellent accuracy of ~99.1%.
3. **Logistic Regression:** Evaluated as a baseline binary classification model.

## Hyperparameter Tuning
* Utilized `GridSearchCV` to optimize the Random Forest model with 3-fold cross-validation.
* Parameters evaluated include `n_estimators`, `max_depth`, and `min_samples_split`.

## Interactive Prediction Widget
The notebook includes an interactive user interface built with `ipywidgets`. Users can input health metrics (Gender, Age, BMI, HbA1c, etc.) via dropdowns and text fields to instantly receive a "Diabetic" or "Not Diabetic" prediction from the trained model.

## Requirements
To run this project, you need the following Python libraries:
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `imbalanced-learn` (`imblearn`)
* `joblib`
* `ipywidgets`

## Model Export
The best-performing model is exported as `diabetes_prediction_model.pkl` using `joblib` for easy deployment and future inference outside of the notebook environment.
