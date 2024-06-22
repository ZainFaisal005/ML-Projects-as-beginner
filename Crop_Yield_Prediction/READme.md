# Crop Yield Prediction

This project utilizes machine learning to predict crop yields based on various agricultural factors. It includes a Streamlit web application for user interaction and visualization of predictions and model performance metrics.

## Overview

The Crop Yield Prediction project aims to provide farmers and agricultural stakeholders with a tool to estimate crop yields based on key environmental and soil parameters. By leveraging machine learning techniques, the application offers insights into potential crop production, aiding in planning and decision-making processes.

### Features

- **Input Features:**
  - Users can input various agricultural parameters such as Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH Value, and Rainfall.
  - Crop selection via a dropdown menu to choose the specific crop for prediction.

- **Prediction:**
  - Real-time prediction of crop yield based on user-provided inputs using a Random Forest Regression model.

- **Model Performance:**
  - Evaluation of model accuracy through Mean Absolute Error (MAE) and Mean Squared Error (MSE) metrics on a test dataset.

- **Feature Importance:**
  - Visual representation of feature importance, highlighting which agricultural factors most significantly influence crop yield predictions.

## Dataset

The dataset used in this project is crucial for training and testing the machine learning model. It includes historical data on agricultural parameters and corresponding crop yields. The dataset can be accessed on Kaggle:

- [Crop Yield Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/varshitanalluri/crop-price-prediction-dataset)

Before running the application, download the dataset and place it in the project directory as `Crop_Yield_Prediction.csv`.

## Getting Started

### Prerequisites

To run the Crop Yield Prediction application locally, ensure you have the following dependencies installed:

- Python 3.6+
- Streamlit
- Pandas
- Joblib
- Matplotlib
- Seaborn
- Scikit-learn
