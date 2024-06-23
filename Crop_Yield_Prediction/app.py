import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the trained ensemble model
model = joblib.load('best_model.pkl')

# Define numerical and categorical columns
numerical_columns = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']
categorical_columns = ['Crop']

# Streamlit application title with improved styling
st.title('Crop Yield Prediction')
st.markdown("---")

# Sidebar for input features with improved styling
st.sidebar.title('Input Features')
st.sidebar.markdown("---")

# Dictionary to store input features
input_features = {}

# Create input fields for numerical features with improved styling
for col in numerical_columns:
    input_features[col] = st.sidebar.number_input(f'{col}', value=0.0, step=0.1)
st.sidebar.markdown("---")

# Dropdown select box for crop selection with improved styling
crop_names = [
    'Rice', 'Maize', 'Jute', 'Cotton', 'Coconut', 'Papaya', 'Orange', 'Apple',
    'Muskmelon', 'Watermelon', 'Grapes', 'Mango', 'Banana', 'Pomegranate',
    'Lentil', 'Blackgram', 'MungBean', 'MothBeans', 'PigeonPeas', 'KidneyBeans',
    'ChickPea', 'Coffee'
]
input_features['Crop'] = st.sidebar.selectbox('Crop', crop_names)
st.sidebar.markdown("---")

# Prediction button with improved styling
if st.sidebar.button('Predict', key='prediction_button'):
    input_df = pd.DataFrame([input_features])
    
    # Perform prediction
    prediction = model.predict(input_df)
    
    # Display prediction result with improved styling
    st.subheader('Prediction')
    st.write(f'Predicted Crop Yield: {prediction[0]:.2f}')
    st.markdown("---")

# Load dataset for model evaluation (assuming 'Crop_Yield_Prediction.csv' contains your data)
data = pd.read_csv('Crop_Yield_Prediction.csv')

# Split data into features and target
X = data.drop(columns=['Yield'])
y = data['Yield']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Display model performance with improved styling
st.subheader('Model Performance on Test Set')
st.write(f'Mean Absolute Error: {mae:.2f}')
st.write(f'Mean Squared Error: {mse:.2f}')
st.markdown("---")

# Visualize predictions vs actual values
st.subheader('Predictions vs Actual Values')
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(x=y_test, y=y_pred, ax=ax, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.title('Predictions vs Actual Values', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
st.pyplot(fig)

# Display residuals
st.subheader('Residuals')
residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(10, 8))
sns.histplot(residuals, kde=True, ax=ax, color='green')
plt.xlabel('Residuals', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Residuals Distribution', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
st.pyplot(fig)
