import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Load the trained model
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

# Extract feature importances from the model
feature_importances = model.named_steps['randomforestregressor'].feature_importances_
feature_names = numerical_columns + list(model.named_steps['columntransformer'].named_transformers_['categorical'].named_steps['encode'].get_feature_names_out(categorical_columns))

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sort features by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

# Display feature importances using a bar plot with improved styling
st.subheader('Feature Importances')
fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, palette='viridis')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importances', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
st.pyplot(fig)