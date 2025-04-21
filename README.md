# wine_quality_project.py
# Step 1: Load Dataset & Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# Step 2: Data Cleaning & Transformation
print(df.info())
print(df.isnull().sum())
df.columns = df.columns.str.replace(' ', '_')

# Feature & target
X = df.drop('quality', axis=1)
y = df['quality']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Exploratory Data Analysis (EDA)
sns.countplot(x='quality', data=df)
plt.title("Distribution of Wine Quality")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title("Feature Distributions and Outliers")
plt.show()

# Step 4: Build Regression Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Step 5: Export Model
joblib.dump(model, 'wine_quality_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Step 6: Streamlit App (save this as app.py)
# import streamlit as st
# model = joblib.load("wine_quality_model.pkl")
# scaler = joblib.load("scaler.pkl")
# st.title("Wine Quality Prediction")
# features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
#             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
#             'density', 'pH', 'sulphates', 'alcohol']
# input_data = []
# for feature in features:
#     val = st.number_input(f"{feature}", min_value=0.0, step=0.1)
#     input_data.append(val)
# if st.button("Predict Quality"):
#     data_scaled = scaler.transform([input_data])
#     prediction = model.predict(data_scaled)
#     st.success(f"Predicted Wine Quality: {round(prediction[0], 2)}")

# Step 7: Run the app with: streamlit run app.py
# Step 8: Record a demo video and upload all files to GitHub.
