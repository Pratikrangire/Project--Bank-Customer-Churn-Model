# Project--Bank-Customer-Churn-Model

# Title of Project
Bank Customer Churn Model

## Objective
The objective of this project is to predict whether a bank customer will churn (leave the bank) using various customer attributes. This prediction can help the bank in implementing retention strategies to maintain its customer base.

## Data Source
The data source for this project is a bank's customer data, which includes attributes such as customer age, tenure, balance, number of products, and more.

## Import Library
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

## Import Data
data = pd.read_csv('https://github.com/Pratikrangire/DataSet/raw/main/Bank%20Churn%20Modelling.csv')

## Describe Data
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())

## Data Visualization
# Example visualizations
sns.countplot(x='Churn', data=data)
plt.title('Distribution of Churn')
plt.show()

sns.boxplot(x='Churn', y='Age', data=data)
plt.title('Age Distribution by Churn Status')
plt.show()

## Data Preprocessing
# Handling missing values if any
data = data.dropna()

# Encoding categorical variables
data = pd.get_dummies(data, drop_first=True)

# Feature scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

## Define Target Variable (y) and Feature Variables (X
X = data.drop('Churn', axis=1)
y = data['Churn']

## Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeling
model = LogisticRegression()
model.fit(X_train, y_train)

## Model Evaluation
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

## Prediction
new_data = [[...]]  # Example new data for prediction
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print(prediction)

## Explanation

This project involves several key steps, starting from importing libraries and data to data visualization, preprocessing, model training, and evaluation. Each step plays a crucial role in building an accurate predictive model. Visualizations help in understanding the data distribution and relationships between variables. Preprocessing ensures that the data is clean and suitable for modeling. The Logistic Regression model is used for its simplicity and effectiveness in binary classification tasks. Model evaluation metrics like accuracy, confusion matrix, and classification report help in assessing the model's performance.
