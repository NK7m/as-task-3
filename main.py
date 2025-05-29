# Linear Regression Task - AI/ML Internship
# Some minor bugs might exist - still learning!

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset (I used Boston housing but couldn't find it so using California housing)
# Oops - just realized Boston is deprecated, using California instead
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target

print("\nDataset head:")
print(df.head())  # Forgot to limit to 5 rows initially

# Simple Linear Regression (using MedInc as feature)
X = df[['MedInc']]  # Double brackets - took me a while to get this right
y = df['Target']

# Split data - forgot to set random_state at first
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Metrics:")
print(f"MAE: {mae:.3f}")  # Formatting decimals was tricky
print(f"MSE: {mse:.3f}")
print(f"R²: {r2:.3f}")

# Plotting
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title('Simple Linear Regression (MedInc vs House Value)')
plt.xlabel('Median Income')
plt.ylabel('House Value')
plt.legend()
plt.savefig('regression_plot.png')  # Forgot this at first and plot didn't save
plt.show()

# Multiple Linear Regression (using all features)
X_multi = df.drop('Target', axis=1)
y_multi = df['Target']

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42)

multi_model = LinearRegression()
multi_model.fit(X_train_m, y_train_m)

# Coefficients
print("\nFeature Coefficients:")
for feature, coef in zip(X_multi.columns, multi_model.coef_):
    print(f"{feature}: {coef:.4f}")

# Multiple regression metrics
y_pred_m = multi_model.predict(X_test_m)
print("\nMultiple Regression R²:", r2_score(y_test_m, y_pred_m))
