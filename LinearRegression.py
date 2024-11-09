# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Load the dataset
data = pd.read_csv("data/StartUpProfits.csv")

# Display the first few rows to understand the data structure
print(data.head())

# Check the number of samples (rows) and features (columns)
num_samples, num_features = data.shape
print(f"Number of samples: {num_samples}")
print(f"Number of features: {num_features}")

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# Perform one-hot encoding for the 'State' column
data = pd.get_dummies(data, columns=['State'], drop_first=True)

# Display the first few rows to confirm the transformation
print(data.head())

# Check the new shape of the data after encoding
num_samples, num_features = data.shape
print(f"Updated number of samples: {num_samples}")
print(f"Updated number of features after encoding: {num_features}")


# Create a folder to save plots if it doesn't exist
output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)

# Calculate the correlation matrix
correlation_matrix = data.corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix with a heatmap and save it
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix Heatmap")
heatmap_path = os.path.join(output_folder, "correlation_matrix_heatmap.png")
plt.savefig(heatmap_path)
plt.show()

# Plot scatter plots to visualize relationships with Profit and save them
features = ['R&D Spend', 'Administration', 'Marketing Spend']
plt.figure(figsize=(15, 4))

for i, feature in enumerate(features, 1):
    plt.subplot(1, 3, i)
    sns.scatterplot(data=data, x=feature, y='Profit')
    plt.title(f'{feature} vs Profit')

# Save the scatter plots as a single figure
scatter_plots_path = os.path.join(output_folder, "scatter_plots.png")
plt.tight_layout()
plt.savefig(scatter_plots_path)
plt.show()

# Split the data into features (X) and target (y)
X = data.drop(columns=['Profit'])
y = data['Profit']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Visualize the predictions vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Profit")
plt.savefig("plots/actual_vs_predicted_profit.png")  # Save the plot to the 'plots' folder
plt.show()


# Perform cross-validation on the linear regression model
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

# Display cross-validation results
print("Cross-Validation R² Scores:", cv_scores)
print("Mean Cross-Validation R² Score:", cv_scores.mean())


# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_y_pred = ridge_model.predict(X_test)
ridge_r2 = r2_score(y_test, ridge_y_pred)
print(f"Ridge Regression R² Score: {ridge_r2:.4f}")

# Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_y_pred = lasso_model.predict(X_test)
lasso_r2 = r2_score(y_test, lasso_y_pred)
print(f"Lasso Regression R² Score: {lasso_r2:.4f}")


# Create a pipeline to add polynomial features and fit a linear regression model
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train, y_train)
poly_y_pred = poly_model.predict(X_test)

# Evaluate the polynomial model
poly_r2 = r2_score(y_test, poly_y_pred)
print(f"Polynomial Regression (Degree 2) R² Score: {poly_r2:.4f}")


# Calculate residuals
residuals = y_test - y_pred

# Scatter plot of residuals
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='blue')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Profit")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Profit")
plt.savefig("plots/residuals_vs_predicted_profit.png")  # Save plot to 'plots' folder
plt.show()

# Histogram of residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.savefig("plots/residuals_histogram.png")  # Save plot to 'plots' folder
plt.show()
