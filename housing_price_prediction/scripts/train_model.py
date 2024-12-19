from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from preprocess_data import load_and_preprocess_data
import numpy as np

# Load the processed data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Function to test different alpha values in Ridge Regression
def test_ridge_alphas(alpha_values):
    best_r2 = 0
    best_alpha = 0

    for alpha in alpha_values:
        ridge_model = Ridge(alpha=alpha)
        ridge_model.fit(X_train, y_train)
        y_test_pred_ridge = ridge_model.predict(X_test)
        r2 = r2_score(y_test, y_test_pred_ridge)
        
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha
        print(f"Alpha: {alpha}, Test R-squared: {r2:.2f}")

    print(f"Best Alpha: {best_alpha}, Best Test R-squared: {best_r2:.2f}")

# Define the range of alpha values to test
alpha_values = [0.1, 1, 10, 100, 1000]
test_ridge_alphas(alpha_values)