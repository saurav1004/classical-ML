import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generating synthetic data for linear regression
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Simple Linear Regression Implementation
class SimpleLinearRegression:
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X, y):
        # Calculating the coefficients
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add x0 = 1 to each instance
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = theta_best[0][0]
        self.coef_ = theta_best[1][0]

    def predict(self, X):
        return self.intercept_ + self.coef_ * X

# Fitting the model to the synthetic data
slr = SimpleLinearRegression()
slr.fit(X, y)
predictions = slr.predict(X)

# Evaluating the model
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

# Using scikit-learn LinearRegression for comparison
lin_reg = LinearRegression()
lin_reg.fit(X, y)
sk_predictions = lin_reg.predict(X)

# Plotting the results
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, predictions, color='red', linewidth=2, label='SLR Model')
plt.plot(X, sk_predictions, color='green', linestyle='dashed', label='Sklearn Model')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

# Printing the parameters and metrics
print(f"Simple Linear Regression intercept: {slr.intercept_:.2f}, coefficient: {slr.coef_:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared: {r2:.2f}")
print(f"Scikit-learn intercept: {lin_reg.intercept_[0]:.2f}, coefficient: {lin_reg.coef_[0][0]:.2f}")
