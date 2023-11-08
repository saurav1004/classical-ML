import numpy as np

class HuberRegressor:
    def __init__(self, epsilon=1.35, max_iter=100, learning_rate=0.01):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weights = None

    def huber_loss_derivative(self, error):
        # Calculate the derivative of the Huber loss
        return np.where(np.abs(error) <= self.epsilon, error, self.epsilon * np.sign(error))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights with zeros
        self.weights = np.zeros(n_features)
        
        for _ in range(self.max_iter):
            # Calculate predictions
            y_pred = X.dot(self.weights)
            # Calculate errors
            errors = y - y_pred
            # Calculate gradient
            gradient = -np.dot(X.T, self.huber_loss_derivative(errors)) / n_samples
            # Update weights
            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        return X.dot(self.weights)

# Generating synthetic data with an outlier
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 0.5 * X + np.random.normal(size=X.size)
# Introducing outliers
y[::10] = 20 * (0.5 + np.random.normal(size=y[::10].size))

# Reshape X for our model
X = X[:, np.newaxis]

# Fit our Huber regressor
huber = HuberRegressor()
huber.fit(X, y)

# Compare with scikit-learn's implementation
from sklearn.linear_model import HuberRegressor as SklearnHuberRegressor

sk_huber = SklearnHuberRegressor(epsilon=1.35, max_iter=100)
sk_huber.fit(X, y)

# Plot the comparison
import matplotlib.pyplot as plt

plt.scatter(X, y, color='gray', label='Data')
plt.plot(X, huber.predict(X), color='red', label='Huber Regressor (from scratch)')
plt.plot(X, sk_huber.predict(X), color='blue', linestyle='--', label='Huber Regressor (scikit-learn)')
plt.legend()
plt.show()
