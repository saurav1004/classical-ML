import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None

    def fit(self, X, y):
        n, m = X.shape
        I = np.eye(m)
        I[0, 0] = 0  # Bias term is not regularized
        self.weights = np.linalg.inv(X.T.dot(X) + self.alpha * I).dot(X.T).dot(y)

    def predict(self, X):
        return X.dot(self.weights)


class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-3):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None

    def soft_threshold(self, rho, lambda_):
        if rho < - lambda_:
            return (rho + lambda_)
        elif rho >  lambda_:
            return (rho - lambda_)
        else:
            return 0

    def fit(self, X, y):
        n, m = X.shape
        self.weights = np.zeros(m)
        for iteration in range(self.max_iter):
            start_weights = np.copy(self.weights)
            for j in range(m):
                tmp_weights = np.copy(self.weights)
                tmp_weights[j] = 0.0
                r_j = y - X.dot(tmp_weights)
                rho_j = X[:, j].dot(r_j)
                self.weights[j] = self.soft_threshold(rho_j, self.alpha)
            if np.sum(np.abs(self.weights - start_weights)) < self.tol:
                break

    def predict(self, X):
        return X.dot(self.weights)


# Usage
if __name__ == "__main__":
    # Generate some data
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = X.dot(np.array([1.5, -2.0, 3.0])) + 0.5 * np.random.randn(100)
    y = y[:, np.newaxis]

    # Ridge Regression
    ridge_reg = RidgeRegression(alpha=1.0)
    ridge_reg.fit(X, y)
    ridge_pred = ridge_reg.predict(X)

    # Lasso Regression
    lasso_reg = LassoRegression(alpha=0.1)
    lasso_reg.fit(X, y)
    lasso_pred = lasso_reg.predict(X)

    print("Ridge coefficients:", ridge_reg.weights.flatten())
    print("Lasso coefficients:", lasso_reg.weights.flatten())
