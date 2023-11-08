import numpy as np

class PolynomialRegression:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias
        self.weights = None

    def _transform_input(self, X):
        """Transforms input X into a polynomial feature matrix."""
        n_samples, n_features = np.shape(X)
        def _get_combinations(x, degree):
            if degree == 0:
                return [[]]
            combinations = []
            for prev_combination in _get_combinations(x, degree - 1):
                for feature_idx in range(len(x)):
                    combination = prev_combination + [feature_idx]
                    combinations.append(combination)
            return combinations

        def _calculate_new_features(features, combination):
            new_feature = 1
            for feature_idx in combination:
                new_feature *= features[feature_idx]
            return new_feature

        combinations = _get_combinations(range(n_features), self.degree)
        n_output_features = len(combinations)
        X_transformed = np.empty((n_samples, n_output_features))

        for i, combination in enumerate(combinations):
            X_transformed[:, i] = np.apply_along_axis(_calculate_new_features, 1, X, combination)

        if self.include_bias:
            X_transformed = np.insert(X_transformed, 0, 1, axis=1)

        return X_transformed

    def fit(self, X, y):
        """Fits the model to the data."""
        # Transform input to polynomial features
        X_transformed = self._transform_input(X)
        # Fit the model
        self.weights = np.linalg.inv(X_transformed.T.dot(X_transformed)).dot(X_transformed.T).dot(y)

    def predict(self, X):
        """Makes predictions using the polynomial regression model."""
        # Transform input to polynomial features
        X_transformed = self._transform_input(X)
        # Make predictions
        y_pred = X_transformed.dot(self.weights)
        return y_pred


# Usage
if __name__ == "__main__":
    # Generate some nonlinear data
    np.random.seed(0)
    X = np.random.rand(100, 1) * 4 - 2  # Features between -2 and 2
    y = 3 * X**2 + 2 * X + 1 + np.random.randn(100, 1) * 0.5  # Quadratic relationship with some noise

    # Reshape y for our PolynomialRegression class
    y = y.reshape(-1, 1)

    # Create a PolynomialRegression model
    poly_reg = PolynomialRegression(degree=2)
    poly_reg.fit(X, y)
    y_pred = poly_reg.predict(X)

    # For plotting we will sort the points by X axis
    sorted_indices = X.flatten().argsort()
    X_sorted = X.flatten()[sorted_indices].reshape(-1, 1)
    y_sorted = y_pred.flatten()[sorted_indices]

    # Plot
    import matplotlib.pyplot as plt
    plt.scatter(X, y, color='blue')
    plt.plot(X_sorted, y_sorted, color='red')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Polynomial Regression')
    plt.show()
