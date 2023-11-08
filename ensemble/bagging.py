from sklearn.tree import DecisionTreeClassifier
import numpy as np

class Bagging:
    def __init__(self, n_estimators, max_samples, max_features):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.estimators = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.features_indices = []
        
        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, self.max_samples, replace=True)
            features_idx = np.random.choice(n_features, self.max_features, replace=True)
            self.features_indices.append(features_idx)
            X_sample = X[indices][:, features_idx]
            y_sample = y[indices]
            clf = DecisionTreeClassifier()
            clf.fit(X_sample, y_sample)
            self.estimators.append(clf)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.estimators)))
        for idx, clf in enumerate(self.estimators):
            predictions[:, idx] = clf.predict(X[:, self.features_indices[idx]])
        return np.squeeze(np.round(np.mean(predictions, axis=1)))

# Example usage:
# X, y are the dataset
# bagging = Bagging(n_estimators=100, max_samples=100, max_features=2)
# bagging.fit(X, y)
# y_pred = bagging.predict(X)
