from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.estimators = []
        self.estimator_weights = []
        self.estimator_errors = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.full(n_samples, (1 / n_samples))

        for _ in range(self.n_estimators):
            clf = DecisionTreeClassifier(max_depth=1)
            clf.fit(X, y, sample_weight=sample_weights)
            stump_pred = clf.predict(X)
            
            # Errors
            err = sample_weights[(stump_pred != y)].sum()
            # Estimator weight
            alpha = 0.5 * np.log((1 - err) / err)
            # Update sample weights
            sample_weights *= np.exp(-alpha * y * stump_pred)
            sample_weights /= sample_weights.sum()

            # Store the estimator, its weight, and error
            self.estimators.append(clf)
            self.estimator_weights.append(alpha)
            self.estimator_errors.append(err)

    def predict(self, X):
        clf_preds = np.array([clf.predict(X) for clf in self.estimators])
        return np.sign(np.dot(self.estimator_weights, clf_preds))

# Example usage:
# X, y are the dataset
# y should be encoded as -1 and 1 for AdaBoost
# adaboost = AdaBoost(n_estimators=50)
# adaboost.fit(X, y)
# y_pred = adaboost.predict(X)
