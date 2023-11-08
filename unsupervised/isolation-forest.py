import numpy as np

class IsolationTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.root = None

    class Node:
        def __init__(self, left, right, feature, split):
            self.left = left
            self.right = right
            self.feature = feature
            self.split = split

    def fit(self, X, depth=0):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples <= 1:
            return None
        feature = np.random.randint(n_features)
        split = np.random.uniform(X[:, feature].min(), X[:, feature].max())
        left_idx = X[:, feature] < split
        right_idx = ~left_idx
        left = self.fit(X[left_idx], depth+1)
        right = self.fit(X[right_idx], depth+1)
        self.root = self.Node(left, right, feature, split)
        return self.Node(left, right, feature, split)

    def path_length(self, x, node, depth):
        if node is None:
            return depth
        if x[node.feature] < node.split:
            return self.path_length(x, node.left, depth+1)
        return self.path_length(x, node.right, depth+1)

class IsolationForest:
    def __init__(self, n_trees=100, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X):
        self.trees = [IsolationTree(max_depth=self.max_depth).fit(X) for _ in range(self.n_trees)]

    def anomaly_score(self, X):
        path_lengths = np.array([tree.path_length(x, tree.root, 0) for x in X for tree in self.trees])
        path_lengths = path_lengths.reshape(self.n_trees, len(X)).T
        avg_path_lengths = path_lengths.mean(axis=1)
        return 2 ** (-avg_path_lengths / c(len(X)))

def c(n):
    if n > 2:
        return 2 * (np.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)
    elif n == 2:
        return 1
    else:
        return 0

# Example usage:
# X would be predefined data
# model = IsolationForest(n_trees=100, max_depth=10)
# model.fit(X)
# scores = model.anomaly_score(X)
# threshold = np.percentile(scores, 95)  # set threshold as desired
# anomalies = scores > threshold
