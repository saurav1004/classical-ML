import numpy as np
from scipy.sparse import csgraph
from sklearn.neighbors import kneighbors_graph

class LabelPropagation:
    def __init__(self, max_iter=1000, tol=1e-3, n_neighbors=7):
        self.max_iter = max_iter
        self.tol = tol
        self.n_neighbors = n_neighbors
        self.labels_ = None

    def fit(self, X, y):
        # Create the graph
        n_samples = len(y)
        self.labels_ = np.copy(y)
        unlabeled_indices = y == -1
        graph = kneighbors_graph(X, self.n_neighbors, mode='connectivity', include_self=True)
        graph = graph.toarray()

        # Iterate until convergence or maximum iterations
        for _ in range(self.max_iter):
            prev_labels = np.copy(self.labels_)
            # Propagate labels
            self.labels_ = np.dot(graph, self.labels_)
            # Keep known labels fixed
            self.labels_[~unlabeled_indices] = y[~unlabeled_indices]
            # Check convergence
            if np.abs(prev_labels - self.labels_).sum() < self.tol:
                break

    def predict(self):
        return self.labels_

# Example usage:
# X would be the feature matrix, y would be the labels with -1 for unlabeled instances
# model = LabelPropagation()
# model.fit(X, y)
# labels = model.predict()
