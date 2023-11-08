import numpy as np
from scipy.sparse import csgraph
from sklearn.neighbors import kneighbors_graph

class LabelSpreading:
    def __init__(self, max_iter=1000, tol=1e-3, n_neighbors=7, alpha=0.2):
        self.max_iter = max_iter
        self.tol = tol
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.labels_ = None

    def fit(self, X, y):
        # Create the graph
        n_samples = len(y)
        self.labels_ = np.copy(y)
        unlabeled_indices = y == -1
        graph = kneighbors_graph(X, self.n_neighbors, mode='connectivity', include_self=True)
        graph = csgraph.laplacian(graph, normed=True)
        graph = graph.toarray()

        # Initialize Y matrix
        Y = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            Y[i, y[i]] = 1

        # Iterate until convergence or maximum iterations
        for _ in range(self.max_iter):
            prev_labels = np.copy(self.labels_)
            # Propagate labels
            Y = self.alpha * np.dot(graph, Y) + (1 - self.alpha) * Y
            self.labels_ = np.argmax(Y, axis=1)
            # Keep known labels fixed
            self.labels_[~unlabeled_indices] = y[~unlabeled_indices]
            # Check convergence
            if np.abs(prev_labels - self.labels_).sum() < self.tol:
                break

    def predict(self):
        return self.labels_

# Example usage:
# X would be the feature matrix, y would be the labels with -1 for unlabeled instances
# model = LabelSpreading()
# model.fit(X, y)
# labels = model.predict()
