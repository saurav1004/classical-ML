import numpy as np

class KMeans:
    def __init__(self, K=3, max_iters=100, random_state=42):
        self.K = K
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = []

    def initialize_centroids(self, X):
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.K]]
        return centroids

    def closest_centroid(self, X, centroids):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.K, X.shape[1]))
        for k in range(self.K):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def predict(self, X):
        self.centroids = self.initialize_centroids(X)
        
        for _ in range(self.max_iters):
            old_centroids = self.centroids
            labels = self.closest_centroid(X, old_centroids)
            self.centroids = self.compute_centroids(X, labels)
            
            if np.all(old_centroids == self.centroids):
                break
        return labels

# Example usage:
# X would be predefined data
# model = KMeans(K=3, max_iters=100)
# labels = model.predict(X)
