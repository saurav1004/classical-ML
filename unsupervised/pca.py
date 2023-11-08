import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        
        # Computing covariance matrix
        cov = np.cov(X.T)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Sorting eigenvectors by eigenvalues in descending order
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        # Storing first n eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        # Project data
        X = X - self.mean
        return np.dot(X, self.components.T)

# Example usage:
# X would be predefined data
# pca = PCA(n_components=2)
# pca.fit(X)
# X_projected = pca.transform(X)
