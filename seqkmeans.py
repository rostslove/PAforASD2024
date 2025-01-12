import numpy as np

class SeqKMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

def fit(self, X):
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    
    for _ in range(self.ma_iter):
        
        rng = np.random.default_rng()
        self.centroids = X[rng.choice(X.shape[0], self.n_clusters, replace=False)]
            
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.n_clusters):
            mask = labels == k
            if np.any(mask):
                new_centroids[k] = X[mask].mean(axis=0)
            else:
                new_centroids[k] = X[np.random.choice(len(X))]
        
        if np.allclose(self.centroids, new_centroids):
            break
            
        self.centroids = new_centroids