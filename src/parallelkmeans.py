import numpy as np
from multiprocessing import Pool


class ParallelKMeans:
    def __init__(self, n_clusters=8, max_iter=300, n_jobs=4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.centroids = None

    def _assign_labels(self, chunk):
        distances = np.linalg.norm(chunk[:, np.newaxis, :] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def fit(self, X):
        X = np.asarray(X)
        
        # Initialize centroids using array indexing
        rng = np.random.default_rng()
        self.centroids = X[rng.choice(X.shape[0], self.n_clusters, replace=False)]

        chunks = np.array_split(X, self.n_jobs)

        for _ in range(self.max_iter):
            # Parallel label assignment
            with Pool(self.n_jobs) as pool:
                labels = np.concatenate(pool.map(self._assign_labels, chunks))

            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = X[rng.choice(X.shape[0])]

            # Early stopping
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids