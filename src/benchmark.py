import numpy as np
import time
from sklearn.datasets import make_blobs
from seqkmeans import SeqKMeans
from parallelkmeans import ParallelKMeans
import matplotlib.pyplot as plt
import os
from multiprocessing import cpu_count
from tqdm import tqdm

def run_benchmark(X, n_clusters, max_iter=300):
    # Pre-calculate max cores and jobs range
    max_cores = min(cpu_count(), 16)
    n_jobs_range = [2**i for i in range(1, int(np.log2(max_cores))+1)]
    
    # Initialize models upfront
    kmeans_seq = SeqKMeans(n_clusters=n_clusters, max_iter=max_iter)
    kmeans_pars = [ParallelKMeans(n_clusters=n_clusters, max_iter=max_iter, n_jobs=n) 
                   for n in n_jobs_range]

    # Sequential version
    start = time.time()
    kmeans_seq.fit(X)
    seq_time = time.time() - start

    # Parallel versions
    times_parallel = []
    for kmeans_par in kmeans_pars:
        start = time.time()
        kmeans_par.fit(X)
        times_parallel.append(time.time() - start)

    return seq_time, times_parallel, n_jobs_range

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    n_samples = 100000
    n_features = 10
    n_clusters = 8
    with tqdm(total=1, desc="Generating data") as pbar:
        X, _ = make_blobs(n_samples=n_samples, n_features=n_features, 
                         centers=n_clusters, random_state=42)
        pbar.update(1)
    print("Dataset generated successfully")

    os.makedirs('../benchmarks', exist_ok=True)

    print("\nStarting benchmark...")
    with tqdm(total=len(range(1)), desc="Sequential K-means") as pbar:
        seq_time, parallel_times, n_jobs_range = run_benchmark(X, n_clusters)
        pbar.update(1)

    speedups = np.array(seq_time) / np.array(parallel_times)

    # Create visualization
    with tqdm(total=1, desc="Creating visualization") as pbar:
        plt.figure(figsize=(10, 6))
        plt.plot(n_jobs_range, speedups, marker='o')
        plt.title('K-means Clustering Speedup')
        plt.xlabel('Number of Processes')
        plt.ylabel('Speedup')
        plt.grid(True)
        plt.savefig('../benchmarks/speedup.png')
        plt.close()
        pbar.update(1)

    print(f"\nResults:")
    print(f"Sequential time: {seq_time:.2f} seconds")
    for jobs, p_time in zip(n_jobs_range, parallel_times):
        print(f"Parallel time ({jobs} processes): {p_time:.2f} seconds")
    print(f"\nSpeedups: {[f'{s:.2f}x' for s in speedups]}")