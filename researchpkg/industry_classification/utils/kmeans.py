
from sklearn.cluster import KMeans
from kneed import KneeLocator  # Install with: pip install kneed
from tqdm import tqdm
import numpy as np


def run_kmeans(embeddings:np.ndarray, k:int)-> KMeans:
    """
    Run kmeans clustering on the embeddings
    :param embeddings: The embeddings to cluster
    :param k: The number of clusters to create
    """
    kmeans = KMeans(n_clusters=k, random_state=0,
                    verbose=True,
                    init="random",
                    max_iter=1000,
                    algorithm='elkan',
                    ).fit(embeddings)
    return kmeans


def optimize_kmeans(embeddings:np.ndarray, max_k:int=16 ,default_k=4, verbose=True)-> KMeans:
    """
    Optimize the number of clusters for kmeans
    :param embeddings: The embeddings to cluster
    :param max_k: The maximum number of clusters to try
    :param default_k: The default number of clusters to use if the elbow is not found
    :return: The kmeans object with the optimal number of clusters
    """

    kmeans_list =[run_kmeans(embeddings, k) for k in tqdm(range(1, max_k+1), "Optimizing kmeans")]   
    inertia = [kmeans.inertia_ for kmeans in kmeans_list]

    kl = KneeLocator(range(1, max_k+1), inertia, curve="convex", direction="decreasing")

    
    if verbose:
        elbow = kl.elbow
        if elbow:
            print(f"Optimal number of clusters: {elbow}")
        else:
            print.write(f"Using default number of clusters: {default_k}")
            elbow = default_k+1
    else:
        elbow = kl.elbow or default_k+1

    return kmeans_list[elbow-1]
