import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Function to compute PCA on feature maps


def compute_pca(feature_maps, n_components=3):
    feature_maps_numpy = feature_maps.numpy()
    pca = PCA(n_components=n_components)
    projection_result = pca.fit_transform(feature_maps_numpy)
    explained_variance_ratios = pca.explained_variance_ratio_
    return projection_result, explained_variance_ratios


# Function to compute t-SNE on feature maps
def compute_tsne(feature_maps, n_components=2, perplexity=30, n_iter=300):
    feature_maps_numpy = feature_maps.numpy()
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
    tsne_result = tsne.fit_transform(feature_maps_numpy)
    return tsne_result, None  # N


def compute_kmeans_and_plot(feature_maps, labels=None, n_clusters=None, method="PCA"):
    assert not (
        n_clusters is None and labels is None
    ), "You should either provide labels or n_clusters argument"

    n_clusters = labels.unique().shape[0] if labels is not None else n_clusters

    if method == "PCA":
        # Use pca
        projection_result, explained_variance_ratios = compute_pca(feature_maps)
    else:
        # Use tsne
        projection_result, explained_variance_ratios = compute_tsne(feature_maps)

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans_result = kmeans.fit_predict(projection_result)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))

    for cluster_id, color in zip(range(n_clusters), colors):
        plt.scatter(
            projection_result[kmeans_result == cluster_id, 0],
            projection_result[kmeans_result == cluster_id, 1],
            label=f"Cluster {cluster_id}",
            color=color,
        )
    print("Pca explained variance :", explained_variance_ratios)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Features Map clusters from kmeans")
    plt.legend()
    plt.show()

    if labels is not None:
        plt.figure(figsize=(10, 8))
        for label, color in zip(range(n_clusters), colors):
            plt.scatter(
                projection_result[labels == label, 0],
                projection_result[labels == label, 1],
                label=label,
                color=color,
            )
        print("Pca explained variance:", explained_variance_ratios)
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.title("Features Map True labels")
        plt.legend()
        plt.show()


def compute_kmeans_and_plot_3d(feature_maps, labels, n_clusters=8, method="PCA"):
    if method == "PCA":
        projection_result, explained_variance_ratios = compute_pca(
            feature_maps, n_components=3
        )
    else:
        projection_result, explained_variance_ratios = compute_tsne(
            feature_maps, n_components=3
        )

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans_result = kmeans.fit_predict(projection_result)

    print("Pca explained variance :", explained_variance_ratios)

    fig = px.scatter_3d(
        x=projection_result[:, 0],
        y=projection_result[:, 1],
        z=projection_result[:, 2],
        color=kmeans_result,
        labels={"color": "Cluster"},
        title="Features Map clusters",
    )
    fig.show()

    fig = px.scatter_3d(
        x=projection_result[:, 0],
        y=projection_result[:, 1],
        z=projection_result[:, 2],
        color=labels,
        labels={"color": "Label"},
        title="Features Map clusters",
    )

    fig.show()
