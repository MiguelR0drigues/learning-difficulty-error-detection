"""
Clustering Analysis Module (Optional)

This module provides clustering and visualization capabilities
to explore whether similar errors naturally group together.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """
    Apply K-means clustering to embeddings.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        n_clusters: Number of clusters to create
        
    Returns:
        Array of cluster labels for each sample
    """
    if len(embeddings) < n_clusters:
        n_clusters = max(2, len(embeddings) // 2)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    return labels


def reduce_dimensions(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Reduce embedding dimensions using t-SNE for visualization.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        n_components: Target dimensionality (default 2 for 2D plots)
        
    Returns:
        Array of shape (n_samples, n_components)
    """
    perplexity = min(30, len(embeddings) - 1)
    perplexity = max(5, perplexity)
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    
    return reduced


def visualize_clusters(
    embeddings: np.ndarray,
    labels: list[str],
    title: str = "Student Answer Clustering",
    save_path: str = None
) -> None:
    """
    Create 2D visualization of embeddings colored by error labels.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        labels: List of error type labels for each sample
        title: Plot title
        save_path: Optional path to save the figure
    """
    # Reduce to 2D for visualization
    reduced = reduce_dimensions(embeddings)
    
    # Create color mapping for error types
    unique_labels = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    for label in unique_labels:
        mask = [l == label for l in labels]
        points = reduced[mask]
        plt.scatter(
            points[:, 0],
            points[:, 1],
            c=[label_to_color[label]],
            label=label,
            alpha=0.7,
            s=100
        )
    
    plt.title(title, fontsize=14)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Error Type", loc="best")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to: {save_path}")
    
    plt.show()


def analyze_clusters(embeddings: np.ndarray, ground_truth_labels: list[str]) -> dict:
    """
    Analyze how well natural clusters align with ground truth error labels.
    
    Args:
        embeddings: Array of embedded student answers
        ground_truth_labels: List of actual error labels
        
    Returns:
        Dictionary with cluster analysis results
    """
    unique_labels = list(set(ground_truth_labels))
    n_clusters = len(unique_labels)
    
    # Perform clustering
    cluster_labels = cluster_embeddings(embeddings, n_clusters)
    
    # Analyze cluster composition
    cluster_composition = {}
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        cluster_gt_labels = [l for l, m in zip(ground_truth_labels, mask) if m]
        
        # Count label distribution in this cluster
        label_counts = {}
        for label in cluster_gt_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Find dominant label
        if label_counts:
            dominant_label = max(label_counts, key=label_counts.get)
            purity = label_counts[dominant_label] / len(cluster_gt_labels)
        else:
            dominant_label = "empty"
            purity = 0.0
        
        cluster_composition[f"cluster_{cluster_id}"] = {
            "size": int(sum(mask)),
            "label_distribution": label_counts,
            "dominant_label": dominant_label,
            "purity": round(purity, 2)
        }
    
    # Calculate overall purity
    total_correct = sum(
        c["purity"] * c["size"] for c in cluster_composition.values()
    )
    overall_purity = total_correct / len(ground_truth_labels) if ground_truth_labels else 0
    
    return {
        "n_clusters": n_clusters,
        "cluster_composition": cluster_composition,
        "overall_purity": round(overall_purity, 2)
    }
