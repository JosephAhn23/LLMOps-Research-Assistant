"""
High-dimensional embedding analysis + UMAP visualization.
Covers: High-dimensional data analysis, dimensionality reduction
"""
from typing import Dict, List

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class EmbeddingAnalyzer:
    """
    Analyzes embedding space quality - clustering, dimensionality, drift.
    Covers: High-dimensional data analysis
    """

    def __init__(self, n_components: int = 2, n_clusters: int = 10):
        self.reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.n_clusters = n_clusters

    def analyze(self, embeddings: np.ndarray, metadata: List[Dict]) -> Dict:
        """Full analysis: UMAP reduction + clustering + quality metrics."""
        print(f"Analyzing {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}...")

        embeddings_scaled = self.scaler.fit_transform(embeddings)
        reduced = self.reducer.fit_transform(embeddings_scaled)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_scaled)
        sil_score = silhouette_score(embeddings_scaled, cluster_labels, sample_size=min(5000, len(embeddings)))

        cov = np.cov(embeddings.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 0]
        participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues**2)

        metrics = {
            "n_embeddings": len(embeddings),
            "embedding_dim": embeddings.shape[1],
            "estimated_intrinsic_dim": float(participation_ratio),
            "silhouette_score": float(sil_score),
            "n_clusters": self.n_clusters,
            "umap_shape": list(reduced.shape),
        }

        with mlflow.start_run(run_name="embedding-analysis", nested=True):
            mlflow.log_metrics(
                {
                    "silhouette_score": sil_score,
                    "intrinsic_dimensionality": participation_ratio,
                    "n_embeddings": len(embeddings),
                }
            )

        return {
            "metrics": metrics,
            "reduced_embeddings": reduced,
            "cluster_labels": cluster_labels,
        }

    def plot(
        self,
        reduced: np.ndarray,
        cluster_labels: np.ndarray,
        metadata: List[Dict],
        output_path: str = "outputs/embedding_space.png",
    ):
        """Visualize embedding clusters."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor("#0f0f0f")

        for ax in axes:
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")

        scatter = axes[0].scatter(reduced[:, 0], reduced[:, 1], c=cluster_labels, cmap="tab20", alpha=0.6, s=2)
        axes[0].set_title("Embedding Space (UMAP)", color="white", fontsize=13)
        axes[0].set_xlabel("UMAP-1", color="#aaa")
        axes[0].set_ylabel("UMAP-2", color="#aaa")
        plt.colorbar(scatter, ax=axes[0]).ax.yaxis.set_tick_params(color="white")

        sources = [m.get("source", "unknown") for m in metadata[: len(reduced)]]
        source_df = pd.Series(sources).value_counts().head(10)
        axes[1].barh(source_df.index, source_df.values, color="#4f8ef7")
        axes[1].set_title("Top Sources in Corpus", color="white", fontsize=13)
        axes[1].set_xlabel("Document Count", color="#aaa")
        axes[1].tick_params(colors="white")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"Saved embedding visualization to {output_path}")
        return output_path
