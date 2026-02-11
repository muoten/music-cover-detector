#!/usr/bin/env python3
"""
Cluster analysis of song embeddings to find patterns in metadata.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Load embeddings and metadata."""
    print("Loading data...")

    # Load embeddings (no header, tab-separated)
    embeddings = np.loadtxt(
        'vectors_without_metadata_v5.tsv',
        delimiter='\t'
    )
    print(f"  Embeddings shape: {embeddings.shape}")

    # Load metadata (has header, tab-separated)
    metadata = pd.read_csv(
        'metadata_clean_v5.tsv',
        sep='\t',
        nrows=len(embeddings)  # Only load rows that have embeddings
    )
    print(f"  Metadata shape: {metadata.shape}")

    return embeddings, metadata


def find_optimal_clusters(embeddings, max_k=20):
    """Find optimal number of clusters using elbow method and silhouette score."""
    print("\nFinding optimal number of clusters...")

    # Normalize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    inertias = []
    silhouettes = []
    k_range = range(5, max_k + 1, 5)

    for k in k_range:
        print(f"  Testing k={k}...", end=" ")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_scaled)
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(embeddings_scaled, labels, sample_size=5000)
        silhouettes.append(sil)
        print(f"silhouette={sil:.3f}")

    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"\nBest k by silhouette score: {best_k}")

    return best_k, list(k_range), silhouettes


def perform_clustering(embeddings, n_clusters):
    """Perform K-means clustering."""
    print(f"\nPerforming K-means clustering with {n_clusters} clusters...")

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_scaled)

    return labels, kmeans, scaler


def analyze_cluster(metadata, cluster_id, cluster_mask):
    """Analyze a single cluster's metadata patterns."""
    cluster_data = metadata[cluster_mask]
    n_songs = len(cluster_data)

    analysis = {
        'cluster_id': cluster_id,
        'n_songs': n_songs,
        'pct_of_total': n_songs / len(metadata) * 100
    }

    # Top channels
    channels = cluster_data['channel'].dropna()
    if len(channels) > 0:
        channel_counts = Counter(channels)
        analysis['top_channels'] = channel_counts.most_common(5)

    # Top uploaders
    uploaders = cluster_data['uploader'].dropna()
    if len(uploaders) > 0:
        uploader_counts = Counter(uploaders)
        analysis['top_uploaders'] = uploader_counts.most_common(5)

    # Common words in titles
    titles = cluster_data['title'].dropna()
    if len(titles) > 0:
        words = []
        for title in titles:
            if isinstance(title, str):
                words.extend(title.lower().split())
        # Filter out common words
        stopwords = {'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by',
                     'from', 'is', 'it', 'and', 'or', '-', '|', '(', ')', 'i', 'you', 'my', 'me'}
        words = [w for w in words if w not in stopwords and len(w) > 2]
        word_counts = Counter(words)
        analysis['top_title_words'] = word_counts.most_common(10)

    # Average metrics
    analysis['avg_views'] = cluster_data['view_count'].mean()
    analysis['avg_likes'] = cluster_data['like_count'].mean()
    analysis['avg_duration'] = cluster_data['duration'].mean()

    # Sample titles
    sample_titles = cluster_data['title'].dropna().head(10).tolist()
    analysis['sample_titles'] = sample_titles

    return analysis


def print_cluster_analysis(analysis):
    """Pretty print cluster analysis."""
    print(f"\n{'='*60}")
    print(f"CLUSTER {analysis['cluster_id']}: {analysis['n_songs']} songs ({analysis['pct_of_total']:.1f}%)")
    print('='*60)

    if 'top_title_words' in analysis:
        print("\nTop words in titles:")
        for word, count in analysis['top_title_words']:
            print(f"  {word}: {count}")

    if 'top_channels' in analysis:
        print("\nTop channels:")
        for channel, count in analysis['top_channels'][:5]:
            print(f"  {channel}: {count}")

    print(f"\nAverage metrics:")
    print(f"  Views: {analysis['avg_views']:,.0f}")
    print(f"  Likes: {analysis['avg_likes']:,.0f}")
    print(f"  Duration: {analysis['avg_duration']:.0f}s ({analysis['avg_duration']/60:.1f}min)")

    print("\nSample titles:")
    for title in analysis['sample_titles'][:5]:
        print(f"  - {title}")


def main():
    # Load data
    embeddings, metadata = load_data()

    # Find optimal clusters
    best_k, k_range, silhouettes = find_optimal_clusters(embeddings, max_k=30)

    # Use 25 clusters to match the web UI structure
    n_clusters = 25

    # Perform clustering
    labels, kmeans, scaler = perform_clustering(embeddings, n_clusters)

    # Add cluster labels to metadata
    metadata['cluster'] = labels

    # Analyze each cluster
    print("\n" + "="*60)
    print("CLUSTER ANALYSIS RESULTS")
    print("="*60)

    cluster_analyses = []
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        analysis = analyze_cluster(metadata, cluster_id, cluster_mask)
        cluster_analyses.append(analysis)

    # Sort by size and print
    cluster_analyses.sort(key=lambda x: x['n_songs'], reverse=True)

    for analysis in cluster_analyses:
        print_cluster_analysis(analysis)

    # Save results
    metadata.to_csv('clustered_songs.csv', index=False)
    print(f"\n\nResults saved to 'clustered_songs.csv'")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total songs analyzed: {len(metadata)}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Cluster size range: {min(a['n_songs'] for a in cluster_analyses)} - {max(a['n_songs'] for a in cluster_analyses)}")

    return metadata, cluster_analyses


if __name__ == "__main__":
    metadata, analyses = main()
