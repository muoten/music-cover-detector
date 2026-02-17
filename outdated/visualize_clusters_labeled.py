#!/usr/bin/env python3
"""
Visualize song embedding clusters with meaningful labels.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

from umap import UMAP


def load_data():
    """Load embeddings and clustered metadata."""
    print("Loading data...")
    embeddings = np.loadtxt('vectors_without_metadata_v4.tsv', delimiter='\t')
    metadata = pd.read_csv('clustered_songs.csv')
    print(f"  {len(embeddings)} songs loaded")
    return embeddings, metadata


def generate_cluster_labels(metadata):
    """Generate meaningful labels for each cluster based on content analysis."""
    print("\nGenerating cluster labels...")

    cluster_labels = {}

    for cluster_id in sorted(metadata['cluster'].unique()):
        cluster_data = metadata[metadata['cluster'] == cluster_id]

        # Get top channels (excluding generic "Topic" channels)
        channels = cluster_data['channel'].dropna()
        channel_counts = Counter(channels)

        # Filter out generic channels
        top_artists = []
        for channel, count in channel_counts.most_common(10):
            if 'Topic' not in channel and 'Release' not in channel:
                # Clean up channel name
                artist = channel.replace(' - Topic', '').replace('Official', '').strip()
                top_artists.append(artist)
                if len(top_artists) >= 2:
                    break

        # Get common title words for genre hints
        titles = cluster_data['title'].dropna()
        words = []
        for title in titles:
            if isinstance(title, str):
                words.extend(title.lower().split())

        stopwords = {'the', 'a', 'an', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by',
                     'from', 'is', 'it', 'and', 'or', '-', '|', '(', ')', 'i', 'you', 'my',
                     'me', '(official', 'video)', 'audio)', 'version)', 'mix)', '(remastered)',
                     'remastered)', '(remastered', 'official', 'video', 'audio'}
        words = [w for w in words if w not in stopwords and len(w) > 2]
        word_counts = Counter(words)
        top_words = [w for w, c in word_counts.most_common(5)]

        # Calculate average views to determine mainstream vs niche
        avg_views = cluster_data['view_count'].mean()

        # Build label
        if top_artists:
            artist_str = ', '.join(top_artists[:2])
        else:
            artist_str = "Various"

        # Detect genre hints
        genre_hints = []
        if 'blues' in top_words:
            genre_hints.append('Blues')
        if 'jazz' in top_words or any(a in artist_str.lower() for a in ['coltrane', 'getz', 'baker', 'miles']):
            genre_hints.append('Jazz')
        if 'country' in top_words or any(a in artist_str.lower() for a in ['cash', 'haggard', 'nelson', 'owens']):
            genre_hints.append('Country')
        if 'rock' in top_words:
            genre_hints.append('Rock')
        if 'amor' in top_words or 'que' in top_words or 'mambo' in top_words:
            genre_hints.append('Latin')
        if 'rag' in top_words or 'swing' in top_words:
            genre_hints.append('Swing')

        # Build final label
        n_songs = len(cluster_data)

        if genre_hints:
            genre_str = '/'.join(genre_hints[:2])
            label = f"{cluster_id}: {genre_str} ({artist_str})"
        else:
            label = f"{cluster_id}: {artist_str}"

        # Add popularity indicator
        if avg_views > 5_000_000:
            label += " ★"

        cluster_labels[cluster_id] = label
        print(f"  Cluster {cluster_id}: {label}")

    return cluster_labels


def auto_label_clusters(metadata):
    """Auto-generate labels based on top artists and characteristics."""
    print("\nAuto-generating cluster labels from data...")

    labels = {}

    for cluster_id in sorted(metadata['cluster'].unique()):
        cluster_data = metadata[metadata['cluster'] == cluster_id]
        n_songs = len(cluster_data)
        avg_views = cluster_data['view_count'].mean()
        avg_duration = cluster_data['duration'].mean() / 60  # minutes

        # Get top 3 non-generic channels
        channels = cluster_data['channel'].dropna()
        top_channels = []
        for ch, cnt in Counter(channels).most_common(15):
            if 'Release - Topic' not in ch:
                clean_name = ch.replace(' - Topic', '').replace('Official', '').strip()
                if len(clean_name) > 0:
                    top_channels.append(clean_name)
                if len(top_channels) >= 3:
                    break

        # Build descriptive label
        if top_channels:
            artists = ', '.join(top_channels[:2])
        else:
            artists = 'Various'

        # Add size and popularity info
        size_label = f"{n_songs} songs"

        if avg_views > 10_000_000:
            pop = "Very Popular"
        elif avg_views > 1_000_000:
            pop = "Popular"
        elif avg_views > 100_000:
            pop = "Moderate"
        else:
            pop = "Niche"

        labels[cluster_id] = f"{artists} [{pop}]"

    return labels


def reduce_dimensions(embeddings, n_components=3):
    """Reduce dimensions using UMAP."""
    print(f"\nReducing to {n_components}D with UMAP...")

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    reducer = UMAP(
        n_components=n_components,
        n_neighbors=30,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    reduced = reducer.fit_transform(embeddings_scaled)
    print(f"  Done: {reduced.shape}")
    return reduced


def create_3d_visualization(reduced, metadata, cluster_labels, output_file='clusters_3d_labeled.html'):
    """Create 3D visualization with meaningful cluster labels."""
    print(f"\nCreating labeled 3D visualization...")

    df = metadata.copy()
    df['x'] = reduced[:, 0]
    df['y'] = reduced[:, 1]
    df['z'] = reduced[:, 2]

    # Map cluster numbers to labels
    df['cluster_label'] = df['cluster'].map(cluster_labels)

    # Sort by cluster for consistent legend order
    df = df.sort_values('cluster')

    fig = px.scatter_3d(
        df,
        x='x', y='y', z='z',
        color='cluster_label',
        hover_name='title',
        hover_data={
            'channel': True,
            'view_count': ':,.0f',
            'cluster': True,
            'x': False, 'y': False, 'z': False,
            'cluster_label': False
        },
        title='Song Embeddings - 3D Cluster Visualization (15,000 songs)',
        color_discrete_sequence=px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    )

    fig.update_traces(marker=dict(size=3, opacity=0.7))
    fig.update_layout(
        scene=dict(
            xaxis_title='',
            yaxis_title='',
            zaxis_title='',
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            zaxis=dict(showticklabels=False),
        ),
        legend_title='Cluster (Top Artists)',
        legend=dict(
            font=dict(size=10),
            itemsizing='constant'
        ),
        height=900,
        margin=dict(l=0, r=250, t=50, b=0)
    )

    fig.write_html(output_file)
    print(f"  Saved to {output_file}")
    return fig


def main():
    embeddings, metadata = load_data()

    # Generate meaningful labels
    cluster_labels = auto_label_clusters(metadata)

    # Reduce dimensions
    reduced_3d = reduce_dimensions(embeddings, n_components=3)

    # Create visualization
    create_3d_visualization(reduced_3d, metadata, cluster_labels, 'clusters_3d_labeled.html')

    print("\n" + "="*60)
    print("Done! Opening clusters_3d_labeled.html")
    print("="*60)


if __name__ == "__main__":
    main()
