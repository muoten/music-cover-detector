#!/usr/bin/env python3
"""
Visualize song embedding clusters in 2D and 3D using UMAP/t-SNE.
"""

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP, fall back to t-SNE if not available
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not installed, using t-SNE instead. Install with: pip install umap-learn")


def load_data():
    """Load embeddings and clustered metadata."""
    print("Loading data...")

    embeddings = np.loadtxt('vectors_without_metadata_v4.tsv', delimiter='\t')

    # Try to load clustered data first, otherwise load original
    try:
        metadata = pd.read_csv('clustered_songs.csv')
        print("  Loaded pre-clustered data")
    except FileNotFoundError:
        metadata = pd.read_csv('metadata_clean_v4.tsv', sep='\t', nrows=len(embeddings))
        print("  Loaded original metadata (no clusters)")

    print(f"  {len(embeddings)} songs loaded")
    return embeddings, metadata


def reduce_dimensions(embeddings, n_components=3, method='umap'):
    """Reduce embedding dimensions using UMAP or t-SNE."""
    print(f"\nReducing dimensions to {n_components}D using {method.upper()}...")

    # Normalize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    if method == 'umap' and HAS_UMAP:
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=30,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        reduced = reducer.fit_transform(embeddings_scaled)
    else:
        # Use t-SNE
        perplexity = min(30, len(embeddings) - 1)
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            n_iter=1000,
            learning_rate='auto',
            init='pca'
        )
        reduced = reducer.fit_transform(embeddings_scaled)

    print(f"  Reduced shape: {reduced.shape}")
    return reduced


def create_3d_visualization(reduced, metadata, output_file='clusters_3d.html'):
    """Create interactive 3D scatter plot."""
    print(f"\nCreating 3D visualization...")

    # Prepare data
    df = metadata.copy()
    df['x'] = reduced[:, 0]
    df['y'] = reduced[:, 1]
    df['z'] = reduced[:, 2]

    # Create hover text
    df['hover_text'] = df.apply(
        lambda row: f"<b>{row['title']}</b><br>"
                    f"Channel: {row['channel']}<br>"
                    f"Views: {row['view_count']:,.0f}<br>"
                    f"Cluster: {row.get('cluster', 'N/A')}",
        axis=1
    )

    # Color by cluster
    if 'cluster' in df.columns:
        df['cluster_str'] = df['cluster'].astype(str)

        fig = px.scatter_3d(
            df,
            x='x', y='y', z='z',
            color='cluster_str',
            hover_name='title',
            hover_data={
                'channel': True,
                'view_count': ':,.0f',
                'cluster': True,
                'x': False, 'y': False, 'z': False,
                'cluster_str': False
            },
            title='Song Embeddings - 3D Cluster Visualization',
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
    else:
        fig = px.scatter_3d(
            df,
            x='x', y='y', z='z',
            hover_name='title',
            title='Song Embeddings - 3D Visualization'
        )

    # Update layout
    fig.update_traces(marker=dict(size=3, opacity=0.7))
    fig.update_layout(
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3',
        ),
        legend_title='Cluster',
        height=800,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    # Save to HTML
    fig.write_html(output_file)
    print(f"  Saved to {output_file}")

    return fig


def create_2d_visualization(reduced_2d, metadata, output_file='clusters_2d.html'):
    """Create interactive 2D scatter plot."""
    print(f"\nCreating 2D visualization...")

    df = metadata.copy()
    df['x'] = reduced_2d[:, 0]
    df['y'] = reduced_2d[:, 1]

    if 'cluster' in df.columns:
        df['cluster_str'] = df['cluster'].astype(str)

        fig = px.scatter(
            df,
            x='x', y='y',
            color='cluster_str',
            hover_name='title',
            hover_data={
                'channel': True,
                'view_count': ':,.0f',
                'cluster': True,
                'x': False, 'y': False,
                'cluster_str': False
            },
            title='Song Embeddings - 2D Cluster Visualization',
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
    else:
        fig = px.scatter(
            df,
            x='x', y='y',
            hover_name='title',
            title='Song Embeddings - 2D Visualization'
        )

    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        legend_title='Cluster',
        height=700
    )

    fig.write_html(output_file)
    print(f"  Saved to {output_file}")

    return fig


def create_cluster_summary_chart(metadata, output_file='cluster_summary.html'):
    """Create a summary visualization of cluster characteristics."""
    print("\nCreating cluster summary chart...")

    if 'cluster' not in metadata.columns:
        print("  No cluster data available")
        return None

    # Aggregate by cluster
    cluster_stats = metadata.groupby('cluster').agg({
        'title': 'count',
        'view_count': 'mean',
        'like_count': 'mean',
        'duration': 'mean'
    }).reset_index()

    cluster_stats.columns = ['cluster', 'song_count', 'avg_views', 'avg_likes', 'avg_duration']
    cluster_stats['avg_duration_min'] = cluster_stats['avg_duration'] / 60

    # Get top artist per cluster
    top_artists = []
    for cluster_id in cluster_stats['cluster']:
        cluster_data = metadata[metadata['cluster'] == cluster_id]
        top_channel = cluster_data['channel'].value_counts().head(1)
        if len(top_channel) > 0:
            top_artists.append(f"{top_channel.index[0]} ({top_channel.values[0]})")
        else:
            top_artists.append("N/A")
    cluster_stats['top_artist'] = top_artists

    # Create bubble chart
    fig = px.scatter(
        cluster_stats,
        x='avg_duration_min',
        y='avg_views',
        size='song_count',
        color='cluster',
        hover_name='cluster',
        hover_data={
            'song_count': True,
            'avg_views': ':,.0f',
            'avg_duration_min': ':.1f',
            'top_artist': True,
            'cluster': False
        },
        title='Cluster Characteristics: Duration vs Views (bubble size = song count)',
        labels={
            'avg_duration_min': 'Average Duration (min)',
            'avg_views': 'Average Views'
        },
        color_continuous_scale='Viridis'
    )

    fig.update_layout(height=600)
    fig.write_html(output_file)
    print(f"  Saved to {output_file}")

    return fig


def main():
    # Load data
    embeddings, metadata = load_data()

    # Choose method
    method = 'umap' if HAS_UMAP else 'tsne'

    # Reduce to 3D
    reduced_3d = reduce_dimensions(embeddings, n_components=3, method=method)

    # Reduce to 2D
    reduced_2d = reduce_dimensions(embeddings, n_components=2, method=method)

    # Create visualizations
    create_3d_visualization(reduced_3d, metadata, 'clusters_3d.html')
    create_2d_visualization(reduced_2d, metadata, 'clusters_2d.html')
    create_cluster_summary_chart(metadata, 'cluster_summary.html')

    # Save reduced coordinates
    metadata['umap_x'] = reduced_2d[:, 0]
    metadata['umap_y'] = reduced_2d[:, 1]
    metadata['umap_z'] = reduced_3d[:, 2]
    metadata.to_csv('clustered_songs_with_coords.csv', index=False)

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - clusters_3d.html  (interactive 3D scatter plot)")
    print("  - clusters_2d.html  (interactive 2D scatter plot)")
    print("  - cluster_summary.html  (cluster characteristics)")
    print("  - clustered_songs_with_coords.csv  (data with coordinates)")
    print("\nOpen the HTML files in a browser to explore interactively!")


if __name__ == "__main__":
    main()
