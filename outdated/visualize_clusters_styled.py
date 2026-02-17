#!/usr/bin/env python3
"""
Visualize song embedding clusters with style-based meaningful labels.
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


def analyze_cluster_style(cluster_data):
    """Analyze a cluster to determine its musical style characteristics."""

    n_songs = len(cluster_data)
    avg_views = cluster_data['view_count'].mean()
    avg_duration = cluster_data['duration'].mean() / 60  # minutes

    # Get channels for artist analysis
    channels = cluster_data['channel'].dropna().tolist()
    channel_counts = Counter(channels)

    # Clean artist names
    artists = []
    for ch, cnt in channel_counts.most_common(20):
        if 'Release - Topic' not in ch:
            clean = ch.replace(' - Topic', '').replace('Official', '').replace('VEVO', '').strip()
            if clean:
                artists.append((clean, cnt))

    # Analyze title words for genre hints
    titles = cluster_data['title'].dropna()
    all_words = []
    for title in titles:
        if isinstance(title, str):
            all_words.extend(title.lower().split())

    word_counts = Counter(all_words)

    # Define genre indicators
    style_indicators = {
        'blues': ['blues', 'boogie', 'shuffle'],
        'jazz': ['jazz', 'swing', 'bop', 'improvisation'],
        'country': ['country', 'honky', 'cowboy', 'nashville'],
        'rock': ['rock', 'guitar', 'metal', 'punk'],
        'soul_rnb': ['soul', 'motown', 'funk', 'groove'],
        'latin': ['amor', 'corazón', 'mambo', 'salsa', 'bolero', 'cumbia', 'que', 'los', 'tango'],
        'pop': ['pop', 'dance', 'club'],
        'folk': ['folk', 'acoustic', 'traditional'],
        'classical': ['symphony', 'concerto', 'orchestra', 'classical'],
        'electronic': ['remix', 'mix', 'dj', 'house', 'techno', 'electronic'],
    }

    # Known artist-to-genre mappings
    artist_genres = {
        # Jazz
        'stan getz': 'jazz', 'chet baker': 'jazz', 'duke ellington': 'jazz',
        'billie holiday': 'jazz', 'ella fitzgerald': 'jazz', 'louis armstrong': 'jazz',
        'miles davis': 'jazz', 'john coltrane': 'jazz', 'charlie parker': 'jazz',
        'sonny rollins': 'jazz', 'herbie hancock': 'jazz', 'wayne shorter': 'jazz',
        'dexter gordon': 'jazz', 'coleman hawkins': 'jazz', 'sidney bechet': 'jazz',
        'benny goodman': 'jazz', 'count basie': 'jazz', 'dizzy gillespie': 'jazz',

        # Blues
        'muddy waters': 'blues', 'b.b. king': 'blues', 'howlin wolf': 'blues',
        'lightnin hopkins': 'blues', 'memphis slim': 'blues', 'john lee hooker': 'blues',
        'lowell fulson': 'blues', 'james cotton': 'blues', 'buddy guy': 'blues',

        # Country
        'johnny cash': 'country', 'willie nelson': 'country', 'merle haggard': 'country',
        'hank williams': 'country', 'buck owens': 'country', 'george jones': 'country',
        'loretta lynn': 'country', 'dolly parton': 'country', 'patsy cline': 'country',

        # Classic Pop/Crooners
        'frank sinatra': 'crooner', 'bing crosby': 'crooner', 'dean martin': 'crooner',
        'nat king cole': 'crooner', 'perry como': 'crooner', 'tony bennett': 'crooner',
        'andy williams': 'crooner',

        # Rock
        'elvis presley': 'rock_n_roll', 'chuck berry': 'rock_n_roll', 'little richard': 'rock_n_roll',
        'bob dylan': 'folk_rock', 'the who': 'rock', 'led zeppelin': 'rock',
        'judas priest': 'metal', 'slayer': 'metal', 'iron maiden': 'metal',
        'eric clapton': 'blues_rock',

        # Soul/R&B
        'the four tops': 'soul', 'the temptations': 'soul', 'marvin gaye': 'soul',
        'aretha franklin': 'soul', 'james brown': 'soul', 'otis redding': 'soul',

        # Modern Pop
        'taylor swift': 'modern_pop', 'britney spears': 'modern_pop',
        'madonna': 'pop', 'michael jackson': 'pop',

        # Latin
        'carlos vives': 'latin', 'luis miguel': 'latin', 'mercedes sosa': 'latin_folk',

        # Easy Listening
        'james last': 'easy_listening', 'mantovani': 'easy_listening',
        'olivia newton-john': 'soft_pop',
    }

    # Score genres based on artists
    genre_scores = Counter()
    top_artist_names = [a[0].lower() for a in artists[:10]]

    for artist_name in top_artist_names:
        for known_artist, genre in artist_genres.items():
            if known_artist in artist_name or artist_name in known_artist:
                genre_scores[genre] += 3

    # Score based on title words
    for word, count in word_counts.most_common(100):
        for style, indicators in style_indicators.items():
            if word in indicators:
                genre_scores[style] += count

    # Determine era based on views and known artists
    era = None
    if avg_views < 100_000:
        era = 'vintage'
    elif avg_views < 500_000:
        era = 'classic'
    elif avg_views > 5_000_000:
        era = 'mainstream'

    # Check for specific era indicators in artists
    vintage_artists = {'bing crosby', 'duke ellington', 'fats waller', 'louis armstrong'}
    classic_artists = {'frank sinatra', 'elvis presley', 'johnny cash'}

    for artist_name in top_artist_names:
        if any(v in artist_name for v in vintage_artists):
            era = 'vintage'
            break
        if any(c in artist_name for c in classic_artists):
            era = 'classic'

    return {
        'genre_scores': genre_scores,
        'top_artists': artists[:5],
        'avg_views': avg_views,
        'avg_duration': avg_duration,
        'n_songs': n_songs,
        'era': era,
        'top_words': word_counts.most_common(20)
    }


def generate_style_label(cluster_id, analysis):
    """Generate a meaningful style-based label for a cluster."""

    genre_scores = analysis['genre_scores']
    top_artists = analysis['top_artists']
    era = analysis['era']
    avg_views = analysis['avg_views']
    avg_duration = analysis['avg_duration']

    # Get top genres
    top_genres = genre_scores.most_common(3)

    # Build style description
    style_parts = []

    # Primary genre
    if top_genres:
        primary_genre = top_genres[0][0]
        genre_map = {
            'jazz': 'Jazz',
            'blues': 'Blues',
            'country': 'Country',
            'rock': 'Rock',
            'rock_n_roll': "Rock 'n' Roll",
            'folk_rock': 'Folk Rock',
            'blues_rock': 'Blues Rock',
            'metal': 'Heavy Metal',
            'soul': 'Soul/Motown',
            'soul_rnb': 'R&B/Soul',
            'latin': 'Latin',
            'latin_folk': 'Latin Folk',
            'pop': 'Pop',
            'modern_pop': 'Modern Pop',
            'soft_pop': 'Soft Pop',
            'crooner': 'Classic Vocal',
            'easy_listening': 'Easy Listening',
            'electronic': 'Electronic/Dance',
            'folk': 'Folk',
            'classical': 'Classical',
        }
        style_parts.append(genre_map.get(primary_genre, primary_genre.title()))

        # Add secondary if strong
        if len(top_genres) > 1 and top_genres[1][1] > top_genres[0][1] * 0.5:
            secondary = genre_map.get(top_genres[1][0], top_genres[1][0].title())
            if secondary != style_parts[0]:
                style_parts[0] = f"{style_parts[0]}/{secondary}"

    # Add era descriptor
    if era == 'vintage':
        style_parts.insert(0, 'Vintage')
    elif era == 'classic':
        style_parts.insert(0, 'Classic')
    elif era == 'mainstream':
        style_parts.append('Hits')

    # Add characteristic based on duration
    if avg_duration > 4.5:
        style_parts.append('(Extended)')
    elif avg_duration < 2.5:
        style_parts.append('(Short Form)')

    # Build label
    if style_parts:
        style = ' '.join(style_parts)
    else:
        style = 'Mixed'

    # Add representative artist
    if top_artists:
        rep_artist = top_artists[0][0]
        # Shorten if too long
        if len(rep_artist) > 20:
            rep_artist = rep_artist[:17] + '...'
        label = f"{style} • {rep_artist}"
    else:
        label = style

    return label


def create_cluster_labels(metadata):
    """Create style-based labels for all clusters."""
    print("\nAnalyzing cluster styles...")

    labels = {}
    analyses = {}

    for cluster_id in sorted(metadata['cluster'].unique()):
        cluster_data = metadata[metadata['cluster'] == cluster_id]
        analysis = analyze_cluster_style(cluster_data)
        analyses[cluster_id] = analysis

        label = generate_style_label(cluster_id, analysis)
        labels[cluster_id] = label

        # Debug output
        top_genres = analysis['genre_scores'].most_common(3)
        genre_str = ', '.join([f"{g}:{s}" for g, s in top_genres]) if top_genres else 'none'
        print(f"  {cluster_id:2d}: {label:45s} | genres: {genre_str}")

    return labels, analyses


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


def create_3d_visualization(reduced, metadata, cluster_labels, output_file='clusters_3d_styled.html'):
    """Create 3D visualization with style-based labels."""
    print(f"\nCreating styled 3D visualization...")

    df = metadata.copy()
    df['x'] = reduced[:, 0]
    df['y'] = reduced[:, 1]
    df['z'] = reduced[:, 2]

    # Map cluster numbers to style labels
    df['style'] = df['cluster'].map(cluster_labels)

    # Sort for consistent legend
    df = df.sort_values('cluster')

    fig = px.scatter_3d(
        df,
        x='x', y='y', z='z',
        color='style',
        hover_name='title',
        hover_data={
            'channel': True,
            'view_count': ':,.0f',
            'style': True,
            'x': False, 'y': False, 'z': False,
        },
        title='Song Embedding Space — 15,000 Songs Clustered by Musical Style',
        color_discrete_sequence=px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    )

    fig.update_traces(marker=dict(size=3, opacity=0.7))
    fig.update_layout(
        scene=dict(
            xaxis_title='',
            yaxis_title='',
            zaxis_title='',
            xaxis=dict(showticklabels=False, showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            yaxis=dict(showticklabels=False, showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            zaxis=dict(showticklabels=False, showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            bgcolor='rgba(250,250,250,1)'
        ),
        legend_title='Musical Style',
        legend=dict(
            font=dict(size=11),
            itemsizing='constant',
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        height=900,
        margin=dict(l=0, r=300, t=50, b=0),
        paper_bgcolor='white',
    )

    fig.write_html(output_file)
    print(f"  Saved to {output_file}")
    return fig


def main():
    embeddings, metadata = load_data()

    # Generate style-based labels
    cluster_labels, analyses = create_cluster_labels(metadata)

    # Reduce dimensions
    reduced_3d = reduce_dimensions(embeddings, n_components=3)

    # Create visualization
    create_3d_visualization(reduced_3d, metadata, cluster_labels, 'clusters_3d_styled.html')

    print("\n" + "="*60)
    print("CLUSTER STYLE SUMMARY")
    print("="*60)
    for cid in sorted(cluster_labels.keys()):
        a = analyses[cid]
        print(f"\n{cluster_labels[cid]}")
        print(f"   {a['n_songs']} songs, avg {a['avg_duration']:.1f} min, {a['avg_views']:,.0f} views")
        if a['top_artists']:
            artists = ', '.join([x[0] for x in a['top_artists'][:3]])
            print(f"   Artists: {artists}")


if __name__ == "__main__":
    main()
