#!/usr/bin/env python3
"""
Visualize song embedding clusters with style + 2 artists labels.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter
import plotly.express as px
import json
import warnings
warnings.filterwarnings('ignore')

from umap import UMAP


def load_data():
    embeddings = np.loadtxt('vectors_without_metadata_v5.tsv', delimiter='\t')
    metadata = pd.read_csv('clustered_songs.csv')
    return embeddings, metadata


def get_curated_labels():
    """Style description • 2 representative artists"""
    return {
        0: "Hip-Hop & Electronic Remix • 2Pac, Eminem",
        1: "Latin Jazz & Tango • Oscar Peterson, Tito Puente",
        2: "R&B & Blues Rock • Ike & Tina Turner, B.B. King",
        3: "Eclectic Rock & Funk • Motörhead, Aretha Franklin",
        4: "Classic Country & Folk • Elvis Presley, Charley Pride",
        5: "Pop Ballads & Soft Rock • Elton John, Johnny Mathis",
        6: "Tropical & Mambo • Celia Cruz, Pérez Prado",
        7: "Classic Rock & Power Pop • Led Zeppelin, The Rolling Stones",
        8: "Modern Pop & Rock Hits • Taylor Swift, Kelly Clarkson",
        9: "Traditional Jazz & Vocal • Billie Holiday, Louis Armstrong",
        10: "Art Rock & New Wave • Astor Piazzolla, Sting",
        11: "60s Pop & Country • Elvis Presley, The Beatles",
        12: "Early Jazz & Dixieland • Sidney Bechet, Fats Waller",
        13: "Vintage Swing & Big Band • Frank Sinatra, Bing Crosby",
        14: "Cool Jazz & Bebop • Stan Getz, Charlie Parker",
        15: "Bossa Nova & Jazz Vocals • Miles Davis, Bill Evans",
        16: "Heavy Metal & Hard Rock • Metallica, Slayer",
        17: "Early Rock & Roll & Soul • Elvis Presley, Otis Redding",
        18: "Chicago & Delta Blues • Muddy Waters, Howlin' Wolf",
        19: "Bluegrass & Americana • Stanley Brothers, Johnny Cash",
        20: "Vintage Pop & Latin Crooners • Flor Silvestre, Pedro Infante",
        21: "Jazz Fusion & Bolero • Herbie Hancock, Olga Guillot",
        22: "Jump Blues & Swing • Louis Jordan, Big Joe Turner",
        23: "80s Synth & Arena Rock • Depeche Mode, Judas Priest",
        24: "Classic Pop Crooners • Frank Sinatra, Nat King Cole",
    }


def reduce_dimensions(embeddings):
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    reducer = UMAP(n_components=3, n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42)
    return reducer.fit_transform(embeddings_scaled)


def create_visualization(reduced, metadata, labels):
    df = metadata.copy()
    df['x'], df['y'], df['z'] = reduced[:, 0], reduced[:, 1], reduced[:, 2]
    df['style'] = df['cluster'].map(labels)

    # Sort by cluster for consistent ordering
    df = df.sort_values('cluster')
    unique_styles = [labels[i] for i in sorted(labels.keys())]

    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='style',
        category_orders={'style': unique_styles},
        hover_name='title',
        hover_data={'channel': True, 'view_count': ':,.0f', 'x': False, 'y': False, 'z': False},
        title='<b>Song Embedding Space</b> — 36,364 Songs by Musical Style',
        color_discrete_sequence=px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    )

    fig.update_traces(marker=dict(size=3, opacity=0.7))

    # Create dropdown menu for isolating clusters
    buttons = [
        dict(
            label="Show All",
            method="update",
            args=[{"visible": [True] * len(unique_styles)}]
        )
    ]

    # Add button for each cluster to isolate it
    for i, style in enumerate(unique_styles):
        visibility = [False] * len(unique_styles)
        visibility[i] = True
        buttons.append(dict(
            label=style[:40] + "..." if len(style) > 40 else style,
            method="update",
            args=[{"visible": visibility}]
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
        ),
        legend_title='<b>Musical Style</b><br><sup>Double-click to isolate</sup>',
        legend=dict(
            font=dict(size=10),
            yanchor="top", y=0.99,
            xanchor="left", x=1.02,
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                active=0,
                x=0.0,
                y=1.15,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
                bgcolor="white",
                bordercolor="#ccc",
                font=dict(size=10),
                showactive=True,
            )
        ],
        annotations=[
            dict(
                text="<b>Select cluster:</b>",
                x=0, y=1.18,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=12)
            )
        ],
        height=950,
        margin=dict(l=0, r=350, t=100, b=0)
    )

    fig.write_html('clusters_3d_final.html')
    print("Saved to clusters_3d_final.html")


def main():
    print("Loading data...")
    embeddings, metadata = load_data()
    labels = get_curated_labels()

    print("Reducing dimensions with UMAP...")
    reduced = reduce_dimensions(embeddings)

    # Save clustered songs with UMAP coordinates
    metadata['umap_x'] = reduced[:, 0]
    metadata['umap_y'] = reduced[:, 1]
    metadata['umap_z'] = reduced[:, 2]
    metadata.to_csv('clustered_songs_with_coords.csv', index=False)
    print("Saved clustered_songs_with_coords.csv")

    # Save docs/data.json for the web app
    data_json = []
    for _, row in metadata.iterrows():
        data_json.append({
            'x': round(row['umap_x'], 2),
            'y': round(row['umap_y'], 2),
            'z': round(row['umap_z'], 2),
            'cluster': int(row['cluster']),
            'title': row['title'],
            'channel': row['channel'],
            'video_id': row['youtube_id'],
        })
    with open('docs/data.json', 'w') as f:
        json.dump(data_json, f)
    print(f"Saved docs/data.json ({len(data_json)} songs)")

    print("Creating visualization...")
    create_visualization(reduced, metadata, labels)


if __name__ == "__main__":
    main()
