#!/usr/bin/env python3
"""Find potential covers using learned metric for ranking, raw embeddings for similarity."""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm
import json

# Load data
print("Loading refined vectors (for ranking)...")
vectors_refined = np.loadtxt('vectors_refined.tsv', delimiter='\t')
print(f"Loaded {len(vectors_refined)} refined vectors, dim={vectors_refined.shape[1]}")

print("Loading raw vectors (for similarity display)...")
vectors_raw = np.loadtxt('vectors_without_metadata_v5.tsv', delimiter='\t')
print(f"Loaded {len(vectors_raw)} raw vectors, dim={vectors_raw.shape[1]}")

print("Loading metadata...")
metadata = pd.read_csv('metadata_clean_v5.tsv', sep='\t')
print(f"Loaded {len(metadata)} metadata rows")

print("Loading clustered data with coordinates...")
clustered = pd.read_csv('clustered_songs_with_coords.csv')
print(f"Loaded {len(clustered)} clustered songs")

# Find top neighbors using refined embeddings (better ranking),
# but store distances from raw embeddings (meaningful similarity)
print("\nFinding covers using learned metric for ranking...")
BATCH_SIZE = 1000

top_neighbors = {}  # song_idx -> list of (other_idx, raw_distance)

for i in tqdm(range(0, len(vectors_refined), BATCH_SIZE)):
    # Use refined embeddings for ranking (finding nearest neighbors)
    batch_refined = vectors_refined[i:i+BATCH_SIZE]
    distances_refined = cosine_distances(batch_refined, vectors_refined)

    # Use raw embeddings for similarity display
    batch_raw = vectors_raw[i:i+BATCH_SIZE]
    distances_raw = cosine_distances(batch_raw, vectors_raw)

    for j, row_refined in enumerate(distances_refined):
        song_idx = i + j
        # Rank by refined distance, but store raw distance
        sorted_indices = np.argsort(row_refined)
        neighbors = []
        for idx in sorted_indices:
            if idx != song_idx:
                neighbors.append((idx, distances_raw[j][idx]))
                if len(neighbors) >= 10:
                    break
        top_neighbors[song_idx] = neighbors

# Create covers data.json — include ALL songs
print("\nCreating covers data (all songs)...")
covers_data = []
for _, row in tqdm(clustered.iterrows(), total=len(clustered)):
    covers_data.append({
        'x': round(row['umap_x'], 2),
        'y': round(row['umap_y'], 2),
        'z': round(row['umap_z'], 2),
        'cluster': int(row['cluster']),
        'title': row['title'] if pd.notna(row['title']) else 'Unknown',
        'channel': row['channel'] if pd.notna(row['channel']) else 'Unknown',
        'video_id': row['youtube_id']
    })

print(f"Created data for {len(covers_data)} songs")
with open('docs/covers_data.json', 'w') as f:
    json.dump(covers_data, f)
print("Saved docs/covers_data.json")

# Create cover links: ranked by learned metric, similarity from raw embeddings
print("\nCreating cover links...")
cover_links = {}
for song_idx, neighbors in top_neighbors.items():
    vid = metadata.iloc[song_idx]['youtube_id']
    cover_links[vid] = [
        {'id': metadata.iloc[other_idx]['youtube_id'], 'dist': round(float(dist), 3)}
        for other_idx, dist in neighbors
    ]

with open('docs/cover_links.json', 'w') as f:
    json.dump(cover_links, f)
print(f"Saved docs/cover_links.json ({len(cover_links)} songs)")

print("\nDone!")
