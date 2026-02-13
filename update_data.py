#!/usr/bin/env python3
"""
Data update pipeline for the song-similarities web app.

Usage:
    fly sftp get /data/vectors.csv vectors.csv
    python update_data.py [--vectors vectors.csv]

Regenerates all web app data: clusters, UMAP projections, refined embeddings,
cover links, and verified covers.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Directory of this script (all paths relative to here)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(SCRIPT_DIR, 'docs')

# Global output prefix (set from --output-prefix arg)
OUTPUT_PREFIX = None


def prefixed_name(base_name):
    """Apply output prefix to a filename. E.g. 'data.json' -> 'data_discogs.json'."""
    if not OUTPUT_PREFIX:
        return base_name
    name, ext = os.path.splitext(base_name)
    return f'{name}_{OUTPUT_PREFIX}{ext}'

VIDEOS_TO_TEST_URL = (
    'https://raw.githubusercontent.com/muoten/yt-CoverHunter/main/data/videos_to_test.csv'
)

# Store metadata cache on persistent volume if available
_persistent_cache = '/app/data/metadata_cache.csv'
METADATA_CACHE_PATH = _persistent_cache if os.path.isdir('/app/data') else os.path.join(SCRIPT_DIR, 'metadata_cache.csv')
METADATA_COLUMNS = [
    'youtube_id', 'title', 'channel', 'uploader',
    'view_count', 'like_count', 'duration', 'upload_date', 'channel_id'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Regenerate all web app data from raw vectors CSV.'
    )
    parser.add_argument(
        '--vectors', default='vectors.csv',
        help='Path to vectors CSV downloaded from yt-CoverHunter (Fly.io)'
    )
    parser.add_argument(
        '--output-prefix', default=None,
        help='Prefix for output files (e.g., "discogs" -> data_discogs.json, cover_links_discogs.json)'
    )
    parser.add_argument(
        '--skip-metadata', action='store_true',
        help='Skip yt-dlp metadata fetching, use cache only (fast, for testing)'
    )
    parser.add_argument(
        '--skip-training', action='store_true',
        help='Skip embedding refiner training, reuse existing vectors_refined.tsv'
    )
    parser.add_argument(
        '--skip-umap', action='store_true',
        help='Skip UMAP projection, reuse existing coords and just regenerate JSONs'
    )
    return parser.parse_args()


# ============================================================================
# Step 0: Download ground truth
# ============================================================================

def download_ground_truth():
    """Download fresh videos_to_test.csv from GitHub."""
    dest = os.path.join(SCRIPT_DIR, 'videos_to_test.csv')
    print(f'Downloading videos_to_test.csv from GitHub...')
    try:
        urllib.request.urlretrieve(VIDEOS_TO_TEST_URL, dest)
        lines = sum(1 for _ in open(dest)) - 1
        print(f'  Downloaded {lines} entries')
    except Exception as e:
        if os.path.exists(dest):
            print(f'  Download failed ({e}), using existing file')
        else:
            print(f'  ERROR: Download failed and no local file exists: {e}')
            sys.exit(1)
    return dest


# ============================================================================
# Step 1: Parse vectors & fetch metadata
# ============================================================================

def parse_vectors_csv(vectors_path, removed_path=None):
    """Parse the raw vectors.csv into youtube_ids and embedding arrays.

    The CSV has columns: youtube_id, embeddings
    where embeddings is a string like '\"[ 0.123 -0.456 ... ]\"'
    """
    print(f'Parsing {vectors_path}...')

    # Load removed IDs to exclude
    removed_ids = set()
    if removed_path and os.path.exists(removed_path):
        with open(removed_path, 'r') as f:
            removed_ids = {line.strip() for line in f if line.strip()}
        if removed_ids:
            print(f'  Excluding {len(removed_ids)} removed videos')

    youtube_ids = []
    embeddings = []
    skipped = 0

    expected_dim = None

    with open(vectors_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        for row in reader:
            if len(row) < 2:
                skipped += 1
                continue
            vid = row[0].strip()
            # Extract YouTube ID from paths like /tmp/.../ABC123.wav
            if '/' in vid:
                vid = vid.rsplit('/', 1)[-1].replace('.wav', '')
            if vid in removed_ids:
                skipped += 1
                continue
            emb_str = row[1].strip().strip('"')
            # Parse "[ 0.123 -0.456 ... ]" -> list of floats
            emb_str = emb_str.strip('[]')
            try:
                values = [float(x) for x in emb_str.split()]
            except ValueError:
                skipped += 1
                continue

            # Auto-detect dimension from first valid row
            if expected_dim is None:
                expected_dim = len(values)
                print(f'  Detected embedding dimension: {expected_dim}')
            elif len(values) != expected_dim:
                skipped += 1
                continue

            arr = np.array(values, dtype=np.float64)
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                skipped += 1
                continue

            youtube_ids.append(vid)
            embeddings.append(arr)

    embeddings = np.array(embeddings)
    print(f'  Parsed {len(youtube_ids)} valid embeddings ({skipped} skipped)')
    return youtube_ids, embeddings


def load_metadata_cache():
    """Load cached metadata from previous runs."""
    if not os.path.exists(METADATA_CACHE_PATH):
        return {}
    cache = {}
    df = pd.read_csv(METADATA_CACHE_PATH, sep='\t', dtype=str, keep_default_na=False)
    for _, row in df.iterrows():
        cache[row['youtube_id']] = {col: row.get(col, '') for col in METADATA_COLUMNS}
    print(f'  Loaded {len(cache)} cached metadata entries')
    return cache


def save_metadata_cache(cache):
    """Save metadata cache for future runs."""
    rows = list(cache.values())
    if not rows:
        return
    df = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    df.to_csv(METADATA_CACHE_PATH, sep='\t', index=False)
    print(f'  Saved {len(rows)} entries to metadata cache')


def fetch_metadata_noembed(video_id):
    """Fetch metadata for a single video using noembed API."""
    import urllib.request, json as _json
    url = f'https://noembed.com/embed?url=https://www.youtube.com/watch?v={video_id}'
    try:
        resp = _json.loads(urllib.request.urlopen(url, timeout=10).read())
        title = resp.get('title', '')
        author = resp.get('author_name', '')
        return {
            'youtube_id': video_id,
            'title': title,
            'channel': author,
            'uploader': author,
            'view_count': '0', 'like_count': '0', 'duration': '0',
            'upload_date': '', 'channel_id': '',
        }
    except Exception:
        pass
    return {
        'youtube_id': video_id,
        'title': '', 'channel': '', 'uploader': '',
        'view_count': '0', 'like_count': '0', 'duration': '0',
        'upload_date': '', 'channel_id': '',
    }


def fetch_all_metadata(youtube_ids, skip_fetch=False):
    """Fetch metadata for all videos, using cache for known ones."""
    cache = load_metadata_cache()
    # Re-fetch entries with empty titles (failed previous fetches)
    new_ids = [vid for vid in youtube_ids if vid not in cache or not cache[vid].get('title')]

    if skip_fetch:
        print(f'  --skip-metadata: skipping yt-dlp fetch for {len(new_ids)} new IDs')
        for vid in new_ids:
            cache[vid] = {
                'youtube_id': vid,
                'title': '', 'channel': '', 'uploader': '',
                'view_count': '0', 'like_count': '0', 'duration': '0',
                'upload_date': '', 'channel_id': '',
            }
    elif new_ids:
        print(f'  Fetching metadata for {len(new_ids)} new videos (parallel)...')
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = {pool.submit(fetch_metadata_noembed, vid): vid for vid in new_ids}
            done = 0
            for future in as_completed(futures):
                vid = futures[future]
                cache[vid] = future.result()
                done += 1
                if done % 500 == 0:
                    print(f'  {done}/{len(new_ids)} fetched')
        print(f'  {done}/{len(new_ids)} fetched')

    save_metadata_cache(cache)

    # Build ordered metadata list matching youtube_ids order
    metadata_rows = []
    for vid in youtube_ids:
        row = cache.get(vid, {
            'youtube_id': vid,
            'title': '', 'channel': '', 'uploader': '',
            'view_count': '0', 'like_count': '0', 'duration': '0',
            'upload_date': '', 'channel_id': '',
        })
        metadata_rows.append(row)

    return metadata_rows


def write_step1_outputs(youtube_ids, embeddings, metadata_rows):
    """Write vectors_without_metadata.tsv and metadata_clean.tsv."""
    vectors_path = os.path.join(SCRIPT_DIR, 'vectors_without_metadata.tsv')
    metadata_path = os.path.join(SCRIPT_DIR, 'metadata_clean.tsv')

    np.savetxt(vectors_path, embeddings, delimiter='\t', fmt='%.6f')
    print(f'  Wrote {vectors_path} ({len(embeddings)} rows)')

    df = pd.DataFrame(metadata_rows, columns=METADATA_COLUMNS)
    # Convert numeric columns
    for col in ['view_count', 'like_count', 'duration']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df.to_csv(metadata_path, sep='\t', index=False)
    print(f'  Wrote {metadata_path} ({len(df)} rows)')

    return vectors_path, metadata_path


# ============================================================================
# Step 2: Cluster
# ============================================================================

def run_clustering(embeddings, n_clusters=25):
    """K-means clustering on embeddings."""
    from sklearn.preprocessing import normalize
    print(f'Running K-means clustering (k={n_clusters})...')
    embeddings = normalize(embeddings, norm='l2')
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_scaled)

    print(f'  Cluster sizes: min={np.bincount(labels).min()}, max={np.bincount(labels).max()}')
    return labels


def write_clustered_songs(metadata_path, labels):
    """Write clustered_songs.csv."""
    output_path = os.path.join(SCRIPT_DIR, 'clustered_songs.csv')
    metadata = pd.read_csv(metadata_path, sep='\t')
    metadata['cluster'] = labels
    metadata.to_csv(output_path, index=False)
    print(f'  Wrote {output_path}')
    return output_path


# ============================================================================
# Step 3: UMAP projection
# ============================================================================

def run_umap(embeddings):
    """UMAP 3D projection."""
    from umap import UMAP
    from sklearn.preprocessing import normalize
    print('Running UMAP 3D projection (n_neighbors=30, min_dist=0.1, metric=cosine)...')
    # L2 normalize to ensure consistent scale across old/new embeddings
    embeddings = normalize(embeddings, norm='l2')
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    reducer = UMAP(
        n_components=3, n_neighbors=30, min_dist=0.1,
        metric='cosine', random_state=42
    )
    reduced = reducer.fit_transform(embeddings_scaled)
    print(f'  UMAP output shape: {reduced.shape}')
    return reduced


def write_umap_outputs(clustered_path, reduced):
    """Write clustered_songs_with_coords.csv and docs/data.json."""
    metadata = pd.read_csv(clustered_path)
    metadata['umap_x'] = reduced[:, 0]
    metadata['umap_y'] = reduced[:, 1]
    metadata['umap_z'] = reduced[:, 2]

    coords_path = os.path.join(SCRIPT_DIR, 'clustered_songs_with_coords.csv')
    metadata.to_csv(coords_path, index=False)
    print(f'  Wrote {coords_path}')

    # docs/data.json
    data_json = []
    for _, row in metadata.iterrows():
        data_json.append({
            'x': round(float(row['umap_x']), 2),
            'y': round(float(row['umap_y']), 2),
            'z': round(float(row['umap_z']), 2),
            'cluster': int(row['cluster']),
            'title': row['title'] if pd.notna(row['title']) else 'Unknown',
            'channel': row['channel'] if pd.notna(row['channel']) else 'Unknown',
            'video_id': row['youtube_id'],
        })

    data_json_path = os.path.join(DOCS_DIR, prefixed_name('data.json'))
    with open(data_json_path, 'w') as f:
        json.dump(data_json, f)
    print(f'  Wrote {data_json_path} ({len(data_json)} songs)')

    return coords_path


# ============================================================================
# Step 4: Train embedding refiner
# ============================================================================

def run_training(vectors_path, metadata_path, ground_truth_path):
    """Run train_embedding_refiner.py as a subprocess with default args."""
    print('Training embedding refiner...')
    train_script = os.path.join(SCRIPT_DIR, 'train_embedding_refiner.py')
    output_model = os.path.join(SCRIPT_DIR, 'embedding_refiner.pt')
    output_embeddings = os.path.join(SCRIPT_DIR, 'vectors_refined.tsv')

    cmd = [
        sys.executable, train_script,
        '--embeddings', vectors_path,
        '--metadata', metadata_path,
        '--ground-truth', ground_truth_path,
        '--output-model', output_model,
        '--output-embeddings', output_embeddings,
    ]
    print(f'  Running: {" ".join(cmd)}')
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f'  WARNING: Training exited with code {result.returncode}')
        if not os.path.exists(output_embeddings):
            print('  ERROR: No refined embeddings produced. Cannot continue.')
            sys.exit(1)

    return output_embeddings


# ============================================================================
# Step 5: Find covers
# ============================================================================

def find_covers(vectors_refined_path, vectors_raw_path, metadata_path, coords_path):
    """Find top-5 neighbors per song using refined embeddings for ranking."""
    print('Finding covers (top-5 neighbors per song)...')

    vectors_refined = np.loadtxt(vectors_refined_path, delimiter='\t')
    vectors_raw = np.loadtxt(vectors_raw_path, delimiter='\t')
    metadata = pd.read_csv(metadata_path, sep='\t')
    clustered = pd.read_csv(coords_path)

    print(f'  {len(vectors_refined)} refined vectors, {len(vectors_raw)} raw vectors')

    BATCH_SIZE = 1000
    top_neighbors = {}

    for i in tqdm(range(0, len(vectors_refined), BATCH_SIZE), desc='  Finding neighbors'):
        batch_refined = vectors_refined[i:i + BATCH_SIZE]
        distances_refined = cosine_distances(batch_refined, vectors_refined)

        batch_raw = vectors_raw[i:i + BATCH_SIZE]
        distances_raw = cosine_distances(batch_raw, vectors_raw)

        for j, row_refined in enumerate(distances_refined):
            song_idx = i + j
            sorted_indices = np.argsort(row_refined)
            neighbors = []
            for idx in sorted_indices:
                if idx != song_idx:
                    neighbors.append((int(idx), float(distances_raw[j][idx])))
                    if len(neighbors) >= 5:
                        break
            top_neighbors[song_idx] = neighbors

    # covers_data.json
    covers_data = []
    for _, row in clustered.iterrows():
        covers_data.append({
            'x': round(float(row['umap_x']), 2),
            'y': round(float(row['umap_y']), 2),
            'z': round(float(row['umap_z']), 2),
            'cluster': int(row['cluster']),
            'title': row['title'] if pd.notna(row['title']) else 'Unknown',
            'channel': row['channel'] if pd.notna(row['channel']) else 'Unknown',
            'video_id': row['youtube_id'],
        })

    covers_data_path = os.path.join(DOCS_DIR, prefixed_name('covers_data.json'))
    with open(covers_data_path, 'w') as f:
        json.dump(covers_data, f)
    print(f'  Wrote {covers_data_path} ({len(covers_data)} songs)')

    # cover_links.json
    cover_links = {}
    for song_idx, neighbors in top_neighbors.items():
        vid = metadata.iloc[song_idx]['youtube_id']
        cover_links[vid] = [
            {'id': metadata.iloc[other_idx]['youtube_id'], 'dist': round(dist, 3)}
            for other_idx, dist in neighbors
        ]

    cover_links_path = os.path.join(DOCS_DIR, prefixed_name('cover_links.json'))
    with open(cover_links_path, 'w') as f:
        json.dump(cover_links, f)
    print(f'  Wrote {cover_links_path} ({len(cover_links)} songs)')


# ============================================================================
# Step 6: Generate verified_covers.json
# ============================================================================

def generate_verified_covers(ground_truth_path, metadata_path):
    """Build bidirectional map of verified cover pairs from videos_to_test.csv."""
    print('Generating verified_covers.json...')

    gt = pd.read_csv(ground_truth_path)
    gt['youtube_id'] = gt['youtube_url'].str.extract(r'v=([A-Za-z0-9_-]{11})')

    metadata = pd.read_csv(metadata_path, sep='\t')
    dataset_ids = set(metadata['youtube_id'].tolist())

    # Group by clique, keep only IDs present in our dataset
    clique_to_ids = {}
    for _, row in gt.iterrows():
        clique = row['clique']
        vid = row['youtube_id']
        if pd.isna(vid) or vid not in dataset_ids:
            continue
        clique_to_ids.setdefault(clique, []).append(vid)

    # Build bidirectional map
    verified = {}
    for clique, ids in clique_to_ids.items():
        if len(ids) < 2:
            continue
        for vid in ids:
            covers = [other for other in ids if other != vid]
            if covers:
                verified[vid] = covers

    output_path = os.path.join(DOCS_DIR, prefixed_name('verified_covers.json'))
    with open(output_path, 'w') as f:
        json.dump(verified, f)
    print(f'  Wrote {output_path} ({len(verified)} songs with verified covers)')

    return verified


# ============================================================================
# Step 7: Summary
# ============================================================================

def print_summary(metadata_path, n_new_songs, labels, verified):
    """Print pipeline summary."""
    metadata = pd.read_csv(metadata_path, sep='\t')
    n_total = len(metadata)
    n_clusters = len(set(labels))
    n_cover_pairs = sum(len(v) for v in verified.values()) // 2

    print('\n' + '=' * 60)
    print('PIPELINE SUMMARY')
    print('=' * 60)
    print(f'  Total songs:          {n_total:,}')
    print(f'  New songs added:      {n_new_songs:,}')
    print(f'  Clusters:             {n_clusters}')
    print(f'  Verified cover pairs: {n_cover_pairs}')
    print()

    print('Output files:')
    for fname in ['data.json', 'covers_data.json', 'cover_links.json', 'verified_covers.json']:
        fpath = os.path.join(DOCS_DIR, prefixed_name(fname))
        actual_name = prefixed_name(fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f'  docs/{actual_name}: {size_kb:,.1f} KB')
        else:
            print(f'  docs/{actual_name}: MISSING')

    print()
    print('Next steps:')
    print('  git add docs/ && git commit -m "data: update to latest snapshot" && git push')


# ============================================================================
# Main
# ============================================================================

def main():
    global OUTPUT_PREFIX
    args = parse_args()
    OUTPUT_PREFIX = args.output_prefix

    # Ensure docs/ exists
    os.makedirs(DOCS_DIR, exist_ok=True)

    # Count existing songs for "new songs" stat
    old_metadata_path = os.path.join(SCRIPT_DIR, 'metadata_clean.tsv')
    old_count = 0
    if os.path.exists(old_metadata_path):
        old_count = sum(1 for _ in open(old_metadata_path)) - 1  # minus header

    # Step 0: Download ground truth
    print('\n' + '=' * 60)
    print('STEP 0: Download ground truth')
    print('=' * 60)
    ground_truth_path = download_ground_truth()

    # Step 1: Parse vectors & fetch metadata
    print('\n' + '=' * 60)
    print('STEP 1: Parse vectors & fetch metadata')
    print('=' * 60)
    # Determine removed.txt path (next to vectors.csv or in /app/data)
    vectors_dir = os.path.dirname(os.path.abspath(args.vectors))
    removed_path = os.path.join(vectors_dir, 'removed.txt')
    youtube_ids, embeddings = parse_vectors_csv(args.vectors, removed_path=removed_path)
    metadata_rows = fetch_all_metadata(youtube_ids, skip_fetch=args.skip_metadata)
    vectors_path, metadata_path = write_step1_outputs(youtube_ids, embeddings, metadata_rows)
    n_new_songs = len(youtube_ids) - old_count if old_count > 0 else len(youtube_ids)

    # Step 2: Cluster
    print('\n' + '=' * 60)
    print('STEP 2: Cluster')
    print('=' * 60)
    labels = run_clustering(embeddings, n_clusters=25)
    clustered_path = write_clustered_songs(metadata_path, labels)

    # Step 3: UMAP projection
    print('\n' + '=' * 60)
    print('STEP 3: UMAP projection')
    print('=' * 60)
    if args.skip_umap:
        coords_path = os.path.join(SCRIPT_DIR, 'clustered_songs_with_coords.csv')
        if not os.path.exists(coords_path):
            print('  ERROR: --skip-umap but no existing clustered_songs_with_coords.csv')
            sys.exit(1)
        print('  Skipping UMAP, reusing existing coordinates')
        # Still regenerate data.json from existing coords
        clustered = pd.read_csv(coords_path)
        data_json = []
        for _, row in clustered.iterrows():
            data_json.append({
                'x': round(float(row['umap_x']), 2),
                'y': round(float(row['umap_y']), 2),
                'z': round(float(row['umap_z']), 2),
                'cluster': int(row['cluster']),
                'title': row['title'] if pd.notna(row['title']) else 'Unknown',
                'channel': row['channel'] if pd.notna(row['channel']) else 'Unknown',
                'video_id': row['youtube_id'],
            })
        data_json_path = os.path.join(DOCS_DIR, prefixed_name('data.json'))
        with open(data_json_path, 'w') as f:
            json.dump(data_json, f)
        print(f'  Wrote {data_json_path} ({len(data_json)} songs)')
    else:
        reduced = run_umap(embeddings)
        coords_path = write_umap_outputs(clustered_path, reduced)

    # Step 4: Train embedding refiner
    print('\n' + '=' * 60)
    print('STEP 4: Train embedding refiner')
    print('=' * 60)
    refined_path = os.path.join(SCRIPT_DIR, 'vectors_refined.tsv')
    if args.skip_training:
        if not os.path.exists(refined_path):
            print('  ERROR: --skip-training but no existing vectors_refined.tsv')
            sys.exit(1)
        print(f'  Skipping training, reusing {refined_path}')
    else:
        refined_path = run_training(vectors_path, metadata_path, ground_truth_path)

    # Step 5: Find covers
    print('\n' + '=' * 60)
    print('STEP 5: Find covers')
    print('=' * 60)
    find_covers(refined_path, vectors_path, metadata_path, coords_path)

    # Step 6: Generate verified_covers.json
    print('\n' + '=' * 60)
    print('STEP 6: Generate verified_covers.json')
    print('=' * 60)
    verified = generate_verified_covers(ground_truth_path, metadata_path)

    # Step 7: Summary
    print_summary(metadata_path, n_new_songs, labels, verified)


if __name__ == '__main__':
    main()
