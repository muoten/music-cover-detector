#!/usr/bin/env python3
"""
Flask API for VINet cover detection.

Serves the Music Cover Detector web app and API.
Accepts a YouTube URL, computes VINet embedding via iTunes preview,
and returns top similar songs from the database.

Usage:
    python api.py [--port 5000] [--host 0.0.0.0]
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request

import numpy as np
import torch
import yaml
from flask import Flask, jsonify, request, send_from_directory

# Add parent dir to path so model/utilities imports work
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import librosa
from model.utils import load_model

# Import functions from pipeline
from pipeline import (
    get_youtube_title,
    get_youtube_info,
    parse_artist_track,
    search_itunes,
    search_itunes_detailed,
    lookup_itunes_by_id,
    smart_search_itunes,
    download_preview,
    convert_to_wav,
    compute_cqt_features,
    compute_embedding,
)

# Static files directory
STATIC_DIR = os.path.join(SCRIPT_DIR, 'static')

app = Flask(__name__, static_folder=STATIC_DIR)

# ============================================================================
# Global state (loaded on startup)
# ============================================================================

model = None
device = None
embeddings_matrix = None  # (N, 512) normalized
video_ids = []  # list of YouTube IDs in same order as embeddings
video_metadata = {}  # youtube_id -> {title, artist, track}
clique_map = {}  # youtube_id -> clique_id
embedding_hashes = {}  # youtube_id -> hash of embedding
track_ids = {}  # youtube_id -> iTunes trackId
db_stats = {}  # database statistics


def compute_embedding_hash(embedding):
    """Compute a short hash of an embedding vector."""
    # Convert to bytes and hash
    emb_bytes = embedding.astype(np.float32).tobytes()
    return hashlib.md5(emb_bytes).hexdigest()[:8]


def load_clique_map():
    """Load clique IDs from videos_to_test.csv."""
    global clique_map

    # Try multiple locations
    csv_paths = [
        os.path.join(SCRIPT_DIR, '..', 'videos_to_test.csv'),
        os.path.join(SCRIPT_DIR, 'videos_to_test.csv'),
        '/code/videos_to_test.csv',
    ]

    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            print(f"Loading clique map from {csv_path}...")
            with open(csv_path, 'r') as f:
                header = f.readline()  # skip header
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        clique_id = parts[0]
                        url = parts[1]
                        # Extract video ID from URL
                        if 'v=' in url:
                            vid = url.split('v=')[1].split('&')[0]
                            clique_map[vid] = clique_id
            print(f"Loaded {len(clique_map)} clique mappings")
            discogs_songs = sum(1 for vid in video_ids if vid in clique_map and clique_map[vid].startswith('C'))
            db_stats['discogs_songs'] = discogs_songs
            print(f"Songs from Discogs: {discogs_songs}/{len(video_ids)}")
            return

    print("No clique map found")


def load_database():
    """Load the CoverHunter embeddings database."""
    global embeddings_matrix, video_ids, video_metadata, embedding_hashes

    # Load CoverHunter embeddings from vectors.csv (128-dim)
    persistent_path = '/app/data/vectors.csv'
    if os.path.exists(persistent_path):
        vectors_path = persistent_path
    else:
        vectors_path = os.path.join(SCRIPT_DIR, '..', 'vectors.csv')
        if not os.path.exists(vectors_path):
            vectors_path = os.path.join(SCRIPT_DIR, 'vectors.csv')

    embeddings = []
    video_ids = []

    print(f"Loading embeddings from {vectors_path}...")
    with open(vectors_path, 'r') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) != 2:
                continue
            vid_or_path = parts[0]
            # Extract YouTube ID from path like /tmp/.../ABC123.wav
            if '/' in vid_or_path:
                vid = os.path.basename(vid_or_path).replace('.wav', '')
            else:
                vid = vid_or_path
            vec_str = parts[1].strip('"').strip('[]')
            try:
                vec = np.array([float(x) for x in vec_str.split()])
                if len(vec) == 128:  # CoverHunter is 128-dim
                    video_ids.append(vid)
                    embeddings.append(vec)
            except:
                continue

    embeddings_matrix = np.array(embeddings, dtype=np.float32)

    # L2 normalize
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embeddings_matrix = embeddings_matrix / norms

    # Compute hashes for each embedding
    print("Computing embedding hashes...")
    for i, vid in enumerate(video_ids):
        embedding_hashes[vid] = compute_embedding_hash(embeddings_matrix[i])

    # Compute collision stats
    hash_to_vids = {}
    for vid, h in embedding_hashes.items():
        if h not in hash_to_vids:
            hash_to_vids[h] = []
        hash_to_vids[h].append(vid)

    # Count songs with duplicate hashes (same embedding, different song)
    songs_with_collisions = sum(len(vids) for vids in hash_to_vids.values() if len(vids) > 1)
    unique_hashes = len(hash_to_vids)
    total_songs = len(video_ids)

    db_stats['total_songs'] = total_songs
    db_stats['unique_hashes'] = unique_hashes
    db_stats['songs_with_collisions'] = songs_with_collisions
    db_stats['collision_ratio'] = round(songs_with_collisions / total_songs, 4) if total_songs > 0 else 0

    print(f"Loaded {len(video_ids)} embeddings")
    print(f"Hash collisions: {songs_with_collisions}/{total_songs} ({db_stats['collision_ratio']*100:.1f}%)")

    # Load metadata if available
    metadata_path = os.path.join(SCRIPT_DIR, 'song_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            video_metadata = json.load(f)
        print(f"Loaded metadata for {len(video_metadata)} songs")

    # Load trackId mappings from track_ids.json - check persistent storage first
    global track_ids
    persistent_track_ids = '/app/data/track_ids.json'
    if os.path.exists(persistent_track_ids):
        track_ids_path = persistent_track_ids
    else:
        track_ids_path = os.path.join(SCRIPT_DIR, 'track_ids.json')
    progress_path = os.path.join(SCRIPT_DIR, 'progress.json')

    if os.path.exists(track_ids_path):
        with open(track_ids_path, 'r') as f:
            track_ids = json.load(f)
        print(f"Loaded {len(track_ids)} trackId mappings from track_ids.json")
    elif os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        track_ids = progress.get('track_ids', {})
        print(f"Loaded {len(track_ids)} trackId mappings from progress.json")

        # Compute trackId collision stats
        trackid_to_vids = {}
        for vid, tid in track_ids.items():
            if tid:
                if tid not in trackid_to_vids:
                    trackid_to_vids[tid] = []
                trackid_to_vids[tid].append(vid)

        vids_with_shared_trackid = sum(len(vids) for vids in trackid_to_vids.values() if len(vids) > 1)
        db_stats['unique_track_ids'] = len(trackid_to_vids)
        db_stats['videos_sharing_track_id'] = vids_with_shared_trackid
        print(f"Videos sharing same iTunes trackId: {vids_with_shared_trackid}")

    # Compute Precision@1
    compute_precision_at_1()


def compute_precision_at_1():
    """Compute Precision@1 using cover_map.json."""
    global db_stats

    import time
    start_time = time.time()

    # Try to find cover_map.json
    cover_map_paths = [
        os.path.join(SCRIPT_DIR, 'cover_map.json'),
        os.path.join(SCRIPT_DIR, '..', 'cover_map.json'),
        '/app/cover_map.json',
    ]

    cover_map = None
    for path in cover_map_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                cover_map = json.load(f)
            break

    if not cover_map:
        print("No cover_map.json found, skipping P@1 computation")
        return

    # Build video_id to index mapping
    vid_to_idx = {vid: i for i, vid in enumerate(video_ids)}

    # Find evaluable songs (have covers in the embedding set)
    eval_songs = []
    for vid in video_ids:
        if vid not in cover_map:
            continue
        covers_in_set = [c for c in cover_map[vid] if c in vid_to_idx]
        if covers_in_set:
            eval_songs.append((vid, covers_in_set))

    if not eval_songs:
        print("No evaluable songs for P@1")
        return

    # Compute P@1
    hits = 0
    for vid, covers in eval_songs:
        query_idx = vid_to_idx[vid]
        cover_idxs = set(vid_to_idx[c] for c in covers)

        # Cosine similarity (already normalized)
        sims = embeddings_matrix[query_idx] @ embeddings_matrix.T
        sims[query_idx] = -2  # Exclude self

        nn_idx = int(np.argmax(sims))
        if nn_idx in cover_idxs:
            hits += 1

    elapsed = time.time() - start_time
    n = len(eval_songs)
    p_at_1 = hits / n
    se = np.sqrt(p_at_1 * (1 - p_at_1) / n)
    db_stats['precision_at_1'] = round(p_at_1, 4)
    db_stats['precision_at_1_se'] = round(se, 4)
    db_stats['precision_at_1_hits'] = hits
    db_stats['precision_at_1_time'] = round(elapsed, 2)
    db_stats['evaluable_songs'] = n
    db_stats['model_name'] = 'CoverHunter'
    print(f"Precision@1: {hits}/{n} = {p_at_1:.2%} \u00b1 {se:.2%} ({elapsed:.2f}s)")


def init_model():
    """Initialize the VINet model."""
    global model, device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config_path = os.path.join(SCRIPT_DIR, 'checkpoint', 'config.yaml')
    checkpoint_path = os.path.join(SCRIPT_DIR, 'checkpoint', 'model_checkpoint.pth')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set checkpoint path in config
    config['MODEL']['CHECKPOINT_PATH'] = checkpoint_path

    model = load_model(config, device)
    print("Model loaded")


def find_similar(query_embedding, top_k=5):
    """Find top-k most similar songs to query embedding."""
    # Normalize query
    query = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    query = query.astype(np.float32)

    # Compute cosine similarities
    similarities = embeddings_matrix @ query

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        vid = video_ids[idx]
        sim = float(similarities[idx])

        # Fetch title from YouTube oEmbed
        info = get_youtube_info(vid)
        title = info.get('title', '') if info else ''
        channel = info.get('channel', '') if info else ''

        # Clean up YouTube channel name
        yt_artist = channel
        if yt_artist.endswith(' - Topic'):
            yt_artist = yt_artist[:-8]

        # Get iTunes metadata - prefer lookup by stored trackId, fallback to search
        itunes_artist, itunes_track, preview_url = '', '', ''
        stored_track_id = track_ids.get(vid)
        if stored_track_id:
            # Use stored trackId for accurate lookup
            itunes_data = lookup_itunes_by_id(stored_track_id)
            if itunes_data:
                itunes_artist = itunes_data.get('artistName', '')
                itunes_track = itunes_data.get('trackName', '')
                preview_url = itunes_data.get('previewUrl', '')
        if not itunes_artist and title:
            # Fallback to search by title
            parsed_artist, parsed_track = parse_artist_track(title)
            search_term = f"{parsed_artist} {parsed_track}" if parsed_artist else title
            itunes_results = search_itunes_detailed(search_term, limit=1)
            if itunes_results:
                itunes_artist = itunes_results[0].get('artistName', '')
                itunes_track = itunes_results[0].get('trackName', '')
                preview_url = itunes_results[0].get('previewUrl', '')

        # For display: prefer iTunes, fallback to parsed title
        _, parsed_track = parse_artist_track(title) if title else (None, None)

        result = {
            'youtube_id': vid,
            'similarity': round(sim, 4),
            'youtube_url': f'https://www.youtube.com/watch?v={vid}',
            'title': title or vid,
            'yt_channel': yt_artist,
            'artist': itunes_artist or yt_artist or '',
            'itunes_artist': itunes_artist,
            'itunes_track': itunes_track,
            'track': itunes_track or parsed_track or title or vid,
            'clique_id': clique_map.get(vid, ''),
            'hash': embedding_hashes.get(vid, ''),
            'track_id': track_ids.get(vid),
            'preview_url': preview_url,
        }
        results.append(result)

    # Mark results that share the same trackId (same iTunes preview)
    track_id_counts = {}
    for r in results:
        tid = r.get('track_id')
        if tid:
            track_id_counts[tid] = track_id_counts.get(tid, 0) + 1

    for r in results:
        tid = r.get('track_id')
        if tid and track_id_counts.get(tid, 0) > 1:
            r['same_preview'] = True

    return results


def extract_youtube_id(url_or_id):
    """Extract YouTube video ID from URL or return as-is if already an ID."""
    # If it's already just an ID (11 chars, alphanumeric + - _)
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id):
        return url_or_id

    # Try to extract from URL
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'v=([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    return None


# ============================================================================
# API Routes
# ============================================================================

@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files (JSON data, etc.)."""
    return send_from_directory(STATIC_DIR, filename)


@app.route('/api/search', methods=['POST'])
def api_search():
    """Search for similar songs given a YouTube URL."""
    data = request.get_json() or {}
    url = data.get('url', '').strip()
    top_k = min(int(data.get('top_k', 20)), 100)

    if not url:
        return jsonify({'error': 'Missing url parameter'}), 400

    # Extract YouTube ID
    youtube_id = extract_youtube_id(url)
    if not youtube_id:
        return jsonify({'error': 'Invalid YouTube URL'}), 400

    # Get YouTube info for response
    info = get_youtube_info(youtube_id)
    title = info.get('title', '') if info else ''
    yt_channel = info.get('channel', '') if info else ''
    if yt_channel.endswith(' - Topic'):
        yt_channel = yt_channel[:-8]

    # Check if video is already in database - use precomputed embedding
    if youtube_id in video_ids:
        idx = video_ids.index(youtube_id)
        embedding = embeddings_matrix[idx]
        query_track_id = track_ids.get(youtube_id)
        query_hash = embedding_hashes.get(youtube_id, '')
        # Get iTunes info from a quick search for display
        itunes_artist, itunes_track, preview_url = '', '', ''
        if title:
            parsed_artist, parsed_track = parse_artist_track(title)
            search_term = f"{parsed_artist} {parsed_track}" if parsed_artist else title
            itunes_results = search_itunes_detailed(search_term, limit=1)
            if itunes_results:
                itunes_artist = itunes_results[0].get('artistName', '')
                itunes_track = itunes_results[0].get('trackName', '')
                preview_url = itunes_results[0].get('previewUrl', '')
        in_database = True
    else:
        # Video not in database - CoverHunter model not available for on-the-fly computation
        return jsonify({
            'error': 'Video not in database. Only pre-indexed videos can be searched.',
            'title': title,
            'youtube_id': youtube_id,
            'database_size': len(video_ids),
        }), 404

    # Find similar songs (request extra in case we need to filter out the query itself)
    similar = find_similar(embedding, top_k + 5)

    # Filter out the query video itself from results
    similar = [r for r in similar if r['youtube_id'] != youtube_id][:top_k]

    # Check if query trackId matches any result trackId (same preview warning)
    query_matches_result = any(
        r.get('track_id') == query_track_id
        for r in similar
        if query_track_id and r.get('track_id')
    )

    return jsonify({
        'query': {
            'youtube_id': youtube_id,
            'youtube_url': f'https://www.youtube.com/watch?v={youtube_id}',
            'title': title,
            'yt_channel': yt_channel,
            'artist': itunes_artist,
            'track': itunes_track,
            'track_id': query_track_id,
            'hash': query_hash,
            'preview_url': preview_url,
            'clique_id': clique_map.get(youtube_id, ''),
            'same_preview_in_db': query_matches_result,
            'in_database': in_database,
        },
        'results': similar,
        'stats': db_stats,
    })


@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'embeddings_count': len(video_ids),
        'device': str(device),
        'stats': db_stats,
    })


@app.route('/api/reload', methods=['POST'])
def reload_database():
    """Reload the embeddings database (for live updates during reprocessing)."""
    try:
        load_database()
        return jsonify({
            'status': 'ok',
            'embeddings_count': len(video_ids),
            'stats': db_stats,
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


# Search history (persisted to file)
HISTORY_FILE = '/app/data/search_history.json' if os.path.isdir('/app/data') else os.path.join(SCRIPT_DIR, 'search_history.json')

def load_search_history():
    """Load search history from file."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return []

def save_search_history(history):
    """Save search history to file."""
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history[:10], f)  # Keep max 10 items

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get search history."""
    return jsonify(load_search_history())

@app.route('/api/history', methods=['POST'])
def add_history():
    """Add to search history."""
    data = request.get_json() or {}
    history = load_search_history()

    # Remove duplicate if exists
    history = [h for h in history if h.get('youtube_id') != data.get('youtube_id')]

    # Add to front
    history.insert(0, {
        'youtube_id': data.get('youtube_id'),
        'artist': data.get('artist'),
        'track': data.get('track')
    })

    save_search_history(history[:10])
    return jsonify({'status': 'ok'})


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VINet Cover Detection API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    args = parser.parse_args()

    print("Initializing VINet Cover Detection API...")
    init_model()
    load_database()
    load_clique_map()

    print(f"\nStarting server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
