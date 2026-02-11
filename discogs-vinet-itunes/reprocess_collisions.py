#!/usr/bin/env python3
"""
Re-process collision groups using smart iTunes matching.
Uses YouTube channel name as artist fallback to find different iTunes previews.

RESUMABLE: Tracks processed videos in reprocess_progress.json so it can
resume after container restarts/redeploys.

Usage:
    python reprocess_collisions.py [--dry-run] [--limit N]
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.request

import numpy as np
import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from pipeline import (
    smart_search_itunes,
    download_preview,
    convert_to_wav,
    compute_cqt_features,
    compute_embedding,
    format_embedding,
)
from model.utils import load_model


def compute_hash(embedding):
    """Compute hash of embedding."""
    emb_bytes = embedding.astype(np.float32).tobytes()
    return hashlib.md5(emb_bytes).hexdigest()[:8]


def load_vectors(csv_path):
    """Load embeddings from CSV."""
    embeddings = {}
    print(f"Loading embeddings from {csv_path}...")
    with open(csv_path, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) != 2:
                continue
            vid = parts[0]
            vec_str = parts[1].strip('"').strip('[]')
            try:
                vec = np.array([float(x) for x in vec_str.split()])
                if len(vec) == 512:
                    embeddings[vid] = vec
            except:
                continue
    print(f"Loaded {len(embeddings)} embeddings")
    return embeddings


def find_collisions(embeddings, processed_vids):
    """Find groups of songs with same hash, excluding already processed."""
    hash_to_vids = {}
    for vid, emb in embeddings.items():
        if vid in processed_vids:
            continue  # Skip already processed
        h = compute_hash(emb)
        if h not in hash_to_vids:
            hash_to_vids[h] = []
        hash_to_vids[h].append(vid)

    # Return only groups with more than 1 song
    collisions = {h: vids for h, vids in hash_to_vids.items() if len(vids) > 1}
    return collisions


def recompute_with_smart_match(vid, model, device, tmp_dir):
    """Re-fetch iTunes using smart matching and recompute embedding."""
    # Smart search uses channel name as artist
    preview_url, artist, track, track_id = smart_search_itunes(vid)
    if not preview_url:
        return None, None, "itunes_fail"

    m4a_path = os.path.join(tmp_dir, f'{vid}.m4a')
    wav_path = os.path.join(tmp_dir, f'{vid}.wav')

    try:
        if not download_preview(preview_url, m4a_path):
            return None, None, "download_fail"

        if not convert_to_wav(m4a_path, wav_path):
            return None, None, "convert_fail"

        cqt = compute_cqt_features(wav_path)
        if cqt is None:
            return None, None, "cqt_fail"

        embedding = compute_embedding(model, cqt, device)
        return embedding, track_id, None

    finally:
        for path in [m4a_path, wav_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass


def save_vectors(embeddings, csv_path):
    """Save embeddings to CSV atomically."""
    tmp_path = csv_path + '.tmp'
    print(f"Saving {len(embeddings)} embeddings to {csv_path}...")
    with open(tmp_path, 'w') as f:
        f.write("youtube_id,embeddings\n")
        for vid, emb in embeddings.items():
            emb_str = format_embedding(emb)
            f.write(f'{vid},{emb_str}\n')
    # Atomic rename
    os.replace(tmp_path, csv_path)
    print("Saved.")


def notify_api_reload(api_url='http://localhost:8080'):
    """Notify the API to reload database."""
    try:
        req = urllib.request.Request(f'{api_url}/api/reload', method='POST')
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            print(f"  API reloaded: {data.get('embeddings_count', '?')} songs, stats updated")
            return True
    except Exception as e:
        print(f"  (API reload failed: {e})")
        return False


def load_progress(progress_path):
    """Load progress from JSON file."""
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            return json.load(f)
    return {
        'processed_videos': [],
        'stats': {
            'groups_processed': 0,
            'songs_updated': 0,
            'collisions_fixed': 0,
            'collisions_remaining': 0,
            'failed': {},
        }
    }


def save_progress(progress_path, processed_videos, stats):
    """Save progress to JSON file atomically."""
    tmp_path = progress_path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump({
            'processed_videos': list(processed_videos),
            'stats': stats,
        }, f)
    os.replace(tmp_path, progress_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Do not modify files')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of collision groups (0=all)')
    parser.add_argument('--vectors', default=os.path.join(SCRIPT_DIR, 'vectors_vinet.csv'))
    parser.add_argument('--config', default=os.path.join(SCRIPT_DIR, 'checkpoint', 'config.yaml'))
    parser.add_argument('--tmp-dir', default=os.path.join(SCRIPT_DIR, 'tmp'))
    parser.add_argument('--log-interval', type=int, default=10, help='Log/save progress every N groups (default: 10)')
    parser.add_argument('--api-url', default='http://localhost:8080', help='API URL for reload notification')
    args = parser.parse_args()

    # Paths - use /app/data if it exists (persistent volume), otherwise use vectors dir
    data_dir = '/app/data' if os.path.isdir('/app/data') else os.path.dirname(args.vectors)
    progress_path = os.path.join(data_dir, 'reprocess_progress.json')
    track_ids_path = os.path.join(data_dir, 'track_ids.json')

    # If persistent volume exists, also save vectors there
    if os.path.isdir('/app/data'):
        vectors_in_data = os.path.join(data_dir, 'vectors_vinet.csv')
        if not os.path.exists(vectors_in_data) and os.path.exists(args.vectors):
            print(f"Copying vectors to persistent storage: {vectors_in_data}")
            import shutil
            shutil.copy(args.vectors, vectors_in_data)
        args.vectors = vectors_in_data

    # Load progress (for resumability)
    progress = load_progress(progress_path)
    processed_videos = set(progress['processed_videos'])
    stats = progress['stats']

    if processed_videos:
        print(f"Resuming from previous run: {len(processed_videos)} videos already processed")
        print(f"  Previous stats: {stats['groups_processed']} groups, {stats['collisions_fixed']} fixed")

    # Load embeddings
    embeddings = load_vectors(args.vectors)

    # Find collisions (excluding already processed videos)
    collisions = find_collisions(embeddings, processed_videos)
    total_collision_songs = sum(len(vids) for vids in collisions.values())
    print(f"Found {len(collisions)} collision groups with {total_collision_songs} songs remaining")

    if not collisions:
        print("No collisions to fix!")
        return

    # Initialize model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if not os.path.isabs(config['MODEL']['CHECKPOINT_PATH']):
        config['MODEL']['CHECKPOINT_PATH'] = os.path.join(
            SCRIPT_DIR, config['MODEL']['CHECKPOINT_PATH']
        )
    model = load_model(config, device)
    print(f"Model loaded on {device}")

    os.makedirs(args.tmp_dir, exist_ok=True)

    # Load existing trackIds if available
    track_ids = {}
    if os.path.exists(track_ids_path):
        with open(track_ids_path, 'r') as f:
            track_ids = json.load(f)
        print(f"Loaded {len(track_ids)} existing trackId mappings")

    # Process collision groups
    collision_list = list(collisions.items())
    if args.limit > 0:
        collision_list = collision_list[:args.limit]

    groups_this_run = 0
    for group_idx, (hash_val, vids) in enumerate(collision_list):
        print(f"\n[{group_idx+1}/{len(collision_list)}] Processing collision group {hash_val} ({len(vids)} songs)")

        new_embeddings = {}
        new_track_ids = {}

        for vid in vids:
            print(f"  {vid}...", end=' ', flush=True)
            new_emb, track_id, error = recompute_with_smart_match(vid, model, device, args.tmp_dir)

            if error:
                print(f"FAILED ({error})")
                stats['failed'][vid] = error
                new_embeddings[vid] = embeddings[vid]  # Keep original
            else:
                new_hash = compute_hash(new_emb)
                print(f"OK (trackId={track_id}, hash={new_hash})")
                new_embeddings[vid] = new_emb
                new_track_ids[vid] = track_id
                # Save trackId to global mapping
                if track_id:
                    track_ids[vid] = track_id

            # Mark as processed
            processed_videos.add(vid)

        # Check if collisions are fixed
        new_hashes = {vid: compute_hash(emb) for vid, emb in new_embeddings.items()}
        unique_hashes = set(new_hashes.values())

        if len(unique_hashes) == len(vids):
            print(f"  ✓ FIXED - all {len(vids)} songs now have unique hashes")
            stats['collisions_fixed'] += 1
        else:
            remaining = len(vids) - len(unique_hashes)
            print(f"  ⚠ Still {remaining} collision(s) in this group")
            stats['collisions_remaining'] += 1

        # Update embeddings
        for vid, emb in new_embeddings.items():
            if not np.array_equal(embeddings.get(vid), emb):
                stats['songs_updated'] += 1
            embeddings[vid] = emb

        stats['groups_processed'] += 1
        groups_this_run += 1

        # Log progress and save every N groups
        if groups_this_run % args.log_interval == 0 and not args.dry_run:
            fix_rate = stats['collisions_fixed'] / stats['groups_processed'] * 100 if stats['groups_processed'] > 0 else 0
            print(f"\n>>> PROGRESS: {stats['groups_processed']} total groups ({fix_rate:.1f}% fixed, {len(stats['failed'])} failed)")
            save_vectors(embeddings, args.vectors)
            with open(track_ids_path, 'w') as f:
                json.dump(track_ids, f)
            save_progress(progress_path, processed_videos, stats)
            notify_api_reload(args.api_url)

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Groups processed (total): {stats['groups_processed']}")
    print(f"Groups processed (this run): {groups_this_run}")
    print(f"Songs updated: {stats['songs_updated']}")
    print(f"Collisions fixed: {stats['collisions_fixed']}")
    print(f"Collisions remaining: {stats['collisions_remaining']}")
    print(f"Failed songs: {len(stats['failed'])}")

    if not args.dry_run:
        save_vectors(embeddings, args.vectors)
        with open(track_ids_path, 'w') as f:
            json.dump(track_ids, f)
        save_progress(progress_path, processed_videos, stats)
        print(f"\nVectors saved to {args.vectors}")
        print(f"TrackIds saved to {track_ids_path} ({len(track_ids)} mappings)")
        print(f"Progress saved to {progress_path} ({len(processed_videos)} videos processed)")
        notify_api_reload(args.api_url)
    else:
        print("\n(Dry run - no files modified)")


if __name__ == '__main__':
    main()
