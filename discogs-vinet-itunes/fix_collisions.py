#!/usr/bin/env python3
"""
Fix hash collisions by re-fetching iTunes previews and recomputing embeddings.
Songs that still collide after retry are removed from the database.

Usage:
    python fix_collisions.py [--dry-run]
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from pipeline import (
    get_youtube_title,
    parse_artist_track,
    search_itunes,
    download_preview,
    convert_to_wav,
    compute_cqt_features,
    compute_embedding,
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


def find_collisions(embeddings):
    """Find groups of songs with same hash."""
    hash_to_vids = {}
    for vid, emb in embeddings.items():
        h = compute_hash(emb)
        if h not in hash_to_vids:
            hash_to_vids[h] = []
        hash_to_vids[h].append(vid)

    # Return only groups with more than 1 song
    collisions = {h: vids for h, vids in hash_to_vids.items() if len(vids) > 1}
    return collisions


def recompute_embedding(vid, model, device):
    """Re-fetch iTunes and recompute embedding for a video."""
    # Get YouTube title
    title = get_youtube_title(vid)
    if not title:
        return None, "oembed_fail"

    # Parse artist/track
    artist, track = parse_artist_track(title)
    if not track:
        return None, "parse_fail"

    # Search iTunes
    time.sleep(0.3)  # Rate limiting
    preview_url, _, _ = search_itunes(artist, track)
    if not preview_url:
        return None, "itunes_fail"

    # Download and process
    with tempfile.TemporaryDirectory() as tmpdir:
        m4a_path = os.path.join(tmpdir, 'preview.m4a')
        wav_path = os.path.join(tmpdir, 'preview.wav')

        if not download_preview(preview_url, m4a_path):
            return None, "download_fail"

        if not convert_to_wav(m4a_path, wav_path):
            return None, "convert_fail"

        cqt = compute_cqt_features(wav_path)
        if cqt is None:
            return None, "cqt_fail"

        embedding = compute_embedding(model, cqt, device)
        return embedding, None


def save_vectors(embeddings, csv_path):
    """Save embeddings to CSV."""
    print(f"Saving {len(embeddings)} embeddings to {csv_path}...")
    with open(csv_path, 'w') as f:
        f.write("youtube_id,embeddings\n")
        for vid, emb in embeddings.items():
            emb_str = ' '.join(f'{x:.6f}' for x in emb)
            f.write(f'{vid},"[ {emb_str} ]"\n')
    print("Saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Do not modify files')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of collision groups to process (0=all)')
    args = parser.parse_args()

    vectors_path = os.path.join(SCRIPT_DIR, 'vectors_vinet.csv')
    progress_path = os.path.join(SCRIPT_DIR, 'fix_collisions_progress.json')

    # Load embeddings
    embeddings = load_vectors(vectors_path)

    # Find collisions
    collisions = find_collisions(embeddings)
    total_collision_songs = sum(len(vids) for vids in collisions.values())
    print(f"Found {len(collisions)} collision groups with {total_collision_songs} songs")

    # Load progress
    progress = {'processed': [], 'removed': [], 'updated': [], 'failed': {}}
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            progress = json.load(f)
        print(f"Resuming: {len(progress['processed'])} groups already processed")

    # Initialize model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_path = os.path.join(SCRIPT_DIR, 'checkpoint', 'config.yaml')
    checkpoint_path = os.path.join(SCRIPT_DIR, 'checkpoint', 'model_checkpoint.pth')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['MODEL']['CHECKPOINT_PATH'] = checkpoint_path
    model = load_model(config, device)
    print(f"Model loaded on {device}")

    # Process collision groups
    processed_count = 0
    for hash_val, vids in collisions.items():
        if hash_val in progress['processed']:
            continue

        if args.limit > 0 and processed_count >= args.limit:
            break

        print(f"\n[{processed_count+1}] Processing collision group {hash_val} with {len(vids)} songs...")

        new_embeddings = {}
        for vid in vids:
            print(f"  Recomputing {vid}...", end=' ')
            new_emb, error = recompute_embedding(vid, model, device)

            if error:
                print(f"FAILED ({error})")
                progress['failed'][vid] = error
                # Keep original embedding
                new_embeddings[vid] = embeddings[vid]
            else:
                new_hash = compute_hash(new_emb)
                print(f"OK (new hash: {new_hash})")
                new_embeddings[vid] = new_emb

        # Check for remaining collisions in this group
        new_hashes = {vid: compute_hash(emb) for vid, emb in new_embeddings.items()}
        hash_counts = {}
        for vid, h in new_hashes.items():
            if h not in hash_counts:
                hash_counts[h] = []
            hash_counts[h].append(vid)

        # Remove songs that still collide (keep first one in each collision)
        for h, collision_vids in hash_counts.items():
            if len(collision_vids) > 1:
                # Keep the first, remove the rest
                keep_vid = collision_vids[0]
                for vid in collision_vids[1:]:
                    print(f"  Removing {vid} (still collides with {keep_vid})")
                    progress['removed'].append(vid)
                    if vid in embeddings:
                        del embeddings[vid]

        # Update embeddings with new values
        for vid, emb in new_embeddings.items():
            if vid not in progress['removed']:
                if not np.array_equal(embeddings.get(vid), emb):
                    progress['updated'].append(vid)
                embeddings[vid] = emb

        progress['processed'].append(hash_val)
        processed_count += 1

        # Save progress periodically
        if processed_count % 10 == 0:
            with open(progress_path, 'w') as f:
                json.dump(progress, f, indent=2)
            print(f"\nProgress saved. Removed: {len(progress['removed'])}, Updated: {len(progress['updated'])}")

            if not args.dry_run:
                save_vectors(embeddings, vectors_path)

    # Final save
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)

    print(f"\n=== DONE ===")
    print(f"Processed: {len(progress['processed'])} collision groups")
    print(f"Removed: {len(progress['removed'])} songs")
    print(f"Updated: {len(progress['updated'])} embeddings")
    print(f"Failed: {len(progress['failed'])} songs")

    if not args.dry_run:
        save_vectors(embeddings, vectors_path)
    else:
        print("(Dry run - no files modified)")


if __name__ == '__main__':
    main()
