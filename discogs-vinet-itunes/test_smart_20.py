#!/usr/bin/env python3
"""
Test smart matching on 20 songs with collisions.
Compare old vs new matching to see if collisions are fixed.
"""

import hashlib
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from pipeline import (
    search_itunes,
    smart_search_itunes,
    parse_artist_track,
    get_youtube_title,
    get_youtube_info,
)


def compute_hash(emb):
    return hashlib.md5(emb.astype(np.float32).tobytes()).hexdigest()[:8]


def load_collision_groups(limit=10):
    """Load collision groups (songs with same hash)."""
    embeddings = {}
    with open(os.path.join(SCRIPT_DIR, 'vectors_vinet.csv')) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                vid = parts[0]
                vec_str = parts[1].strip('"').strip('[]')
                try:
                    vec = np.array([float(x) for x in vec_str.split()])
                    if len(vec) == 512:
                        embeddings[vid] = vec
                except:
                    pass

    hash_to_vids = {}
    for vid, emb in embeddings.items():
        h = compute_hash(emb)
        if h not in hash_to_vids:
            hash_to_vids[h] = []
        hash_to_vids[h].append(vid)

    # Get groups with 2+ songs
    groups = [(h, vids) for h, vids in hash_to_vids.items() if len(vids) >= 2]
    return groups[:limit]


def main():
    print("Loading collision groups...")
    groups = load_collision_groups(limit=10)
    print(f"Testing {len(groups)} collision groups\n")

    old_collisions = 0
    new_collisions = 0
    fixed = 0

    for hash_val, vids in groups:
        print(f"{'='*60}")
        print(f"Collision group {hash_val} ({len(vids)} songs)")
        print('='*60)

        old_track_ids = set()
        new_track_ids = set()

        for vid in vids[:3]:  # Test first 3 in each group
            info = get_youtube_info(vid)
            if not info:
                print(f"  {vid}: Could not get YouTube info")
                continue

            print(f"\n  {vid}")
            print(f"  Title: {info['title']}")
            print(f"  Channel: {info['channel']}")

            # Old method
            artist, track = parse_artist_track(info['title'])
            time.sleep(0.2)
            old_preview, old_artist, old_track = search_itunes(artist, track)

            # New method
            time.sleep(0.2)
            new_preview, new_artist, new_track, new_track_id = smart_search_itunes(vid)

            print(f"  OLD: {old_artist} - {old_track}")
            print(f"  NEW: {new_artist} - {new_track} (trackId: {new_track_id})")

            if old_preview:
                old_track_ids.add(old_preview[:50])  # Use preview URL prefix as ID
            if new_track_id:
                new_track_ids.add(new_track_id)

        old_has_collision = len(old_track_ids) < len(vids[:3])
        new_has_collision = len(new_track_ids) < len(vids[:3])

        if old_has_collision:
            old_collisions += 1
        if new_has_collision:
            new_collisions += 1
        if old_has_collision and not new_has_collision:
            fixed += 1
            print(f"\n  ✓ FIXED! New method found different tracks.")
        elif not new_has_collision:
            print(f"\n  ✓ No collision with new method.")
        else:
            print(f"\n  ⚠ Still has collision with new method.")

        print()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Groups tested: {len(groups)}")
    print(f"Old method collisions: {old_collisions}")
    print(f"New method collisions: {new_collisions}")
    print(f"Fixed by new method: {fixed}")
    print(f"Improvement: {(old_collisions - new_collisions) / old_collisions * 100:.1f}%" if old_collisions > 0 else "N/A")


if __name__ == '__main__':
    main()
